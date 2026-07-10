import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW

from base_models.base_model import VGGTAdapter,Pi3Adapter,MapAnythingAdapter
from base_models.vggtomega_adapter import VGGTOmegaAdapter

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config
from pathlib import Path

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result


def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""

    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file")

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 

class VGGT_Long:
    def __init__(self, image_dir, save_dir, config, selected_frames=None):
        self.config = config
        self.selected_frames = selected_frames  # STAC patch: optional keyframe subset

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = self.config['Model'].get('mask_sky', True)  # STAC: remove sky from
                                                                    # the reconstruction with
                                                                    # skyseg.onnx. Default ON.
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        elif self.config['Weights']['model'] == 'VGGTOmega':
            self.model = VGGTOmegaAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):

        # STAC patch (resume): if loop_closures.txt already exists, load the pairs and
        # SKIP the DINOv2/SALAD feature extraction (~20 min). Pairs are written as
        # "i, j, sim" lines; "#" lines are headers/the image-path list.
        loop_txt = os.path.join(self.output_dir, "loop_closures.txt")
        if not self.useDBoW and os.path.exists(loop_txt):
            try:
                pairs = []
                with open(loop_txt) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            pairs.append((int(parts[0]), int(parts[1])))
                self.loop_list = pairs
                print(f"[STAC resume] loop_closures.txt found — {len(pairs)} loops loaded, "
                      f"DINOv2 extraction skipped")
                return
            except Exception as _e:
                print(f"[STAC resume] loop_closures.txt unreadable ({_e}) — re-detecting")

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def _stac_mask_sky(self, predictions, chunk_image_paths):
        """STAC: zero per-pixel confidence at sky regions so the cloud builder's
        confidence filter (keeps world_points_conf >= thr) drops sky points. Uses
        VGGT-Long's OWN skyseg.onnx — the same mechanism loop_utils.visual_util applies
        for visualization (`world_points_conf *= non_sky`), wired into the
        reconstruction here. Per-frame masks are cached under <save_dir>/sky_masks/;
        skyseg.onnx is fetched once next to this file. No-op (and harmless) indoors,
        where the segmenter finds no sky. Failures degrade to 'no masking', never crash."""
        if not self.sky_mask:
            return
        wpc = predictions.get('world_points_conf', None)
        if wpc is None or getattr(wpc, 'ndim', 0) != 3:
            return
        try:
            import onnxruntime
            from loop_utils.visual_util import segment_sky, download_file_from_url
        except Exception as _e:
            print(f"[STAC sky] skyseg unavailable ({_e}) — skipping sky mask")
            return
        S, H, W = wpc.shape
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skyseg.onnx")
        if not os.path.exists(onnx_path):
            print("[STAC sky] downloading skyseg.onnx ...")
            try:
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                    onnx_path)
            except Exception as _e:
                print(f"[STAC sky] skyseg.onnx download failed ({_e}) — skipping sky mask")
                return
        sky_dir = os.path.join(os.path.dirname(self.result_unaligned_dir), "sky_masks")
        os.makedirs(sky_dir, exist_ok=True)
        n = 0
        for i, p in enumerate(chunk_image_paths[:S]):
            mask_fp = os.path.join(sky_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            try:
                if os.path.exists(mask_fp):
                    sky = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)  # 255=non-sky, 0=sky
                else:
                    if self.skyseg_session is None:
                        self.skyseg_session = onnxruntime.InferenceSession(onnx_path)
                    sky = segment_sky(p, self.skyseg_session, mask_fp)
                if sky is None:
                    continue
                if sky.shape[0] != H or sky.shape[1] != W:
                    sky = cv2.resize(sky, (W, H), interpolation=cv2.INTER_NEAREST)
                nonsky = (sky > 0.1).astype(wpc.dtype)  # 1=keep (non-sky), 0=drop (sky)
                wpc[i] *= nonsky
                n += 1
            except Exception as _e:
                print(f"[STAC sky] frame {os.path.basename(p)} skip ({_e})")
        predictions['world_points_conf'] = wpc
        print(f"[STAC sky] masked sky on {n}/{S} chunk frames")

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # Resolve the output path FIRST so we can resume from an existing chunk.
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        save_path = os.path.join(save_dir, filename)

        # STAC patch (resume): if this chunk's .npy already exists (prior run), load it
        # and SKIP the expensive inference. Restore the camera state EXACTLY as the
        # inference path does, so SIM3 + alignment + pose saving are unaffected. A
        # corrupt/half-written file falls through and re-infers.
        if os.path.exists(save_path):
            try:
                predictions = np.load(save_path, allow_pickle=True).item()
                if not is_loop and range_2 is None:
                    self.all_camera_poses.append((self.chunk_indices[chunk_idx], predictions['extrinsic']))
                    self.all_camera_intrinsics.append((self.chunk_indices[chunk_idx], predictions['intrinsic']))
                print(f'[STAC resume] {filename} on disk — inference skipped')
                return predictions if is_loop or range_2 is not None else None
            except Exception as _e:
                print(f'[STAC resume] {filename} unreadable ({_e}) — re-inferring')

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        predictions['depth'] = np.squeeze(predictions['depth'])

        # STAC: drop sky points (zero their confidence) before saving, so the cloud
        # builder's confidence filter removes them. Sky detection is MapAnything-side
        # (skyseg.onnx) — DA3 priors are NOT relied on for this.
        self._stac_mask_sky(predictions, chunk_image_paths)

        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None
    
    def _stac_metric_lock(self):
        """STAC patch: lock every unaligned chunk (and loop-bridge prediction) to METRIC
        scale from isolated DA3 anchor depths, before alignment. No-op unless the config
        enables it (Model.metric_lock.enable). Chunks without any anchor inherit the
        median scale of the anchored ones (logged loudly). Writes metric_lock.json."""
        ml = (self.config['Model'].get('metric_lock') or {})
        if not ml.get('enable'):
            return
        from loop_utils.metric_lock import (chunk_scale, chunk_anchor_ratios,
                                            apply_scale, apply_scale_drift,
                                            real_frame_number,
                                            seam_relative_scale, solve_scale_graph,
                                            solve_scale_drift, scale_drift_gate,
                                            chunk_tri_angle, flag_sick_chunks,
                                            flag_suspect_chunks)
        anchor_dir = ml['anchor_dir']
        near_frac = float(ml.get('near_frac', 0.25))
        import json as _json

        # Resume guard: locking multiplies the npys IN PLACE — a resumed run must not
        # scale an already-locked chunk twice. metric_lock.json records what was locked.
        already = set()
        _prev_path = os.path.join(self.output_dir, "metric_lock.json")
        if os.path.exists(_prev_path):
            try:
                already = {int(k) for k, v in _json.load(open(_prev_path))
                           .get("chunks", {}).items() if v.get("s") is not None
                           or v.get("s_applied") is not None}
                print(f"[metric-lock] resume: {len(already)} chunk(s) already locked "
                      f"in a previous run — skipping those")
            except Exception:
                already = set()

        report = {"chunks": {}, "loops": {}, "seams": {}}
        scales = {}
        n_anchor_map = {}
        seam_rel = {}
        anchors_pos = {}     # chunk -> [(u in [0,1], ratio)] positioned DA3 anchors
        seam_obs = {}        # seam k -> [(u_k, u_{k+1}, r)] per-shared-frame scale
        tri_map = {}         # chunk -> triangulation angle (health gate signal 1)
        fx_map = {}          # chunk -> median estimated focal (zoom detector)
        prev_shared = None   # (chunk_idx, {global_frame: depth}) for the seam ratio
        # 1) main chunks: DA3 absolute scale per chunk + RELATIVE scale per seam
        # (same frame, same pixels, in both neighbours — the precise sensor)
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_unaligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                print(f"[metric-lock] chunk {k}: missing {path} — skipped")
                prev_shared = None
                continue
            data = np.load(path, allow_pickle=True).item()
            nums = [real_frame_number(self.img_list[i]) for i in range(start, end)]
            locs, ratios = chunk_anchor_ratios(data, nums, anchor_dir,
                                               near_frac=near_frac)
            s = float(np.median(ratios)) if ratios else None
            n = len(ratios)
            S_k = max(end - start, 1)
            anchors_pos[k] = [(loc / (S_k - 1) if S_k > 1 else 0.5, r)
                              for loc, r in zip(locs, ratios)]
            if s is not None:
                scales[k] = s
                n_anchor_map[k] = n
                spread = (f" spread {min(ratios):.3f}-{max(ratios):.3f}"
                          if len(ratios) > 1 else "")
                print(f"[metric-lock] chunk {k}: DA3 s={s:.4f} from {n} anchor(s){spread}")
            report["chunks"][str(k)] = {"s": s, "n_anchors": n, "ratios": ratios,
                                        "anchor_locals": locs}
            # health gate signal: parallax the chunk actually holds (scale-invariant,
            # so valid for fresh AND resumed/already-locked chunks alike)
            tri_map[k] = chunk_tri_angle(data.get("depth"),
                                         data.get("world_points_conf"),
                                         data.get("extrinsic"))
            if data.get("intrinsic") is not None:
                fx_map[k] = float(np.median(np.asarray(data["intrinsic"])[:, 0, 0]))
            # seam with the previous chunk: ratio over ALL shared frames —
            # kept per frame WITH its position inside both chunks (the drift
            # model needs to know WHERE on each chunk the seam was measured)
            depth_k = np.asarray(data["depth"])
            if prev_shared is not None and prev_shared[0] == k - 1:
                p_start, p_end = self.chunk_indices[k - 1]
                S_prev = max(p_end - p_start, 1)
                rs = []
                for local, g in enumerate(range(start, end)):
                    if g in prev_shared[1]:
                        r = seam_relative_scale(prev_shared[1][g], depth_k[local])
                        if r is not None:
                            rs.append(r)
                            u_prev = ((g - p_start) / (S_prev - 1)
                                      if S_prev > 1 else 0.5)
                            u_cur = ((g - start) / (S_k - 1) if S_k > 1 else 0.5)
                            seam_obs.setdefault(k - 1, []).append(
                                (u_prev, u_cur, r))
                if rs:
                    seam_rel[k - 1] = float(np.median(rs))
                    report["seams"][str(k - 1)] = {"rel": seam_rel[k - 1],
                                                   "n_frames": len(rs)}
                    print(f"[metric-lock] seam {k-1}->{k}: relative scale "
                          f"{seam_rel[k-1]:.4f} over {len(rs)} shared frames")
            # stash THIS chunk's tail frames (the next chunk's overlap) for its seam
            nxt = self.chunk_indices[k + 1] if k + 1 < len(self.chunk_indices) else None
            if nxt is not None:
                shared = {g: depth_k[g - start] for g in range(max(nxt[0], start), end)}
                prev_shared = (k, shared)
            else:
                prev_shared = None

        # 2) fuse both sensors: seams make neighbours CONSISTENT (0.1-1% noise),
        # anchors pin the global metre (±8-15% each). Weighted LS in log space.
        if scales and seam_rel:
            s_opt = solve_scale_graph(scales, n_anchor_map, seam_rel,
                                      len(self.chunk_indices),
                                      sigma_seam=float(ml.get('sigma_seam', 0.003)),
                                      sigma_anchor=float(ml.get('sigma_anchor', 0.08)))
            for k in range(len(self.chunk_indices)):
                if np.isfinite(s_opt[k]) and s_opt[k] > 0:
                    old_s = scales.get(k)
                    scales[k] = float(s_opt[k])
                    report["chunks"].setdefault(str(k), {})["s_graph"] = float(s_opt[k])
                    tag = (f" (DA3 alone said {old_s:.4f})" if old_s
                           else " (no anchors — from seams)")
                    print(f"[metric-lock] chunk {k}: OPTIMAL s={s_opt[k]:.4f}{tag}")
        # merge previously locked chunks into the scale table (report continuity + the
        # fallback median), but NEVER re-apply them.
        if already and os.path.exists(_prev_path):
            try:
                for k_str, v in _json.load(open(_prev_path)).get("chunks", {}).items():
                    s_prev = v.get("s_applied", v.get("s"))
                    if int(k_str) in already and s_prev is not None:
                        scales.setdefault(int(k_str), float(s_prev))
                        report["chunks"].setdefault(k_str, v)
            except Exception:
                pass
        if not scales:
            raise RuntimeError(
                "[metric-lock] NO chunk found any DA3 anchor — cannot lock metric scale. "
                f"anchor_dir={anchor_dir}. The anchors must cover every chunk's frame range.")
        fallback = float(np.median(list(scales.values())))

        # ── SCALE DRIFT (self-gated): a chunk whose internal scale DRIFTS is not
        # one number (test4: anchor ratios spread 48% inside chunk 3; adjacent
        # locked scales jumped 18-29% — the leftover warp is the z-drift on tall
        # structures). Linear log-scale per chunk, positioned anchors + positioned
        # per-frame seam ratios; held-out anchors judge whether the model explains
        # the data — an unearned correction is a warp, not a fix. Resumed runs
        # keep the constant path (mixed already-scaled chunks would poison the
        # seam observations).
        drift_frames = None
        n_chunks_all = len(self.chunk_indices)
        if ml.get('scale_drift', True) and not already and anchors_pos and seam_obs:
            ok, info = scale_drift_gate(anchors_pos, scales, seam_obs, n_chunks_all)
            report["drift"] = dict(info, verdict="APPLY" if ok else "SKIP")
            if ok:
                s0, s1 = solve_scale_drift(anchors_pos, seam_obs, n_chunks_all,
                                           prior=scales)
                drift_frames = {}
                for k, (start, end) in enumerate(self.chunk_indices):
                    S_k = max(end - start, 1)
                    u = np.linspace(0.0, 1.0, S_k) if S_k > 1 else np.array([0.5])
                    drift_frames[k] = np.exp((1 - u) * np.log(s0[k])
                                             + u * np.log(s1[k]))
                    scales[k] = float(np.sqrt(s0[k] * s1[k]))
                    report["chunks"].setdefault(str(k), {})["s0_s1"] = \
                        [float(s0[k]), float(s1[k])]
                    print(f"[metric-lock] chunk {k}: DRIFT s {s0[k]:.4f}→{s1[k]:.4f} "
                          f"({(s1[k]/s0[k]-1)*100:+.1f}% along the chunk)")
                print(f"[metric-lock] scale drift APPLIED — held-out anchor error "
                      f"{info['holdout_const']*100:.2f}% → {info['holdout_drift']*100:.2f}%")
            else:
                print(f"[metric-lock] scale drift SKIPPED (constant scale kept): "
                      f"{info.get('reason', '')}"
                      + (f" held-out {info['holdout_const']*100:.2f}% → "
                         f"{info['holdout_drift']*100:.2f}%, max drift "
                         f"{info['max_drift_log']:.3f} (bound {info['bound_log']:.3f})"
                         if 'holdout_const' in info else ""))
        for k in range(len(self.chunk_indices)):
            if k in already:
                continue
            path = os.path.join(self.result_unaligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                continue
            s = scales.get(k)
            if s is None:
                s = fallback
                print(f"[metric-lock] chunk {k}: NO anchors in range — inheriting the "
                      f"median scale of the anchored chunks (s={s:.4f})")
                report["chunks"][str(k)]["s_applied"] = s
            data = np.load(path, allow_pickle=True).item()
            if drift_frames is not None and k in drift_frames:
                apply_scale_drift(data, drift_frames[k])
            else:
                apply_scale(data, s)
            np.save(path, data)
        # 2) loop-bridge predictions (in memory) — they are Omega passes with their own
        # arbitrary scale; the SE(3) loop legs need them metric too. Anchors inside the
        # bridge's two ranges when available, else the mean of the two parent chunks.
        for li, (item, pred) in enumerate(getattr(self, 'loop_predict_list', []) or []):
            ka, (a0, a1), kb, (b0, b1) = item[0], item[1], item[2], item[3]
            nums = ([real_frame_number(self.img_list[i]) for i in range(a0, a1)]
                    + [real_frame_number(self.img_list[i]) for i in range(b0, b1)])
            s, n, _ = chunk_scale(pred, nums, anchor_dir, near_frac=near_frac)
            if s is None:
                s = float(np.mean([scales.get(ka, fallback), scales.get(kb, fallback)]))
                n = 0
            apply_scale(pred, s)
            print(f"[metric-lock] loop bridge {ka}<->{kb}: s={s:.4f} "
                  f"({n} anchor(s)" + (")" if n else " — parents' mean)"))
            report["loops"][str(li)] = {"a": ka, "b": kb, "s": s, "n_anchors": n}
        _sv = np.array(list(scales.values()))
        print(f"[metric-lock] ✅ {len(scales)}/{len(self.chunk_indices)} chunks anchored; "
              f"scale spread {_sv.min():.3f}-{_sv.max():.3f} (x{_sv.max()/_sv.min():.2f})")
        with open(os.path.join(self.output_dir, "metric_lock.json"), "w") as f:
            _json.dump(report, f, indent=1)

        # ── CHUNK HEALTH GATE (see flag_sick_chunks) ──
        # A chunk whose own numbers say "no 3D information" (rotation-only/far-field
        # parallax) or "no single internal scale" (DA3 anchors disagreeing within it)
        # must not WRITE points — garbage dies at its source. It still participates
        # in the alignment and the scale graph: with 50% overlap it is the only
        # bridge between its neighbours.
        anchor_iqr = {}
        for k_str, v in report["chunks"].items():
            k_int = int(k_str)
            obs = anchors_pos.get(k_int)
            if obs:
                r = np.asarray([ri for _, ri in obs], np.float64)
                if drift_frames is not None and v.get("s0_s1"):
                    # the drift model EXPLAINS part of the spread — judge the
                    # chunk on what remains, not on what was corrected
                    s0k, s1k = v["s0_s1"]
                    pred = np.exp([(1 - u) * np.log(s0k) + u * np.log(s1k)
                                   for u, _ in obs])
                    r = r / pred
            else:
                r = np.asarray(v.get("ratios") or [], np.float64)
            if len(r) > 1 and np.median(r) > 0:
                anchor_iqr[k_int] = float(
                    (np.percentile(r, 75) - np.percentile(r, 25)) / np.median(r))
            else:
                anchor_iqr[k_int] = None
        sick = flag_sick_chunks(tri_map, anchor_iqr, fx_median=fx_map)
        suspect = flag_suspect_chunks(anchor_iqr, sick=sick,
                                      spread_cut=float(ml.get('suspect_spread', 0.30)))
        # resume: chunks whose unaligned npy is already consumed cannot be re-evaluated
        # — inherit their previous verdict so a resumed run never un-flags them.
        _health_path = os.path.join(self.output_dir, "chunk_health.json")
        if os.path.exists(_health_path):
            try:
                _prev_health = _json.load(open(_health_path))
                for k_str, rs in (_prev_health.get("sick") or {}).items():
                    if int(k_str) not in tri_map:
                        sick.setdefault(int(k_str), rs)
                for k_str, rs in (_prev_health.get("suspect") or {}).items():
                    if int(k_str) not in tri_map and int(k_str) not in sick:
                        suspect.setdefault(int(k_str), rs)
            except Exception:
                pass
        self._stac_sick_chunks = set(sick)
        self._stac_suspect_chunks = set(suspect)
        with open(_health_path, "w") as f:
            _json.dump({"tri_angle": {str(k): tri_map.get(k)
                                      for k in range(len(self.chunk_indices))},
                        "anchor_iqr_over_median": {str(k): anchor_iqr.get(k)
                                                   for k in range(len(self.chunk_indices))},
                        "fx_median": {str(k): fx_map.get(k)
                                      for k in range(len(self.chunk_indices))},
                        "sick": {str(k): rs for k, rs in sorted(sick.items())},
                        "suspect": {str(k): rs for k, rs
                                    in sorted(suspect.items())}}, f, indent=1)
        for k in sorted(suspect):
            print(f"[health] chunk {k} SUSPECT: {suspect[k]}")
        if sick:
            for k in sorted(sick):
                for r in sick[k]:
                    print(f"[health] chunk {k} SICK: {r}")
            print(f"[health] ⛔ {len(sick)} chunk(s) EXCLUDED from the cloud "
                  f"({sorted(sick)}) — kept as alignment bridges, see chunk_health.json")
        else:
            print(f"[health] ✅ all {len(self.chunk_indices)} chunks healthy "
                  f"(parallax + anchor coherence)")

    def _stac_conf_threshold(self, confs):
        """STAC patch: the ONE confidence threshold for a chunk's PLY + origins.

        np.mean() over ~50M float32 accumulates in float32, so its last ULPs depend on
        how numpy vectorises the reduction — recomputing it in two places shifted the
        threshold at the 7th significant digit and flipped ~5 boundary points, leaving
        the PLY and {K}_origins.npz off by 5 (CloudComPy then aborts: it cannot inject
        traceability into a size-mismatched cloud). Accumulate in float64 and compute
        it ONCE, then pass the same value to both writers.

        `conf_percentile` (when set) is the web demo's semantics and the default here:
        drop the bottom P% of the VALID points by confidence, keep the rest. A
        mean-relative coef cannot do that — how much it keeps depends on the shape of
        each scene's confidence histogram (measured: mean*0.6 kept 53% of one scan and
        89% of another). Points with conf<=1e-5 are the sky mask, excluded from the
        percentile so P refers to real geometry."""
        ps = self.config['Model']['Pointcloud_Save']
        if not ps.get('use_conf_filter', True):
            return -1.0
        confs = np.asarray(confs).reshape(-1)
        pct = ps.get('conf_percentile')
        if pct is not None:
            valid = confs[confs > 1e-5]
            if valid.size == 0:
                return -1.0
            return float(np.percentile(valid.astype(np.float64), float(pct)))
        return float(np.mean(confs, dtype=np.float64)) * ps['conf_threshold_coef']

    def _stac_owned_confs(self, confs, chunk_idx):
        """FRAME OWNERSHIP: zero the confidence of frames this chunk does not OWN,
        so both the PLY writer and the origins writer drop them with the same mask.
        Every overlap frame used to be written by BOTH chunks — two displaced copies
        of the same pixels in the cloud (the mechanical half of the duplicated
        objects). One frame → one writer (the chunk whose centre is nearest)."""
        # HEALTH GATE: a sick chunk (see _stac_metric_lock) writes NOTHING — its
        # owned frames become a declared hole instead of garbage in the cloud.
        if chunk_idx in getattr(self, '_stac_sick_chunks', ()):
            cf = np.asarray(confs, np.float32).reshape(-1).copy()
            cf[:] = 0.0
            print(f"[health] chunk {chunk_idx}: sick — all frames dropped from the cloud")
            return cf
        if (not self.config['Model'].get('frame_ownership')
                or self.chunk_indices is None or len(self.chunk_indices) <= 1):
            return confs
        from loop_utils.metric_lock import frame_owner
        owner = frame_owner(self.chunk_indices, len(self.img_list))
        start, end = self.chunk_indices[chunk_idx]
        S = end - start
        cf = np.asarray(confs).reshape(S, -1).copy()
        kept = 0
        for local in range(S):
            if owner[start + local] != chunk_idx:
                cf[local] = 0.0
            else:
                kept += 1
        print(f"[frame-owner] chunk {chunk_idx}: writes {kept}/{S} frames "
              f"(the rest belong to neighbours)")
        return cf.reshape(-1)

    def _stac_write_origins(self, chunk_data, K, conf_threshold=None, confs_override=None):
        """STAC patch: write per-point origins (frame_global = REAL frame number,
        pixel_row/col, confidence) for chunk K using the SAME confidence mask that
        save_confident_pointcloud_batch uses for the PLY. `conf_threshold` MUST be the
        exact value handed to that writer (see _stac_conf_threshold) → guaranteed 1:1
        with the PLY points. Saved next to the PLY as {K}_origins.npz."""
        import re as _re
        try:
            wp = chunk_data['world_points']
            if wp.ndim == 5:
                wp = wp[0]
            S, H, W = wp.shape[:3]
            confs = (np.asarray(confs_override).reshape(-1) if confs_override is not None
                     else chunk_data['world_points_conf'].reshape(-1))
            cfs32 = confs.astype(np.float32)
            thr = self._stac_conf_threshold(confs) if conf_threshold is None else conf_threshold
            mask = (cfs32 >= thr) & (cfs32 > 1e-5)           # identical to save_confident
            surviving = np.flatnonzero(mask)
            HW = H * W
            frame_local = surviving // HW
            within = surviving % HW
            pixel_row = (within // W).astype(np.int16)
            pixel_col = (within % W).astype(np.int16)
            # real frame number = numeric stem of each processed frame's filename
            start = self.chunk_indices[K][0]
            real_per_local = np.array(
                [int(_re.search(r'(\d+)', os.path.basename(self.img_list[start + fl])).group(1))
                 if _re.search(r'(\d+)', os.path.basename(self.img_list[start + fl])) else (start + fl)
                 for fl in range(S)], dtype=np.int32)
            frame_global = real_per_local[frame_local]
            np.savez_compressed(
                os.path.join(self.pcd_dir, f"{K}_origins.npz"),
                frame_global=frame_global, pixel_row=pixel_row, pixel_col=pixel_col,
                confidence=cfs32[surviving].astype(np.float32),
                scaled_resolution=np.array([H, W], np.int32))
            print(f"[STAC] wrote {K}_origins.npz ({len(surviving)} pts, 1:1 with PLY)")
        except Exception as _e:
            print(f"[STAC] WARN: could not write {K}_origins.npz: {_e}")

    def _stac_write_chunk_outputs(self, chunk_data, K):
        """PLY + origins for chunk K from its (aligned, possibly elastic-corrected)
        data — ONE ownership mask and ONE conf threshold shared by both writers, so
        the PLY and {K}_origins.npz stay 1:1 by construction. Sick chunks (health
        gate) write NO files at all: an absent chunk is a declared hole, while an
        empty PLY would trip every downstream reader."""
        if K in getattr(self, '_stac_sick_chunks', ()):
            print(f"[health] chunk {K}: sick — PLY/origins NOT written "
                  f"(declared hole, see chunk_health.json)")
            return
        points = chunk_data['world_points'].reshape(-1, 3)
        colors = (chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        confs = self._stac_owned_confs(chunk_data['world_points_conf'].reshape(-1), K)
        drops = getattr(self, '_stac_far_drop', None)
        if drops:
            start, end = self.chunk_indices[K]
            S = end - start
            confs = np.asarray(confs).reshape(S, -1).copy()
            n_drop = 0
            for local in range(S):
                m = drops.get(start + local)
                if m is not None:
                    confs[local][m.reshape(-1)] = 0.0
                    n_drop += int(m.sum())
            confs = confs.reshape(-1)
            if n_drop:
                print(f"[depth-cap] chunk {K}: {n_drop:,} contradicted far points "
                      f"dropped (displaced duplicates of near-observed surfaces)")
        thr = self._stac_conf_threshold(confs)
        save_confident_pointcloud_batch(
            points=points,
            colors=colors,
            confs=confs,
            output_path=os.path.join(self.pcd_dir, f'{K}_pcd.ply'),
            conf_threshold=thr,
            sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio'])
        self._stac_write_origins(chunk_data, K, conf_threshold=thr, confs_override=confs)

    def _stac_elastic_fit_seams(self):
        """Per-shared-frame rigid residual fits over every seam, computed on the
        ALIGNED (not yet elastic-corrected) chunks. Returns (fits, report) where
        fits[j][g] = (R, t) maps chunk j+1's copy of global frame g onto chunk j's
        copy — EXACT pixel-to-pixel correspondences, robust to the intra-frame
        non-rigid noise (IRLS Cauchy)."""
        from loop_utils.metric_lock import robust_rigid
        fits, report_seams = {}, {}
        prev_tail = None      # (k, {g: (points [HW,3] f32, conf [HW] f32)})
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            data = np.load(path, allow_pickle=True).item()
            if data.get('_stac_elastic_applied'):
                raise RuntimeError(
                    f"[elastic] chunk {k} is already elastic-corrected but the fits in "
                    f"elastic_seams.json are missing or stale — refitting on corrected "
                    f"data would be wrong. Delete _tmp_results_aligned + pcd and re-run.")
            wp = np.asarray(data['world_points'])
            if wp.ndim == 5:
                wp = wp[0]
            S = wp.shape[0]
            conf = np.asarray(data['world_points_conf']).reshape(S, -1)
            if prev_tail is not None and prev_tail[0] == k - 1:
                j = k - 1
                sf, rep, before_all, res_all = {}, {}, [], []
                for local, g in enumerate(range(start, end)):
                    if g not in prev_tail[1]:
                        continue
                    p_dst, c_dst = prev_tail[1][g]
                    p_src = wp[local].reshape(-1, 3)
                    ok = (c_dst > 1e-5) & (conf[local] > 1e-5)
                    entry = {"n_valid_px": int(ok.sum())}
                    if ok.any():
                        d = p_dst[ok].astype(np.float64) - p_src[ok].astype(np.float64)
                        entry["before_m"] = float(np.median(np.linalg.norm(d, axis=1)))
                        before_all.append(entry["before_m"])
                    fit = robust_rigid(p_src[ok], p_dst[ok])
                    if fit is not None:
                        R_, t_, res_, n_ = fit
                        sf[g] = (R_, t_)
                        entry.update({"R": R_.tolist(), "t": t_.tolist(),
                                      "residual_m": res_, "n_fit": n_})
                        res_all.append(res_)
                    else:
                        entry["starved"] = True
                    rep[str(g)] = entry
                fits[j] = sf
                report_seams[str(j)] = rep
                bm = float(np.median(before_all)) * 100 if before_all else float("nan")
                rm = float(np.median(res_all)) * 100 if res_all else float("nan")
                print(f"[elastic] seam {j}->{k}: {len(sf)}/{len(rep)} frame fits — "
                      f"copies disagreed {bm:.1f} cm median → per-frame residual "
                      f"{rm:.2f} cm (the non-rigid floor both copies now share)")
            if k + 1 < len(self.chunk_indices):
                nxt0 = self.chunk_indices[k + 1][0]
                prev_tail = (k, {g: (wp[g - start].reshape(-1, 3).astype(np.float32),
                                     conf[g - start].astype(np.float32))
                                 for g in range(max(nxt0, start), end)})
            else:
                prev_tail = None
            del data
        report = {"chunk_indices": [list(ci) for ci in self.chunk_indices],
                  "seams": report_seams}
        return fits, report

    def _stac_elastic_seams(self):
        """STAC patch: per-frame ELASTIC seam consensus — the final stitching stage.

        The per-chunk rigid glue (exact_seam_align) leaves each seam with a ~cm
        NON-rigid residual: every shared frame disagrees between its two chunks by
        its own small rigid offset (measured on test4: 1.7-6.1 cm median per seam
        after the scale graph + rigid glue). The anchoring directive: the same
        pixel of a frame present in two chunks MUST land at the same 3D position.
        So per shared frame the exact-correspondence rigid residual T_g between the
        two aligned copies is fitted, and BOTH copies move onto a consensus
        interpolated across the overlap — identity at each chunk's centre (see
        loop_utils.metric_lock.elastic_corrections). Camera poses follow their
        frames (save_camera_poses applies the same per-frame moves), so per-frame
        depth and the TSDF stay consistent by rigidity.

        Resume safety: fits are computed BEFORE any chunk is modified and persisted
        to elastic_seams.json; every corrected chunk npy carries a
        '_stac_elastic_applied' stamp so a resumed run never double-applies."""
        if not self.config['Model'].get('elastic_seam') or len(self.chunk_indices) < 2:
            return
        import json as _json
        from loop_utils.metric_lock import elastic_corrections

        seams_path = os.path.join(self.output_dir, "elastic_seams.json")
        fits = None
        if os.path.exists(seams_path):
            try:
                prev = _json.load(open(seams_path))
                if prev.get("chunk_indices") == [list(ci) for ci in self.chunk_indices]:
                    fits = {int(j): {int(g): (np.asarray(v["R"], np.float64),
                                              np.asarray(v["t"], np.float64))
                                     for g, v in d.items() if "R" in v}
                            for j, d in prev.get("seams", {}).items()}
                    print(f"[elastic] resume: fits loaded from elastic_seams.json "
                          f"({sum(len(d) for d in fits.values())} frame fits)")
                else:
                    print("[elastic] elastic_seams.json belongs to another chunk plan "
                          "— refitting")
            except Exception as _e:
                print(f"[elastic] elastic_seams.json unreadable ({_e}) — refitting")
        if fits is None:
            fits, report = self._stac_elastic_fit_seams()
            with open(seams_path, "w") as f:
                _json.dump(report, f, indent=1)

        # apply: every chunk gets its per-frame field. Both sides of every seam land
        # on the SAME consensus, so the fused cloud (one writer per frame) is
        # continuous through the ownership switch in the middle of each overlap.
        self._stac_elastic_corr = {}
        _sick = getattr(self, '_stac_sick_chunks', set())
        _suspect = getattr(self, '_stac_suspect_chunks', set())
        for k in range(len(self.chunk_indices)):
            # sick chunks: healthy neighbours never bend toward them (alpha pinned
            # inside elastic_corrections); their own npy still gets corrected —
            # harmless, and it keeps every seam's two copies coincident — but they
            # write no outputs later (declared hole). Suspect chunks (soft tier):
            # the consensus is BIASED toward the trusted side, not pinned.
            corr = elastic_corrections(self.chunk_indices, k, fits, sick=_sick,
                                       suspect=_suspect)
            self._stac_elastic_corr[k] = corr
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            data = np.load(path, allow_pickle=True).item()
            if data.get('_stac_elastic_applied'):
                print(f"[elastic] chunk {k}: already corrected — skipped")
                continue
            wp = np.asarray(data['world_points'])
            lead = wp.ndim == 5               # (1,S,H,W,3) — same guard as the origins writer
            if lead:
                wp = wp[0]
            moved = 0
            for local in range(wp.shape[0]):
                M = corr[local]
                if np.allclose(M, np.eye(4), atol=1e-12):
                    continue
                p = wp[local].reshape(-1, 3).astype(np.float64)
                wp[local] = (p @ M[:3, :3].T + M[:3, 3]).reshape(wp[local].shape).astype(wp.dtype)
                moved += 1
            data['world_points'] = wp[None] if lead else wp
            data['_stac_elastic_applied'] = True
            np.save(path, data)
            dmax = float(np.max(np.linalg.norm(corr[:, :3, 3], axis=1)))
            print(f"[elastic] chunk {k}: {moved}/{wp.shape[0]} frames moved "
                  f"(max frame translation {dmax * 100:.1f} cm)")
        print(f"[elastic] ✅ all {len(self.chunk_indices)} chunks on the per-frame "
              f"seam consensus — shared pixels now share ONE 3D position")

    def _stac_aligned_pose(self, k, local, ext_c2w):
        """World-space c2w of chunk k's local frame as the CLOUD sees it: the
        metric-locked npy extrinsic composed with the chunk's accumulated Sim3 and
        the per-frame elastic correction — the same composition save_camera_poses
        writes. `ext_c2w` is the 4x4 from the ALIGNED npy."""
        M = np.asarray(ext_c2w, np.float64)
        if k > 0:
            s, R, t = self.sim3_list[k - 1]
            S = np.eye(4)
            S[:3, :3] = float(s) * np.asarray(R)
            S[:3, 3] = np.asarray(t)
            M = S @ M
            M[:3, :3] /= float(s)
        ecorr = getattr(self, '_stac_elastic_corr', None)
        if ecorr is not None:
            M = ecorr[k][local] @ M
        return M

    def _stac_depth_graph(self):
        """STAC patch: per-frame DEPTH GRAPH — kills the depth duplication.

        Measured on test4 after the elastic stage: two frames of the SAME chunk
        10 apart disagree ~1.5% on the depth of the same surface (Omega's internal
        multi-view consistency floor), and crossing a chunk boundary doubles it
        (2.6-4.5%) — 9 cm at 3 m, 18-35 cm at 8-15 m: the objects duplicated IN
        DEPTH the user sees, invisible in height where everything is near. The
        elastic stage cannot touch this: it glues two copies of the SAME frame,
        while this is DIFFERENT frames looking at the same surface.

        Fix, purely geometric (loop_utils.metric_lock): for nearby frame pairs,
        project one frame's points into the other and robustly fit the affine
        relation between the two depth readings of the same surface; solve one
        global least squares for per-frame corrections z' = a_f z + b_f (gauge:
        mean 0 — the global metre stays where the metric lock + scale_align put
        it); move every frame's points ALONG THEIR RAYS (pixels and cameras do
        not move, so poses, origins traceability and the TSDF depth all stay
        valid by construction). Sick chunks are excluded from both measurement
        and correction. Resume-safe via depth_graph.json + per-npy stamp."""
        if not self.config['Model'].get('depth_graph') or len(self.chunk_indices) < 2:
            return
        import json as _json
        from loop_utils.metric_lock import (depth_pair_samples, pair_depth_relation,
                                            solve_depth_graph, apply_depth_correction,
                                            frame_owner)
        _sick = getattr(self, '_stac_sick_chunks', set())
        N = len(self.img_list)
        owner = frame_owner(self.chunk_indices, N)
        sick_frames = {g for g in range(N) if owner[g] in _sick}

        dg_path = os.path.join(self.output_dir, "depth_graph.json")
        sol = None
        if os.path.exists(dg_path):
            try:
                prev = _json.load(open(dg_path))
                if prev.get("chunk_indices") == [list(ci) for ci in self.chunk_indices]:
                    sol = (np.asarray(prev["a"], np.float64), np.asarray(prev["b"], np.float64))
                    print(f"[depth-graph] resume: solution loaded from depth_graph.json")
            except Exception as _e:
                print(f"[depth-graph] depth_graph.json unreadable ({_e}) — remeasuring")

        if sol is None:
            # per-frame owned view: points+conf+pose+K of each frame's canonical copy
            cache = {}
            for k, (start, end) in enumerate(self.chunk_indices):
                data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                               allow_pickle=True).item()
                if data.get('_stac_depth_graph_applied'):
                    raise RuntimeError(
                        f"[depth-graph] chunk {k} already depth-corrected but "
                        f"depth_graph.json is missing/stale — remeasuring on corrected "
                        f"data would be wrong. Delete _tmp_results_aligned + pcd and re-run.")
                wp = np.asarray(data['world_points'])
                if wp.ndim == 5:
                    wp = wp[0]
                cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
                ext = np.asarray(data['extrinsic'])
                K = np.asarray(data['intrinsic'])
                for local, g in enumerate(range(start, end)):
                    if owner[g] == k and g not in sick_frames:
                        c2w = self._stac_aligned_pose(k, local, ext[local])
                        cache[g] = (wp[local].astype(np.float32), cf[local].astype(np.float32),
                                    np.linalg.inv(c2w), K[local])
                del data
            # pairs up to one chunk length apart — the measured range where two
            # frames still share enough surface (beyond it samples starve anyway).
            # HELD-OUT offsets are measured but NEVER fitted: they are the honest
            # judge of whether the per-frame model actually explains the data.
            fit_offsets = (1, 2, 3, 5, 8, 12)
            holdout_offsets = (4, 10)
            meas, held, before_all = [], [], []
            for f in sorted(cache):
                for d in fit_offsets + holdout_offsets:
                    g = f + d
                    if g not in cache:
                        continue
                    zs = depth_pair_samples(cache[f][0], cache[f][1],
                                            cache[g][0], cache[g][1],
                                            cache[g][2], cache[g][3])
                    if zs is None:
                        continue
                    rel = pair_depth_relation(zs[0], zs[1])
                    if rel is None:
                        continue
                    al, be, before, n = rel
                    (meas if d in fit_offsets else held).append((f, g, al, be))
                    if d in fit_offsets:
                        before_all.append(before)
            del cache
            if len(meas) < N // 4 or len(held) < N // 8:
                print(f"[depth-graph] too thin ({len(meas)} fit / {len(held)} held-out "
                      f"pairs for {N} frames) — SKIPPING (nothing modified)")
                return
            # ── SELF-VALIDATION GATE + MODEL LADDER — the stage must EARN the
            # right to touch the geometry (held-out pairs judge; corrections must
            # stay within 5x the pairwise signal). Two rungs, most expressive
            # first:
            #   1. AFFINE (a_f, b_f): full model. test4 2026-07-10: after the
            #      scale drift it finally IMPROVED held-out (1.22%→0.34%) but the
            #      free offset ran to ±112 cm — b is where an unbounded
            #      low-frequency warp hides, so the rung failed bounded.
            #   2. SCALE-ONLY (a_f, b=0): the physically-motivated fallback —
            #      after metric lock + scale drift the residual disagreement is
            #      mostly multiplicative. Scaling depth about the camera with a
            #      bounded a_f cannot produce the offset warp.
            # A rung applies only if it is BOTH bounded and improving; otherwise
            # try the next; no rung → geometry untouched.
            from loop_utils.metric_lock import depth_graph_verdict
            print(f"[depth-graph] {len(meas)} fit + {len(held)} held-out pairs — "
                  f"disagreement before {np.median(before_all) * 100:.2f}% median")
            ladder = []
            a = np.ones(N)
            b = np.zeros(N)
            verdict, model = "SKIP", None
            for rung, kw in (("affine", {}), ("scale-only", {"scale_only": True})):
                a_r, b_r = solve_depth_graph(meas, N, sick_frames=sick_frames, **kw)
                v = depth_graph_verdict(a_r, b_r, meas, held)
                ladder.append(dict(v, model=rung))
                print(f"[depth-graph] {rung}: held-out {v['med_before'] * 100:.2f}% -> "
                      f"{v['med_after'] * 100:.2f}% | a[{a_r.min():.4f},{a_r.max():.4f}] "
                      f"b[{b_r.min() * 100:.1f},{b_r.max() * 100:.1f}]cm | "
                      f"bounded={v['bounded']} improves={v['improves']}")
                if v["bounded"] and v["improves"]:
                    a, b = a_r, b_r
                    verdict, model = "APPLY", rung
                    break
            if verdict == "APPLY":
                print(f"[depth-graph] ✅ {model} model earned the correction")
            else:
                print(f"[depth-graph] ⛔ no rung of the model ladder explains this "
                      f"scan's depth disagreement — geometry left UNTOUCHED (an "
                      f"unearned correction is a warp, not a fix). See depth_graph.json.")
            with open(dg_path, "w") as f_:
                _json.dump({"chunk_indices": [list(ci) for ci in self.chunk_indices],
                            "verdict": verdict, "model": model,
                            "ladder": ladder,
                            "a": a.tolist(), "b": b.tolist(),
                            "n_pairs_fit": len(meas), "n_pairs_holdout": len(held),
                            "pair_disagreement_before_pct": float(np.median(before_all) * 100)},
                           f_, indent=1)
            sol = (a, b)

        a, b = sol
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            data = np.load(path, allow_pickle=True).item()
            if data.get('_stac_depth_graph_applied'):
                print(f"[depth-graph] chunk {k}: already applied — skipped")
                continue
            wp = np.asarray(data['world_points'])
            lead = wp.ndim == 5
            if lead:
                wp = wp[0]
            dep = np.asarray(data['depth'])
            ext = np.asarray(data['extrinsic'])
            moved = 0
            for local, g in enumerate(range(start, end)):
                if g in sick_frames or (abs(a[g] - 1.0) < 1e-9 and abs(b[g]) < 1e-12):
                    continue
                c2w = self._stac_aligned_pose(k, local, ext[local])
                wp[local], dep[local] = apply_depth_correction(
                    wp[local], dep[local], c2w[:3, 3], a[g], b[g])
                moved += 1
            data['world_points'] = wp[None] if lead else wp
            data['depth'] = dep
            data['_stac_depth_graph_applied'] = True
            np.save(path, data)
            print(f"[depth-graph] chunk {k}: {moved}/{end - start} frames re-depthed")
        print(f"[depth-graph] ✅ every frame now agrees with its neighbours on the "
              f"depth of shared surfaces (per-frame z' = a*z + b along rays)")

    def _stac_blend_copies(self):
        """STAC patch: TWO-COPY CONSENSUS — every overlap frame is predicted by
        both of its chunks; instead of discarding the non-owner copy, both copies
        become their per-pixel mean (see loop_utils.metric_lock.blend_two_copies;
        measured on test4: cross-owner depth disagreement 1.51% -> 1.01%, the
        ownership-switch step halves; same-owner pairs unchanged). Runs AFTER the
        elastic consensus (copies rigidly coincide) and depth graph, BEFORE the
        outputs. Seams touching a sick chunk are skipped — a healthy field never
        averages with garbage. Both copies are written back, so every downstream
        reader (PLY via ownership, TSDF, omega-depth) sees the same consensus;
        the operation is idempotent (blending identical copies is a no-op), so
        resume needs no special casing beyond the skip stamp."""
        if not self.config['Model'].get('blend_copies') or len(self.chunk_indices) < 2:
            return
        from loop_utils.metric_lock import blend_two_copies
        _sick = getattr(self, '_stac_sick_chunks', set())
        prev = None            # (k, data, dirty)
        for k in range(len(self.chunk_indices) - 1):
            if k in _sick or (k + 1) in _sick:
                print(f"[blend] seam {k}->{k + 1}: touches a sick chunk — skipped")
                continue
            if prev is not None and prev[0] == k:
                data_a, dirty_a = prev[1], prev[2]
            else:
                if prev is not None and prev[2]:
                    np.save(os.path.join(self.result_aligned_dir, f"chunk_{prev[0]}.npy"),
                            prev[1])
                data_a = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                                 allow_pickle=True).item()
                dirty_a = False
            data_b = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k + 1}.npy"),
                             allow_pickle=True).item()
            if data_a.get('_stac_copies_blended') and data_b.get('_stac_copies_blended'):
                print(f"[blend] seam {k}->{k + 1}: already blended — skipped")
                prev = (k + 1, data_b, False)
                continue
            sa, ea = self.chunk_indices[k]
            sb, eb = self.chunk_indices[k + 1]
            wa = np.asarray(data_a['world_points']); la = wa.ndim == 5
            if la:
                wa = wa[0]
            wb = np.asarray(data_b['world_points']); lb = wb.ndim == 5
            if lb:
                wb = wb[0]
            ca = np.asarray(data_a['world_points_conf']).reshape(wa.shape[:3])
            cb = np.asarray(data_b['world_points_conf']).reshape(wb.shape[:3])
            da = np.asarray(data_a['depth'])
            db = np.asarray(data_b['depth'])
            n_bl = 0
            for g in range(max(sb, sa), min(ea, eb)):
                ia, ib = g - sa, g - sb
                wp, cf, dd = blend_two_copies(wa[ia], ca[ia], wb[ib], cb[ib],
                                              da[ia], db[ib])
                wa[ia] = wp; wb[ib] = wp
                ca[ia] = cf; cb[ib] = cf
                da[ia] = dd; db[ib] = dd
                n_bl += 1
            data_a['world_points'] = wa[None] if la else wa
            data_b['world_points'] = wb[None] if lb else wb
            data_a['world_points_conf'] = ca.reshape(np.asarray(data_a['world_points_conf']).shape)
            data_b['world_points_conf'] = cb.reshape(np.asarray(data_b['world_points_conf']).shape)
            data_a['depth'] = da
            data_b['depth'] = db
            data_a['_stac_copies_blended'] = True
            data_b['_stac_copies_blended'] = True
            np.save(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"), data_a)
            print(f"[blend] seam {k}->{k + 1}: {n_bl} shared frames -> two-copy consensus")
            prev = (k + 1, data_b, True)
        if prev is not None and prev[2]:
            np.save(os.path.join(self.result_aligned_dir, f"chunk_{prev[0]}.npy"), prev[1])

    def _stac_write_depth_cap(self):
        """OBSERVATION-DISTANCE write policy: a surface observed from afar carries
        a depth error PROPORTIONAL to the distance (measured: ~0.7-1.5% per frame
        pair), so far observations of a zone later seen up close write a displaced
        DUPLICATE of it (measured on test4: the cone-gallery zone written by chunk
        4 from 15-25 m sat +0.5..+6 m off the copies chunks 5-6 wrote from up
        close — seams were fine, ownership/blend don't apply: different frames
        legitimately see the same zone). Points whose expected error exceeds what
        the cloud can hold add garbage, not coverage.

        The cap is derived from THIS session's own numbers — no magic constants:
            cap = (median elastic per-frame seam residual)  <- the cloud's floor
                  / (median pairwise depth error rate)      <- error per metre
        Both are already measured and persisted by the elastic and depth-graph
        stages. Config Model.max_write_depth_m overrides explicitly. Returns the
        cap in metres, or None (no policy) when the inputs are unavailable."""
        import json as _json
        explicit = self.config['Model'].get('max_write_depth_m')
        if explicit:
            print(f"[depth-cap] explicit Model.max_write_depth_m = {float(explicit):.1f} m")
            return float(explicit)
        try:
            es = _json.load(open(os.path.join(self.output_dir, "elastic_seams.json")))
            floors = [v["residual_m"] for d in (es.get("seams") or {}).values()
                      for v in d.values() if "residual_m" in v]
            dg = _json.load(open(os.path.join(self.output_dir, "depth_graph.json")))
            rate = float(dg.get("pair_disagreement_before_pct", 0.0)) / 100.0
        except Exception as _e:
            print(f"[depth-cap] session error stats unavailable ({_e}) — "
                  f"no observation-distance policy this run")
            return None
        if not floors or rate <= 0:
            print("[depth-cap] no seam floor / error rate measured — policy off")
            return None
        floor = float(np.median(floors))
        cap = floor / rate
        self._stac_cap_stats = (floor, rate)
        print(f"[depth-cap] session floor {floor * 100:.1f} cm / error rate "
              f"{rate * 100:.2f}%/m of depth → error budget at {cap:.1f} m; far "
              f"points drop ONLY when a nearer frame CONTRADICTS them")
        with open(os.path.join(self.output_dir, "write_depth_cap.json"), "w") as f:
            _json.dump({"cap_m": cap, "seam_floor_m": floor,
                        "pair_error_rate_pct": rate * 100,
                        "source": "median elastic per-frame residual / median "
                                  "pairwise depth disagreement"}, f, indent=1)
        return cap

    def _stac_far_contradictions(self):
        """CONTRADICTION-based far-point policy (v2 of the observation-distance
        cap). v1 dropped every point observed beyond the error budget — measured
        on test4 it destroyed 87% REAL coverage (narrow ~38° FOV: side/elevated
        structures never get a near pass while in frame; only 13% of the dropped
        volume was actual duplication). v2 keeps unique coverage: a far point
        drops ONLY if some frame that was close enough to it (within the budget)
        looked at that spot and saw a different surface — the far observation is
        then a displaced duplicate (the cone gallery: chunk 4's 15-25 m view sat
        +0.5..+6 m off the surfaces chunks 5-6 nailed from up close) or a
        free-space violation. Corroborated points (a near frame AGREES within the
        session tolerance) and unseen points are kept.

        Returns {global_frame: bool HxW drop-mask} for owner frames; {} when the
        session error stats are unavailable."""
        cap = getattr(self, '_stac_max_write_depth', None)
        stats = getattr(self, '_stac_cap_stats', None)
        if not cap or not stats or len(self.chunk_indices) < 2:
            return {}
        from loop_utils.metric_lock import frame_owner, classify_far_points
        floor, rate = stats
        owner = frame_owner(self.chunk_indices, len(self.img_list))
        _sick = getattr(self, '_stac_sick_chunks', set())
        cache = {}          # g -> (wp, conf, depth, w2c, K, cam)
        for k, (start, end) in enumerate(self.chunk_indices):
            if k in _sick:
                continue
            data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                           allow_pickle=True).item()
            wp = np.asarray(data['world_points']); wp = wp[0] if wp.ndim == 5 else wp
            cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
            dd = np.asarray(data['depth']).reshape(wp.shape[:3])
            K = np.asarray(data['intrinsic'])
            ext = np.asarray(data['extrinsic'])
            for local, g in enumerate(range(start, end)):
                if owner[g] == k:
                    c2w = self._stac_aligned_pose(k, local, ext[local])
                    cache[g] = (wp[local].astype(np.float32), cf[local].astype(np.float32),
                                dd[local].astype(np.float32), np.linalg.inv(c2w),
                                K[local], c2w[:3, 3])
            del data
        masks = {}
        n_far_tot = n_drop_tot = 0
        for f, (wp_f, cf_f, dd_f, _, _, _) in cache.items():
            far = (cf_f > 1e-5) & (dd_f > cap)
            if not far.any():
                continue
            pts = wp_f[far].reshape(-1, 3).astype(np.float64)
            ok = np.ones(len(pts), bool)
            agree = np.zeros(len(pts), bool)
            contra = np.zeros(len(pts), bool)
            lo, hi = pts.min(0) - cap, pts.max(0) + cap
            for g, (_, cf_g, dd_g, w2c_g, K_g, cam_g) in cache.items():
                if g == f or not ((cam_g >= lo).all() and (cam_g <= hi).all()):
                    continue
                a, c = classify_far_points(pts, ok, cam_g, dd_g, cf_g, w2c_g, K_g,
                                           cap, floor, rate)
                agree |= a
                contra |= c
            drop = contra & ~agree
            n_far_tot += len(pts)
            n_drop_tot += int(drop.sum())
            if drop.any():
                m = np.zeros(far.shape, bool)
                m[far] = drop
                masks[f] = m
        if n_far_tot:
            print(f"[depth-cap] contradiction test: {n_drop_tot:,}/{n_far_tot:,} far "
                  f"points ({100 * n_drop_tot / max(n_far_tot, 1):.1f}%) are displaced "
                  f"duplicates of near-observed surfaces → dropped; the rest is unique "
                  f"far coverage → KEPT")
        return masks

    def _stac_write_deferred_outputs(self):
        """PLY + origins for every chunk, AFTER all geometric stages (elastic seam
        consensus, depth graph, two-copy blend) have finished mutating the aligned
        npys. Sick chunks write nothing (declared hole); far points contradicted
        by near observations are dropped (see _stac_far_contradictions). Resume:
        existing outputs are kept."""
        self._stac_max_write_depth = self._stac_write_depth_cap()
        self._stac_far_drop = self._stac_far_contradictions()
        for k in range(len(self.chunk_indices)):
            if (os.path.exists(os.path.join(self.pcd_dir, f"{k}_pcd.ply"))
                    and os.path.exists(os.path.join(self.pcd_dir, f"{k}_origins.npz"))):
                print(f"[outputs] chunk {k}: PLY + origins already written — skipped")
                continue
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True).item()
            self._stac_write_chunk_outputs(data, k)

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()


        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices,
                                             self.loop_list,
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        # STAC patch: per-chunk METRIC LOCK (see loop_utils/metric_lock.py). Every chunk
        # (and every loop-bridge prediction) is scaled to metric via its DA3 anchors
        # BEFORE any alignment, so the overlap alignment can run as SE(3)
        # (Model.using_sim3: false) — relative scale stops being a negotiable degree of
        # freedom, which is what chained the ±18-50% per-chunk scale errors ("onion").
        self._stac_metric_lock()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            # STAC: EXACT seam alignment. The overlap maps are the SAME frames pixel
            # for pixel — millions of exact correspondences. A robust rigid fit on
            # them lands at millimetres, where the generic point-map fit tolerated
            # 25-30 cm (measured: one seam glued 50 cm off in depth). Scale is
            # already consistent chunk-to-chunk (metric lock scale graph), so the
            # seam is rigid by construction. Falls back to the vendor fit if starved.
            s = R = t = None
            if self.config['Model'].get('exact_seam_align'):
                from loop_utils.metric_lock import robust_rigid
                _p1 = np.asarray(point_map1, np.float64).reshape(-1, 3)
                _p2 = np.asarray(point_map2, np.float64).reshape(-1, 3)
                _c1 = np.asarray(conf1).reshape(-1)
                _c2 = np.asarray(conf2).reshape(-1)
                _ok = (_c1 > 1e-5) & (_c2 > 1e-5)
                if mask is not None:
                    _ok &= np.asarray(mask).reshape(-1).astype(bool)
                _fit = robust_rigid(_p2[_ok], _p1[_ok])
                if _fit is not None:
                    R, t, _res, _n = _fit[0], _fit[1], _fit[2], _fit[3]
                    s = 1.0
                    self._stac_exact_seams = getattr(self, '_stac_exact_seams', 0) + 1
                    print(f"[exact-seam] {chunk_idx}->{chunk_idx+1}: rigid fit on "
                          f"{_n:,} exact correspondences, median residual {_res*100:.1f} cm")
                else:
                    print(f"[exact-seam] {chunk_idx}->{chunk_idx+1}: starved — "
                          f"falling back to the point-map fit")
            if R is None:
                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
                else:
                    conf_threshold = -1.0
                s, R, t = weighted_align_point_maps(point_map1, 
                                                    conf1, 
                                                    point_map2, 
                                                    conf2,
                                                    mask,
                                                    conf_threshold=conf_threshold,
                                                    config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            # STAC: adjacent chunks share `overlap` (60) IDENTICAL frames → their relative
            # Sim3 scale MUST be ~1. weighted_align_point_maps can return a degenerate scale
            # (e.g. 0.19 or 1.75) on low-parallax / near-planar overlap that still passes the
            # inlier check but is geometrically wrong; compounded over many chunks it shatters
            # the whole reconstruction (same object metres apart — exactly the scatter seen on
            # long scans). Reject out-of-range scales → 1.0 (rigid SE3 for that seam), like the
            # SE3 backbones (mapanything/da3) that never scattered.
            # Range tuned to reject ONLY geometrically-impossible degeneracies (measured 0.19
            # and 1.75) while KEEPING legitimate per-chunk scale variation (measured 0.84-1.24
            # with mm residuals + 60/60 inliers — those produced GOOD clouds). [0.9,1.1] was
            # too tight: it rejected legit 0.84/1.16 and wrecked a previously-good reconstruction.
            _S_LO, _S_HI = 0.6, 1.6
            if not (_S_LO <= float(s) <= _S_HI):
                print(f"[STAC] chunk {chunk_idx}->{chunk_idx+1}: REJECTING degenerate Sim3 "
                      f"scale {float(s):.4f} (outside [{_S_LO},{_S_HI}]) → 1.0 (rigid)")
                s = 1.0

            self.sim3_list.append((s, R, t))


        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask = mask_loop.squeeze() & mask_a.squeeze()
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop,
                                                          mask,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask = mask_loop.squeeze() & mask_b.squeeze()
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop,
                                                          mask,
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))


        _n_seams = max(len(self.chunk_indices) - 1, 0)
        if (self.config['Model'].get('exact_seam_align')
                and getattr(self, '_stac_exact_seams', 0) == _n_seams and _n_seams > 0):
            # every seam is an exact-correspondence rigid fit (mm-scale). The loop
            # optimizer's constraints come from the COARSE point-map fits — letting
            # them redistribute error would degrade the precise chain, not help it.
            if self.loop_enable:
                print(f"[exact-seam] all {_n_seams} seams exact — loop optimizer SKIPPED "
                      f"(coarse loop constraints must not drag a mm-precise chain)")
            self.loop_enable_opt = False
        else:
            self.loop_enable_opt = self.loop_enable
        if self.loop_enable_opt:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        # STAC patch: single-chunk case (frames <= chunk_size → exactly 1 chunk). The
        # pairwise apply loop below is range(0) and saves NOTHING (chunk_0 is normally
        # written inside the idx==0 branch of that loop), so short scans produced zero
        # PLYs. Write chunk_0 here directly (identity transform — nothing to align).
        if len(self.chunk_indices) == 1:
            if not (os.path.exists(os.path.join(self.result_aligned_dir, "chunk_0.npy"))
                    and os.path.exists(os.path.join(self.pcd_dir, "0_pcd.ply"))):
                cd0 = np.load(os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                              allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), cd0)
                self._stac_write_chunk_outputs(cd0, 0)
                print(f'[STAC] single chunk: saved 0_pcd.ply')

        # STAC: with the ELASTIC seam consensus and/or the DEPTH GRAPH enabled, each
        # chunk's PLY + origins must be written AFTER those stages finish mutating
        # the aligned npys — the apply loop only produces the aligned .npys and
        # _stac_write_deferred_outputs() writes the outputs at the end.
        _elastic = ((self.config['Model'].get('elastic_seam')
                     or self.config['Model'].get('depth_graph')
                     or self.config['Model'].get('blend_copies'))
                    and len(self.chunk_indices) > 1)
        if _elastic:
            print("[STAC] per-chunk PLY/origins deferred until after the elastic/depth stages")

        for chunk_idx in range(len(self.chunk_indices) - 1):
            # STAC patch (resume): skip this chunk's apply if its aligned .npy + pcd .ply
            # already exist (from a prior run). chunk_0 is written inside the idx==0 branch.
            # With elastic ON the .ply is written later, so only the .npy gates the skip.
            _done = os.path.exists(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy"))
            if not _elastic:
                _done = _done and os.path.exists(os.path.join(self.pcd_dir, f"{chunk_idx + 1}_pcd.ply"))
            if chunk_idx == 0:
                _done = _done and os.path.exists(os.path.join(self.result_aligned_dir, "chunk_0.npy"))
                if not _elastic:
                    _done = _done and os.path.exists(os.path.join(self.pcd_dir, "0_pcd.ply"))
            if _done:
                print(f'[STAC resume] chunk {chunk_idx + 1} aligned+pcd exist — apply skipped')
                continue
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]


            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                                     allow_pickle=True).item()

            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)
            # STAC: the per-chunk Sim3 scales world_points by s, but the raw per-camera `depth`
            # was left unscaled → the cloud (built from world_points) and the TSDF (which
            # integrates `depth`) diverge, growing with chunk drift (measured chunk0 1.00 →
            # chunk10 1.19) → far walls/ceilings displaced. Scale depth by the same s so the
            # integrated depth stays consistent with the aligned world_points / cloud.
            if chunk_data.get('depth') is not None:
                chunk_data['depth'] = chunk_data['depth'] * s


            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:

                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"),
                                               allow_pickle=True).item()

                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                if not _elastic:
                    self._stac_write_chunk_outputs(chunk_data_first, 0)
                # STAC: free unaligned chunk_0 NOW — its aligned .npy + pcd + origins are
                # written and nothing downstream reads unaligned (the TSDF reads
                # _tmp_results_aligned). Incremental cleanup so the apply phase never holds
                # ALL unaligned + ALL aligned at once (the ~2× peak that overflowed the disk).
                try:
                    os.remove(os.path.join(self.result_unaligned_dir, "chunk_0.npy"))
                except OSError:
                    pass


            # STAC fix: ALWAYS load the freshly-aligned chunk_{chunk_idx+1} (saved at the
            # aligned_path np.save above for every chunk_idx, including 0). The previous
            # `... if chunk_idx > 0 else chunk_data_first` wrote chunk_0's geometry into
            # chunk_1's PLY + origins on the first iteration → chunk_001.ply duplicated
            # chunk_000's points (identical counts) while chunk_001_origins held the real
            # chunk_1 count → reproject_chunks aborted on the points!=origins mismatch.
            if not _elastic:
                aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                                                 allow_pickle=True).item()
                self._stac_write_chunk_outputs(aligned_chunk_data, chunk_idx + 1)
            # STAC: free this chunk's unaligned .npy immediately (see chunk_0 note above) —
            # incremental cleanup keeps the apply phase ~flat on disk instead of doubling.
            try:
                os.remove(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"))
            except OSError:
                pass

        # STAC patches, in order: per-frame ELASTIC seam consensus (the two copies
        # of every shared frame coincide), per-frame DEPTH GRAPH (different frames
        # agree on the depth of shared surfaces), then the deferred PLY/origins
        # from the FINAL geometry. save_camera_poses applies the elastic pose moves.
        self._stac_elastic_seams()
        self._stac_depth_graph()
        self._stac_blend_copies()
        if _elastic:
            self._stac_write_deferred_outputs()

        self.save_camera_poses()

        print('Done.')

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # STAC patch: restrict to selected keyframes (selected_frames.json -> "selected_files")
        if self.selected_frames:
            import json as _json
            with open(self.selected_frames, 'r') as _f:
                _sel = set(_json.load(_f).get("selected_files", []))
            if _sel:
                _filtered = [p for p in self.img_list if os.path.basename(p) in _sel]
                print(f"[STAC] keyframe filter: {len(self.img_list)} -> {len(_filtered)} "
                      f"frames (from {self.selected_frames})")
                self.img_list = _filtered
        # STAC patch: uniform temporal stride (1-of-N). SAME value the loop detector
        # (LoopModel.get_image_paths) reads, so loop indices stay aligned with chunks.
        _stride = int(self.config.get('Model', {}).get('frame_stride', 1) or 1)
        if _stride > 1:
            _before = len(self.img_list)
            self.img_list = self.img_list[::_stride]
            print(f"[STAC] frame stride {_stride}: {_before} -> {len(self.img_list)} frames")
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        # STAC patch: dump the EXACT ordered list of frames processed (after any
        # keyframe filter + stride). This is the single source of truth that lets
        # the downstream origins map frame_global (index) -> real frame number,
        # so per-point traceability survives stride/keyframe subsetting.
        try:
            import json as _json
            with open(os.path.join(self.output_dir, "frame_list.json"), "w") as _fl:
                _json.dump([os.path.basename(p) for p in self.img_list], _fl)
            print(f"[STAC] wrote frame_list.json ({len(self.img_list)} frames)")
        except Exception as _e:
            print(f"[STAC] WARN: could not write frame_list.json: {_e}")

        # STAC patch: pin the loop detector to the EXACT same frame set as the chunks.
        # LoopDetector.run() re-globs the full image_dir and applies only frame_stride
        # (which is 1 here — the stride is baked into selected_frames.json), so without
        # this it processes ALL frames → loop pairs index full-dir space that does NOT
        # match the filtered/strided chunks (self.img_list) → misaligned loop closures
        # → corrupted global Sim3 alignment. Override get_image_paths so the loop detector
        # uses self.img_list, keeping loop indices 1:1 with the chunks. (Same fix the da3
        # backend applies in stray_da3_streaming.py.)
        _ld = getattr(self, "loop_detector", None)
        if _ld is not None:
            from pathlib import Path as _Path
            _kf = [_Path(p) for p in self.img_list]

            def _stac_fixed_image_paths(_ld=_ld, _kf=_kf):
                _ld.image_paths = _kf
                return _kf

            _ld.get_image_paths = _stac_fixed_image_paths
            _ld.image_paths = _kf
            print(f"[STAC] Loop detector pinned to {len(_kf)} frames (aligned with chunks)")

        # STAC patch (resume): only skip if VGGT-Long FULLY completed — camera_poses.txt
        # AND at least one pcd/*_pcd.ply. A run that saved poses but no PLY (e.g. the old
        # single-chunk bug) is NOT complete and must re-run (it'll reuse the cached chunks).
        if (os.path.exists(os.path.join(self.output_dir, "camera_poses.txt"))
                and glob.glob(os.path.join(self.pcd_dir, "*_pcd.ply"))):
            print("[STAC resume] camera_poses.txt + pcd exist — VGGT-Long already complete, skipping")
            return

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()  # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector  # Save GPU Memory
        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        # STAC fix: all_camera_poses holds the extrinsics captured at INFERENCE time —
        # raw per-chunk Omega scale. The metric lock scales world_points/depth/extrinsic
        # in the npys but never this in-memory list, so every pose block came out ~s_k×
        # compressed (s_k 7-23 measured on test4) placed at metric offsets: pose blocks
        # 2.3-8 m apart (165× the 2.4 cm within-block step) while the CLOUDS glued at cm
        # — and camera_poses.txt feeds omega-depth/scale_align, orient, TSDF and the
        # fine registration. Refresh every chunk's extrinsics from its ALIGNED npy (the
        # metric-locked poses the rest of the pipeline actually uses; sim3 is applied
        # below as before). Also immune to the resume path, where the in-memory list
        # mixes raw (fresh inference) and already-locked (resumed-from-disk) chunks.
        for _k in range(len(self.all_camera_poses)):
            _rng = self.all_camera_poses[_k][0]
            _p = os.path.join(self.result_aligned_dir, f"chunk_{_k}.npy")
            try:
                _cd = np.load(_p, allow_pickle=True).item()
                if _cd.get('extrinsic') is not None:
                    self.all_camera_poses[_k] = (_rng, np.asarray(_cd['extrinsic']))
                else:
                    print(f"[STAC] WARN: chunk {_k} npy has no extrinsic — keeping the "
                          f"in-memory (raw-scale) poses for that block")
            except Exception as _e:
                print(f"[STAC] WARN: could not refresh chunk {_k} poses from {_p} ({_e}) "
                      f"— keeping the in-memory (raw-scale) poses for that block")

        # STAC: when the elastic seam consensus ran, every frame's points were moved
        # by a per-frame rigid correction — the camera must move WITH its points
        # (depth maps and the TSDF stay valid by rigidity). _stac_elastic_corr[k][i]
        # is chunk k's world-space 4x4 for local frame i, applied AFTER the chunk's
        # accumulated Sim3 (the corrections were fitted on the aligned chunks).
        _ecorr = getattr(self, '_stac_elastic_corr', None)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            c2w = first_chunk_extrinsics[i]
            if _ecorr is not None:
                c2w = _ecorr[0][i] @ c2w
            all_poses[idx] = c2w
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]  #

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                if _ecorr is not None:
                    transformed_c2w = _ecorr[chunk_idx][i] @ transformed_c2w

                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        print(f"Camera poses saved to {poses_path}")
        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='config path')
    # STAC patch: explicit output dir + optional keyframe subset (restores STAC fork CLI)
    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='explicit output dir (default: auto timestamped under ./exps)')
    parser.add_argument('--selected_frames', type=str, required=False, default=None,
                        help='path to selected_frames.json (uses its "selected_files" list)')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    if args.save_dir:                       # STAC patch: honor explicit save_dir
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(
                exp_dir, image_dir.replace("/", "_"), current_datetime
            )

    # save_dir = os.path.join(
    #     exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
    # )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)
    else:
        copy_file(args.config, save_dir)    # STAC patch: save_dir may pre-exist

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_long = VGGT_Long(image_dir, save_dir, config, selected_frames=args.selected_frames)
    vggt_long.run()
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    # STAC patch: do NOT build pcd/combined_pcd.ply — the STAC pipeline never uses it
    # (_postprocess_reconstruction merges the per-chunk pcd/{K}_pcd.ply, and skips any
    # "combined" file). It was tens of GB of wasted disk + merge time.
    print('All done.')
    sys.exit()