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
        # "i, j, sim[, source]" lines; "#" lines are headers/the image-path list.
        # The optional 4th column is the candidate SOURCE (salad | instance |
        # manual — claude_stac.txt §4.4): the server appends instance/manual
        # candidates to this same file, and a resumed run consumes them.
        loop_txt = os.path.join(self.output_dir, "loop_closures.txt")
        if not self.useDBoW and os.path.exists(loop_txt):
            from loop_utils.loop_bridges import load_loop_candidates
            cands = load_loop_candidates(loop_txt)
            self.loop_cands = cands
            self.loop_list = [(c["i"], c["j"]) for c in cands]
            srcs = {}
            for c in cands:
                srcs[c["source"]] = srcs.get(c["source"], 0) + 1
            print(f"[STAC resume] loop_closures.txt found — {len(cands)} candidate(s) "
                  f"loaded by source {srcs}, DINOv2 extraction skipped")
            return

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
            self.loop_cands = [{"i": int(i), "j": int(j), "sim": float(s), "source": "salad"}
                               for i, j, s in (self.loop_detector.loop_closures or [])]

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

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False,
                             extra_1=None, extra_2=None):
        start_idx, end_idx = range_1
        chunk_image_paths = list(self.img_list[start_idx:end_idx])
        # STAC (claude_stac.txt §4.9): a loop bridge may carry NON-keyframe frames
        # (extra_1 after window 1, extra_2 after window 2) to densify the revisit.
        # They never enter the chain; the layout records which bridge frame is
        # which chunk frame so the exact correspondences ignore them.
        extra_1 = list(extra_1 or [])
        extra_2 = list(extra_2 or [])
        chunk_image_paths += extra_1
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += list(self.img_list[start_idx:end_idx]) + extra_2

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
        expected_frames = len(chunk_image_paths)

        # STAC patch (resume): if this chunk's .npy already exists (prior run), load it
        # and SKIP the expensive inference. Restore the camera state EXACTLY as the
        # inference path does, so SIM3 + alignment + pose saving are unaffected. A
        # corrupt/half-written file falls through and re-infers.
        if os.path.exists(save_path):
            try:
                predictions = np.load(save_path, allow_pickle=True).item()
                _n_on_disk = int(np.asarray(predictions['depth']).shape[0]
                                 if np.asarray(predictions['depth']).ndim == 3
                                 else np.asarray(predictions['depth']).shape[1])
                if is_loop and _n_on_disk != expected_frames:
                    raise ValueError(f"bridge on disk has {_n_on_disk} frames, this run "
                                     f"needs {expected_frames} (extra-frame policy changed)")
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
        if is_loop:
            predictions['_stac_extra'] = {"n_extra_1": len(extra_1), "n_extra_2": len(extra_2),
                                          "extra_1": [os.path.basename(p) for p in extra_1],
                                          "extra_2": [os.path.basename(p) for p in extra_2]}

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

        # ── ZOOM SCALE FIX (USER ORDER 2026-09-04): a zoomed chunk's DA3
        # anchors are broken (assumed focal) — the scale must be CORRECTED,
        # never the chunk discarded. Same session-relative focal criterion as
        # flag_sick_chunks; the affected chunks' anchor scales are EXCLUDED so
        # the seam graph carries scale into them from their neighbours (seam
        # ratios are relative depth on shared frames — zoom does not break
        # them the way it breaks DA3's absolute metre).
        zoom_chunks = set()
        if ml.get('zoom_scale_fix', True) and len(fx_map) >= 3:
            _fxv = np.array(list(fx_map.values()), np.float64)
            _fmed = float(np.median(_fxv))
            _fmad = float(np.median(np.abs(_fxv - _fmed)))
            if _fmad > 0:
                for k, v in fx_map.items():
                    z = abs(v - _fmed) / (1.4826 * _fmad)
                    if z > 3.5 and k in scales:
                        zoom_chunks.add(k)
                        report["chunks"].setdefault(str(k), {})[
                            "zoom_anchor_excluded"] = True
                        scales.pop(k, None)
                        n_anchor_map.pop(k, None)
                        print(f"[metric-lock] chunk {k}: optical ZOOM (fx {v:.0f} "
                              f"vs session {_fmed:.0f}, z={z:.1f}) — DA3 anchors "
                              f"EXCLUDED, scale comes from the seam graph "
                              f"(neighbours)")

        # 2) fuse both sensors: seams make neighbours CONSISTENT (0.1-1% noise),
        # anchors pin the global metre (±8-15% each). Weighted LS in log space.
        # STAC F1 (§5.2): ABSOLUTE rows from other sources (VIO, regulated
        # dimensions, a user measurement — Model.metric_lock.absolute_rows,
        # each {chunk, log_s, sigma, source}) join the same solve; the loop rows
        # (§5.1) are added by _stac_scale_close once the bridges are measured.
        _abs_rows = [(int(r["chunk"]), float(r["log_s"]), float(r["sigma"]),
                      str(r.get("source", "unspecified")))
                     for r in (ml.get('absolute_rows') or [])]
        if ml.get('vio') and not already:
            _abs_rows += self._stac_vio_rows(ml['vio'], report)
        self._stac_scale_inputs = None
        if scales and seam_rel:
            _sigma_seam = float(ml.get('sigma_seam', 0.003))
            _sigma_anchor = float(ml.get('sigma_anchor', 0.08))
            self._stac_scale_inputs = {"s_da3": dict(scales), "n_anchors": dict(n_anchor_map),
                                       "seam_rel": dict(seam_rel), "sigma_seam": _sigma_seam,
                                       "sigma_anchor": _sigma_anchor, "absolute": _abs_rows}
            if _abs_rows:
                print(f"[metric-lock] {len(_abs_rows)} absolute scale row(s) from "
                      f"{sorted({r[3] for r in _abs_rows})} enter the graph")
            s_opt = solve_scale_graph(scales, n_anchor_map, seam_rel,
                                      len(self.chunk_indices),
                                      sigma_seam=_sigma_seam,
                                      sigma_anchor=_sigma_anchor,
                                      absolute=_abs_rows)
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
            self._stac_hybrid_da3(data, k, anchor_dir, near_frac)
            np.save(path, data)
        # STAC F1 bookkeeping for the loop stage (_stac_scale_close): what was
        # applied per chunk (the drift stage's geometric mean when it ran).
        self._stac_scales_applied = {int(k): float(v) for k, v in scales.items()}
        for k in range(len(self.chunk_indices)):
            self._stac_scales_applied.setdefault(k, fallback)
        self._stac_drift_frames = drift_frames
        # 2) loop-bridge predictions (in memory) — they are Omega passes with their own
        # arbitrary scale; the SE(3) loop legs need them metric too. Anchors inside the
        # bridge's two ranges when available, else the mean of the two parent chunks.
        # (Vendor loop path only: the STAC path locks its bridges in _stac_lock_bridges,
        # after their own anchors were extracted.)
        for li, (item, pred) in enumerate(
                [] if self._stac_loops_cfg() is not None
                else (getattr(self, 'loop_predict_list', []) or [])):
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
        # USER ORDER 2026-09-04: the write-blocking gate is OPT-IN
        # (Model.metric_lock.health_gate). Default: flags are DIAGNOSTIC ONLY
        # — every chunk writes; a zoomed chunk's scale is corrected upstream
        # (zoom_scale_fix) instead of being turned into a declared hole.
        _gate = bool(ml.get('health_gate', False))
        self._stac_sick_chunks = set(sick) if _gate else set()
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
            if _gate:
                print(f"[health] ⛔ {len(sick)} chunk(s) EXCLUDED from the cloud "
                      f"({sorted(sick)}) — kept as alignment bridges, see chunk_health.json")
            else:
                print(f"[health] {len(sick)} chunk(s) FLAGGED ({sorted(sick)}) — "
                      f"DIAGNOSTIC ONLY (health_gate off): they still write; zoom "
                      f"chunks got seam-graph scale instead")
        else:
            print(f"[health] ✅ all {len(self.chunk_indices)} chunks healthy "
                  f"(parallax + anchor coherence)")

    def _stac_vio_rows(self, vio_cfg, report):
        """§5.2: one ABSOLUTE scale row per chunk from a VIO trajectory — the
        per-segment ratio of VIO arc length to the chunk's RAW camera path
        (reconstruction.vio_scale.estimate_vio_scale, per chunk). A chunk with
        too few voting segments gets no row and says so in metric_lock.json."""
        from loop_utils.loop_bridges import cfg_req
        from loop_utils.metric_lock import real_frame_number
        scfg = self._stac_scale_cfg()
        if scfg is None:
            raise RuntimeError("Model.metric_lock.vio is set but Model.scale is missing — "
                               "the VIO row σ (scale.sigma_vio) must be configured")
        vs = self._stac_server_module("reconstruction.vio_scale")
        if vs is None:
            raise RuntimeError("Model.metric_lock.vio needs Model.loops.stac_server_dir "
                               "(reconstruction.vio_scale) — refusing to silently skip VIO")
        vt, vp, _ = vs.load_vio_trajectory(vio_cfg['path'])
        fps = float(vio_cfg['fps'])
        sigma_vio = float(cfg_req(scfg, "sigma_vio", "scale"))
        seg_s = float(cfg_req(scfg, "vio_segment_s", "scale"))
        min_seg = int(cfg_req(scfg, "vio_min_segments_chunk", "scale"))
        min_disp = float(cfg_req(scfg, "vio_min_seg_disp_m", "scale"))
        min_cov = float(vio_cfg.get('min_coverage', 0.0))
        rows = []
        report["vio"] = {}
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_unaligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True).item()
            ext = np.asarray(data['extrinsic'])
            if ext.ndim == 4:
                ext = ext[0]
            kf_c = ext[:, :3, 3].astype(np.float64)
            kf_t = np.array([real_frame_number(self.img_list[g]) for g in range(start, end)],
                            np.float64) / fps
            del data
            try:
                info = vs.estimate_vio_scale(vt, vp, kf_t, kf_c, segment_s=seg_s,
                                             min_seg_disp_m=min_disp, min_segments=min_seg,
                                             min_coverage=min_cov)
            except RuntimeError as _e:
                report["vio"][str(k)] = {"row": False, "reason": str(_e)}
                print(f"[metric-lock] chunk {k}: no VIO row ({_e})")
                continue
            s_k = float(info["s_vio"])
            sig = max(sigma_vio, float(info.get("mad_rel") or 0.0))
            rows.append((k, float(np.log(s_k)), sig, "vio"))
            report["vio"][str(k)] = {"row": True, "s_vio": s_k, "sigma": sig,
                                     "n_segments": info["n_segments"],
                                     "coverage_frac": info["coverage_frac"]}
            print(f"[metric-lock] chunk {k}: VIO absolute row s={s_k:.4f} σ={sig:.3f} "
                  f"({info['n_segments']} segments)")
        return rows

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
        from loop_utils.metric_lock import frame_owner, backfill_mask
        owner = frame_owner(self.chunk_indices, len(self.img_list))
        start, end = self.chunk_indices[chunk_idx]
        S = end - start
        cf = np.asarray(confs).reshape(S, -1).copy()
        bfstore = getattr(self, '_stac_backfill', None) or {}
        kept, bf_frames, bf_px = 0, 0, 0
        for local in range(S):
            if owner[start + local] != chunk_idx:
                entry = bfstore.get((chunk_idx, local))
                if entry is not None:
                    # BACKFILL: keep exactly the pixels the owner will NOT write
                    oc, othr = entry
                    m = (backfill_mask(oc, othr) if oc is not None
                         else np.ones(cf[local].shape, bool))
                    cf[local] = np.where(m, cf[local], 0.0)
                    n = int((cf[local] > 1e-5).sum())
                    if n:
                        bf_frames += 1
                        bf_px += n
                else:
                    cf[local] = 0.0
            else:
                kept += 1
        msg = (f"[frame-owner] chunk {chunk_idx}: writes {kept}/{S} frames "
               f"(the rest belong to neighbours)")
        if bf_px:
            msg += (f" + backfills {bf_px:,} owner-dropped px over "
                    f"{bf_frames} shared frame(s)")
        print(msg)
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
        # frozen threshold (backfill prepare pass) keeps the owner-writes
        # prediction exact; without backfill the live computation is unchanged
        thr = getattr(self, '_stac_write_thr', {}).get(K)
        if thr is None:
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

        # tame the raw fits: along-trajectory smoothing + translation cap (both
        # sides consume the same smoothed fit, so copy coincidence is preserved)
        _win = int(self.config['Model'].get('elastic_smooth_win', 5) or 1)
        _cap = self.config['Model'].get('elastic_max_t_m', 0.30)
        _cap = float(_cap) if _cap else None
        if _win > 1 or _cap:
            from loop_utils.metric_lock import smooth_seam_fits
            fits, _ncap = smooth_seam_fits(fits, window=_win, max_t=_cap)
            print(f"[elastic] fits tamed: smoothing window {_win} frames along the "
                  f"trajectory, |t| cap {(_cap or 0) * 100:.0f} cm "
                  f"({_ncap} frame fit(s) capped)")

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
        if self._stac_authority_cfg() and _cap:
            _t_all = max(float(np.max(np.linalg.norm(c[:, :3, 3], axis=1)))
                         for c in self._stac_elastic_corr.values())
            _fr = _t_all / float(_cap)
            self._stac_authority_record(
                "elastic_seam", _fr,
                _fr > float(self._stac_authority_cfg()["saturation_warn"]),
                {"max_m": float(_cap), "used_max_m": _t_all, "n_capped_fits": int(_ncap)})

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

    def _stac_intra_chunk(self):
        """STAC patch: INTRA-CHUNK per-frame consensus — bounded fields with
        anchored boundaries.

        The residual warp lives BETWEEN frames of the same chunk (test4: the
        same ground reconstructed ~7-9 cm apart by frames a few indices apart
        — the depth-graph ladder proved it is a POSE error). The failed global
        pose graph taught the constraint this stage is built on: short-span
        evidence must never produce corrections longer than its span. Here
        every chunk is solved ALONE, its endpoint frames CLAMPED to zero (the
        seam consensus the elastic already established), so no correction
        longer than one chunk can exist; per-chunk held-out pairs gate each
        field independently — a chunk that does not earn its correction stays
        identity, alone. Shared frames get ONE blended correction (fields are
        zero at chunk edges → the blend is continuous and both copies stay
        coincident). Points and pose move together: depth/TSDF invariant.
        Resume-safe via intra_chunk.json + per-npy stamp."""
        if not self.config['Model'].get('intra_chunk') or len(self.chunk_indices) < 1:
            return
        import json as _json
        from loop_utils.metric_lock import (surface_pair_correspondences,
                                            robust_rigid, filter_pair_fits,
                                            solve_chunk_field, blend_chunk_fields,
                                            chunk_field_verdict, se3_matrices)
        _sick = getattr(self, '_stac_sick_chunks', set())
        N = len(self.img_list)

        ic_path = os.path.join(self.output_dir, "intra_chunk.json")
        xi = None
        if os.path.exists(ic_path):
            try:
                prev = _json.load(open(ic_path))
                if prev.get("chunk_indices") == [list(ci) for ci in self.chunk_indices]:
                    xi = np.asarray(prev["xi"], np.float64)
                    print("[intra-chunk] resume: solution loaded from intra_chunk.json")
            except Exception as _e:
                print(f"[intra-chunk] intra_chunk.json unreadable ({_e}) — remeasuring")

        if xi is None:
            fit_offsets = (1, 2, 3, 5, 8, 12)
            holdout_offsets = (4, 10)
            fields, report = {}, {}
            for k, (start, end) in enumerate(self.chunk_indices):
                S = end - start
                if k in _sick or S < 8:
                    report[str(k)] = {"verdict": "SKIP", "reason": "sick or tiny"}
                    continue
                data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                               allow_pickle=True).item()
                if data.get('_stac_intra_applied'):
                    raise RuntimeError(
                        f"[intra-chunk] chunk {k} already corrected but intra_chunk.json "
                        f"is missing/stale — remeasuring on corrected data would be "
                        f"wrong. Delete _tmp_results_aligned + pcd and re-run.")
                wp = np.asarray(data['world_points'])
                if wp.ndim == 5:
                    wp = wp[0]
                cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
                ext = np.asarray(data['extrinsic'])
                K = np.asarray(data['intrinsic'])
                cache = {}
                for local in range(S):
                    c2w = self._stac_aligned_pose(k, local, ext[local])
                    cache[local] = (wp[local].astype(np.float32),
                                    cf[local].astype(np.float32),
                                    np.linalg.inv(c2w), K[local])
                del data
                fits, held = [], []
                for f in range(S):
                    for d in fit_offsets + holdout_offsets:
                        g = f + d
                        if g >= S:
                            continue
                        pq = surface_pair_correspondences(
                            cache[f][0], cache[f][1],
                            cache[g][0], cache[g][1], cache[g][2], cache[g][3])
                        if pq is None:
                            continue
                        if d in holdout_offsets:
                            held.append((f, g, pq[0][:2000], pq[1][:2000]))
                            continue
                        fit = robust_rigid(pq[0], pq[1], sample=8000)
                        if fit is not None:
                            fits.append((f, g, fit[0], fit[1], fit[2], fit[3]))
                del cache
                taus = filter_pair_fits(fits)
                if len(taus) < S or len(held) < S // 4:
                    report[str(k)] = {"verdict": "SKIP",
                                      "reason": f"thin ({len(taus)} fit/{len(held)} held)"}
                    print(f"[intra-chunk] chunk {k}: too thin "
                          f"({len(taus)} fit / {len(held)} held) — identity")
                    continue
                fld = solve_chunk_field(taus, S)
                v = chunk_field_verdict(fld, taus, held)
                ok = v["bounded"] and v["improves"]
                tmax = float(np.max(np.linalg.norm(fld[:, 3:], axis=1)))
                print(f"[intra-chunk] chunk {k}: held-out "
                      f"{v['med_before'] * 100:.2f} -> {v['med_after'] * 100:.2f} cm | "
                      f"max correction {tmax * 100:.1f} cm | bounded={v['bounded']} "
                      f"improves={v['improves']} -> {'APPLY' if ok else 'IDENTITY'}")
                report[str(k)] = {"verdict": "APPLY" if ok else "SKIP",
                                  "held_before_cm": v["med_before"] * 100,
                                  "held_after_cm": v["med_after"] * 100,
                                  "max_correction_cm": tmax * 100,
                                  "bounded": bool(v["bounded"]),
                                  "improves": bool(v["improves"]),
                                  "n_fit": len(taus), "n_held": len(held)}
                if ok:
                    fields[k] = fld
            xi = blend_chunk_fields(self.chunk_indices, fields, N)
            n_ok = len(fields)
            print(f"[intra-chunk] {n_ok}/{len(self.chunk_indices)} chunk field(s) "
                  f"earned; blended max correction "
                  f"{float(np.max(np.linalg.norm(xi[:, 3:], axis=1))) * 100:.1f} cm")
            with open(ic_path, "w") as f_:
                _json.dump({"chunk_indices": [list(ci) for ci in self.chunk_indices],
                            "xi": xi.tolist(), "chunks": report}, f_, indent=1)

        X = se3_matrices(xi)
        if self._stac_authority_cfg():
            _acfg = self._stac_authority_cfg()
            _used = float(np.max(np.linalg.norm(np.asarray(xi)[:, 3:], axis=1)))
            _fr = _used / float(_acfg["intra_chunk_max_m"])
            self._stac_authority_record("intra_chunk", _fr,
                                        _fr > float(_acfg["saturation_warn"]),
                                        {"max_m": float(_acfg["intra_chunk_max_m"]),
                                         "used_max_m": _used})
        # compose into the elastic per-frame fields FIRST: poses, depth graph,
        # depth cap and the origins writer all read _stac_elastic_corr
        ecorr = getattr(self, '_stac_elastic_corr', None)
        if ecorr is None:
            ecorr = {k: np.tile(np.eye(4), (end - start, 1, 1))
                     for k, (start, end) in enumerate(self.chunk_indices)}
            self._stac_elastic_corr = ecorr
        for k, (start, end) in enumerate(self.chunk_indices):
            for local, g in enumerate(range(start, end)):
                ecorr[k][local] = X[g] @ ecorr[k][local]
        # apply to points — BOTH copies of every shared frame get the same
        # rigid move, so the elastic seam consensus is preserved exactly
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            data = np.load(path, allow_pickle=True).item()
            if data.get('_stac_intra_applied'):
                print(f"[intra-chunk] chunk {k}: already corrected — skipped")
                continue
            wp = np.asarray(data['world_points'])
            lead = wp.ndim == 5
            if lead:
                wp = wp[0]
            moved = 0
            for local, g in enumerate(range(start, end)):
                M = X[g]
                if np.allclose(M, np.eye(4), atol=1e-12):
                    continue
                p = wp[local].reshape(-1, 3).astype(np.float64)
                wp[local] = (p @ M[:3, :3].T + M[:3, 3]).reshape(wp[local].shape).astype(wp.dtype)
                moved += 1
            data['world_points'] = wp[None] if lead else wp
            data['_stac_intra_applied'] = True
            np.save(path, data)
            if moved:
                print(f"[intra-chunk] chunk {k}: {moved}/{end - start} frames moved")

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
        if self._stac_authority_cfg():
            _acfg = self._stac_authority_cfg()
            _fa = float(np.max(np.abs(np.log(np.asarray(a))))) / float(_acfg["depth_graph_max_log_a"])
            _fb = float(np.max(np.abs(np.asarray(b)))) / float(_acfg["depth_graph_max_b_m"])
            _fr = max(_fa, _fb)
            self._stac_authority_record("depth_graph", _fr,
                                        _fr > float(_acfg["saturation_warn"]),
                                        {"max_log_a": float(_acfg["depth_graph_max_log_a"]),
                                         "max_b_m": float(_acfg["depth_graph_max_b_m"]),
                                         "used_max_log_a": float(np.max(np.abs(np.log(np.asarray(a))))),
                                         "used_max_b_m": float(np.max(np.abs(np.asarray(b))))})
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
        resume needs no special casing beyond the skip stamp.

        AUDIT vs frame_ownership (claude_stac.txt §7, 2026-09-13): the two
        operate on DIFFERENT things and stay as they are. Ownership is a WRITE
        policy — which copy of a shared frame puts pixels into the PLY
        (_stac_owned_confs); the blend is a PREDICTION consensus — the two
        copies are two independent measurements of the same depth field and
        both become their mean. The non-owner copy is not dead after the
        blend: (1) ownership_backfill writes the owner-dropped pixels FROM the
        non-owner copy (backfill_mask), (2) _emit_omega_depth iterates every
        chunk's frames and the LAST chunk's copy of a shared frame wins the
        omega depth npz, (3) the depth graph / depth cap read owner frames but
        the seam sensors read both copies. Restricting the blend to the owner
        copy would leave those readers with an un-blended, inconsistent copy
        — the blend into BOTH is what keeps every downstream reader on one
        consensus. Proof: after _stac_blend_copies both copies are bitwise
        identical (same wp/cf/dd written back), so no reader can observe a
        difference between "owner" and "non-owner" data."""
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

    def _stac_hybrid_da3(self, data, k, anchor_dir, near_frac):
        """HYBRID WRITE driver for one metric-scaled chunk (in place, BEFORE the
        Sim3 chain and every consensus stage, while world_points ↔ depth ↔
        extrinsic are still one coherent chunk-local system): each frame with an
        isolated DA3 depth map adopts DA3's depth SHAPE at omega's scale and
        pose (loop_utils.metric_lock.hybrid_substitute). Downstream stages
        (exact seams, elastic, depth graph, writers) then consume the straighter
        geometry with zero pose bookkeeping. Frames without a DA3 map keep omega
        (log the count — the server extracts DA3 for every keyframe when
        Model.hybrid_da3 is on)."""
        if not self.config['Model'].get('hybrid_da3'):
            return
        from loop_utils.metric_lock import (anchor_ratio, hybrid_substitute,
                                            real_frame_number)
        far_m = self.config['Model'].get('hybrid_da3_far_m', 15.0)
        far_m = float(far_m) if far_m else None
        start, end = self.chunk_indices[k]
        wp = np.asarray(data['world_points'])
        lead = wp.ndim == 5
        if lead:
            wp = wp[0]
        S = wp.shape[0]
        depth = np.asarray(data['depth'])
        dlead = depth.ndim == 4 and depth.shape[0] == 1
        if dlead:
            depth = depth[0]
        conf = np.asarray(data['world_points_conf'])
        ext = np.asarray(data['extrinsic'])
        if ext.ndim == 4:
            ext = ext[0]
        n_sub_frames, n_missing, n_starved, px_total, deltas = 0, 0, 0, 0, []
        for local in range(S):
            num = real_frame_number(self.img_list[start + local])
            npz_path = os.path.join(str(anchor_dir), f"frame_{int(num)}.npz")
            if not os.path.exists(npz_path):
                n_missing += 1
                continue
            z = np.load(npz_path)
            if "depth" not in z:
                n_missing += 1
                continue
            r = anchor_ratio(depth[local], z["depth"], conf=conf[local],
                             near_frac=near_frac)
            if r is None or not np.isfinite(r) or r <= 0:
                n_starved += 1
                continue
            wp_new, d_new, n_px, med = hybrid_substitute(
                wp[local], conf[local], depth[local], ext[local],
                z["depth"], r, far_m=far_m)
            if n_px == 0:
                n_starved += 1
                continue
            wp[local] = wp_new
            depth[local] = d_new
            n_sub_frames += 1
            px_total += n_px
            deltas.append(med)
        data['world_points'] = wp[None] if lead else wp
        data['depth'] = depth[None] if dlead else depth
        med_all = float(np.median(deltas)) * 100 if deltas else 0.0
        print(f"[hybrid-da3] chunk {k}: {n_sub_frames}/{S} frames re-shaped on DA3 "
              f"depth at omega scale/pose ({px_total:,} px, median shape "
              f"correction {med_all:.1f}%)"
              + (f"; {n_missing} frame(s) without DA3 map" if n_missing else "")
              + (f"; {n_starved} starved/gated" if n_starved else ""))

    def _stac_prepare_backfill(self):
        """One sequential pass over the aligned chunks BEFORE any output is
        written: freeze every chunk's write threshold on its OWNED confidences,
        and for every shared frame stash the OWNER copy's confidences so the
        non-owner can backfill exactly the pixels the owner will drop
        (loop_utils.metric_lock.backfill_mask). Frozen thresholds make the
        owner-writes prediction exact: owner-writes and backfill stay disjoint,
        so no pixel enters the cloud twice. A sick owner writes nothing → its
        healthy neighbour may backfill everything valid."""
        self._stac_write_thr = {}
        self._stac_backfill = {}
        if (not self.config['Model'].get('ownership_backfill')
                or not self.config['Model'].get('frame_ownership')
                or self.chunk_indices is None or len(self.chunk_indices) <= 1):
            return
        from loop_utils.metric_lock import frame_owner
        owner = frame_owner(self.chunk_indices, len(self.img_list))
        sick = getattr(self, '_stac_sick_chunks', set())
        prev_tail = None       # (k-1, {g: conf_row}) for the shared frames
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                print(f"[frame-owner] backfill: chunk_{k}.npy missing — backfill "
                      f"disabled this run (plain ownership)")
                self._stac_write_thr, self._stac_backfill = {}, {}
                return
            data = np.load(path, allow_pickle=True).item()
            S = end - start
            cf = np.asarray(data['world_points_conf']).reshape(S, -1).astype(np.float32)
            del data
            owned_rows = [l for l in range(S) if owner[start + l] == k]
            self._stac_write_thr[k] = (
                None if k in sick or not owned_rows
                else self._stac_conf_threshold(cf[owned_rows].reshape(-1)))
            if prev_tail is not None and prev_tail[0] == k - 1:
                j = k - 1
                s_j, e_j = self.chunk_indices[j]
                for g in range(start, min(e_j, end)):
                    if owner[g] == j and j not in sick:
                        # k is the non-owner: it backfills what j drops
                        self._stac_backfill[(k, g - start)] = (
                            prev_tail[1][g], self._stac_write_thr[j])
                    elif owner[g] == j and j in sick:
                        self._stac_backfill[(k, g - start)] = (None, None)
                    elif owner[g] == k:
                        # j is the non-owner (thr_k already frozen above)
                        self._stac_backfill[(j, g - s_j)] = (
                            cf[g - start].copy(),
                            None if k in sick else self._stac_write_thr[k])
            if k + 1 < len(self.chunk_indices):
                nxt0 = self.chunk_indices[k + 1][0]
                prev_tail = (k, {g: cf[g - start].copy()
                                 for g in range(max(nxt0, start), end)})
            else:
                prev_tail = None
        print(f"[frame-owner] backfill armed: {len(self._stac_backfill)} shared "
              f"frame(s) may recover owner-dropped pixels (frozen thresholds)")

    def _stac_write_deferred_outputs(self):
        """PLY + origins for every chunk, AFTER all geometric stages (elastic seam
        consensus, depth graph, two-copy blend) have finished mutating the aligned
        npys. Sick chunks write nothing (declared hole); far points contradicted
        by near observations are dropped (see _stac_far_contradictions). Resume:
        existing outputs are kept."""
        self._stac_max_write_depth = self._stac_write_depth_cap()
        self._stac_far_drop = self._stac_far_contradictions()
        self._stac_prepare_backfill()
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
            del data
            # NOTE (2026-09-04): do NOT delete the aligned npy here — the
            # omega-depth writer + scale_align still read it after this stage
            # (deleting here caused 'wrote 0 omega depths' → scale FAIL).
            # map_worker deletes _tmp_results_aligned right after the metric
            # scale succeeds.

    # ══════════════════════════════════════════════════════════════════════
    # STAC F1 — EXACT LOOP BRIDGES, SPATIAL GATE, CLOSED SCALE
    # (claude_stac.txt §4.1, §4.2, §4.5 step 0, §4.9, §5). Active when the
    # config carries Model.loops (built by server/workers/map_worker.py from
    # config.yaml `loops:`/`scale:`); without it the vendor loop path runs.
    # ══════════════════════════════════════════════════════════════════════
    def _stac_loops_cfg(self):
        return self.config['Model'].get('loops')

    def _stac_scale_cfg(self):
        return self.config['Model'].get('scale')

    def _stac_server_module(self, name):
        """Import a STAC server module (e.g. reconstruction.loops.spatial_gate)
        when Model.loops.stac_server_dir is configured. None (logged) when the
        fork runs standalone — the feature that needs it is then declared OFF
        in the loop report, never silently skipped."""
        cfg = self._stac_loops_cfg() or {}
        sdir = cfg.get('stac_server_dir')
        if not sdir:
            return None
        if sdir not in sys.path:
            sys.path.insert(0, sdir)
        import importlib
        return importlib.import_module(name)

    def _stac_load_chunk(self, k):
        """Unaligned (metric-locked) chunk k, with a two-entry cache so the
        gate/bridge stages never hold more than two chunks in memory."""
        cache = getattr(self, '_stac_chunk_cache', None)
        if cache is None:
            cache = self._stac_chunk_cache = {}
        if k in cache:
            return cache[k]
        data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{k}.npy"),
                       allow_pickle=True).item()
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[k] = data
        return data

    def _stac_drop_chunk_cache(self):
        if getattr(self, '_stac_chunk_cache', None):
            self._stac_chunk_cache.clear()

    def _stac_seam_chain(self, tag):
        """Exact rigid seams over the CURRENT unaligned chunks → sequential
        (s=1, R, t) list (chunk k+1 → chunk k). Same fit the vendor path uses
        (robust_rigid on the shared frames); starved seams fall back to the
        vendor point-map fit and are recorded as such."""
        from loop_utils.metric_lock import robust_rigid
        seq, report = [], {}
        for chunk_idx in range(len(self.chunk_indices) - 1):
            d1 = self._stac_load_chunk(chunk_idx)
            d2 = self._stac_load_chunk(chunk_idx + 1)
            pm1 = d1['world_points'][-self.overlap:]
            pm2 = d2['world_points'][:self.overlap]
            c1 = d1['world_points_conf'][-self.overlap:]
            c2 = d2['world_points_conf'][:self.overlap]
            _p1 = np.asarray(pm1, np.float64).reshape(-1, 3)
            _p2 = np.asarray(pm2, np.float64).reshape(-1, 3)
            _ok = (np.asarray(c1).reshape(-1) > 1e-5) & (np.asarray(c2).reshape(-1) > 1e-5)
            fit = robust_rigid(_p2[_ok], _p1[_ok])
            if fit is not None:
                R, t, res, n = fit
                seq.append((1.0, R, t))
                report[str(chunk_idx)] = {"exact": True, "residual_m": float(res), "n_fit": int(n)}
            else:
                conf_threshold = min(np.median(c1), np.median(c2)) * 0.1
                s, R, t = weighted_align_point_maps(pm1, c1, pm2, c2, None,
                                                    conf_threshold=conf_threshold,
                                                    config=self.config)
                seq.append((1.0, R, t))
                report[str(chunk_idx)] = {"exact": False, "starved": True}
                print(f"[exact-seam:{tag}] {chunk_idx}->{chunk_idx+1}: starved — vendor "
                      f"point-map fit used (recorded)")
        return seq, report

    class _StacChainView:
        """Duck-typed TrajectoryView for reconstruction.loops.spatial_gate:
        per-global-frame c2w + K under the provisional seam chain, and world
        points of a frame on demand (owner chunk, cumulative transform)."""

        def __init__(self, owner_self, seq):
            from loop_utils.metric_lock import frame_owner
            self._o = owner_self
            N = len(owner_self.img_list)
            self.n_frames = N
            self._owner = frame_owner(owner_self.chunk_indices, N)
            cum = accumulate_sim3_transforms(seq) if seq else []
            self._cum = {0: (1.0, np.eye(3), np.zeros(3))}
            for k in range(1, len(owner_self.chunk_indices)):
                self._cum[k] = cum[k - 1]
            self._poses, self._Ks = {}, {}
            self.hw = None
            for k, (start, end) in enumerate(owner_self.chunk_indices):
                d = owner_self._stac_load_chunk(k)
                ext = np.asarray(d['extrinsic'])
                if ext.ndim == 4:
                    ext = ext[0]
                K = np.asarray(d['intrinsic'])
                if K.ndim == 4:
                    K = K[0]
                dep = np.asarray(d['depth'])
                self.hw = tuple(int(x) for x in (dep.shape[-2], dep.shape[-1]))
                s, R, t = self._cum[k]
                S = np.eye(4)
                S[:3, :3] = float(s) * np.asarray(R)
                S[:3, 3] = np.asarray(t)
                for local, g in enumerate(range(start, end)):
                    if self._owner[g] != k:
                        continue
                    M = S @ np.asarray(ext[local], np.float64)
                    M[:3, :3] /= float(s)
                    self._poses[g] = M
                    self._Ks[g] = np.asarray(K[local], np.float64)

        def pose(self, g):
            return self._poses.get(int(g))

        def K(self, g):
            return self._Ks.get(int(g))

        def centres(self):
            out = np.full((self.n_frames, 3), np.nan)
            for g, M in self._poses.items():
                out[g] = M[:3, 3]
            return out

        def depth(self, g):
            """The frame's own measured depth (camera z, chunk units after the
            lock) — the occlusion witness of the spatial gate."""
            k = int(self._owner[int(g)])
            if k < 0:
                return None
            d = self._o._stac_load_chunk(k)
            dep = np.asarray(d['depth'])
            if dep.ndim == 4:
                dep = dep[0]
            local = int(g) - self._o.chunk_indices[k][0]
            s = float(self._cum[k][0])
            return dep[local].astype(np.float64) * s

        def points(self, g, n, seed=0):
            k = int(self._owner[int(g)])
            if k < 0:
                return np.zeros((0, 3))
            d = self._o._stac_load_chunk(k)
            wp = np.asarray(d['world_points'])
            wp = wp[0] if wp.ndim == 5 else wp
            local = int(g) - self._o.chunk_indices[k][0]
            cf = np.asarray(d['world_points_conf']).reshape(wp.shape[:3])[local].reshape(-1)
            p = wp[local].reshape(-1, 3)[cf > 1e-5].astype(np.float64)
            if len(p) > n:
                p = p[np.random.default_rng(seed).choice(len(p), int(n), replace=False)]
            s, R, t = self._cum[k]
            return float(s) * (p @ np.asarray(R).T) + np.asarray(t)

    def _stac_plan_bridges(self, loop_results):
        """§4.2 step 0 / §4.5: every candidate passes the spatial-plausibility
        gate BEFORE a bridge is spent on it. Returns [(item, cand, gate)] for
        the survivors; rejected candidates go to the loop report."""
        from loop_utils.loop_bridges import cfg_req
        lcfg = self._stac_loops_cfg()
        gate_mod = self._stac_server_module("reconstruction.loops.spatial_gate")
        by_pair = {}
        for c in getattr(self, 'loop_cands', []) or []:
            by_pair[(int(c['i']), int(c['j']))] = c
        planned, rejected = [], []
        view = None
        if gate_mod is not None:
            seq, seam_rep = self._stac_seam_chain("provisional")
            self._stac_provisional_seams = seam_rep
            view = self._StacChainView(self, seq)
        else:
            print("[loop-gate] Model.loops.stac_server_dir not configured — spatial "
                  "gate OFF (declared in loop_edges.json); every candidate gets a bridge")
        for item in loop_results:
            i_hi, j_lo = int(item[1][0] + (item[1][1] - item[1][0]) // 2), \
                int(item[3][0] + (item[3][1] - item[3][0]) // 2)
            cand = by_pair.get((item[-1][0], item[-1][1])) if len(item) > 4 else None
            if cand is None:
                cand = {"i": i_hi, "j": j_lo, "sim": None, "source": "salad"}
            gate = {"verdict": "off", "rules": {}}
            if view is not None:
                gate = gate_mod.gate_frame_pair(int(cand['i']), int(cand['j']), view,
                                                lcfg['spatial'])
            if gate.get("verdict") == "reject":
                rejected.append({"item": [int(item[0]), list(item[1]), int(item[2]), list(item[3])],
                                 "candidate": cand, "gate": gate, "status": "rejected",
                                 "stage": "spatial_gate"})
                print(f"[loop-gate] candidate {cand['i']}<->{cand['j']} ({cand['source']}): "
                      f"REJECTED by the spatial gate — {gate.get('reason', '')}")
                continue
            planned.append((item, cand, gate))
        self._stac_loops_rejected = rejected
        self._stac_drop_chunk_cache()
        return planned

    def _stac_extra_frames(self, range_kf, n_extra, exclude):
        """§4.9: up to n_extra NON-keyframe frames spread over the real-frame
        span of a bridge window, blur-valid per frame_quality.json (the same
        filter the keyframe selector applies), sharpest per slot."""
        if n_extra <= 0:
            return []
        from loop_utils.metric_lock import real_frame_number
        lo = real_frame_number(self.img_list[range_kf[0]])
        hi = real_frame_number(self.img_list[range_kf[1] - 1])
        if hi <= lo:
            return []
        all_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                           glob.glob(os.path.join(self.img_dir, "*.png")))
        quality = {}
        fq = os.path.join(self.img_dir, "frame_quality.json")
        if os.path.exists(fq):
            import json as _json
            for e in _json.load(open(fq)).get("frames", []):
                quality[e["file"]] = (float(e.get("fft_score", 0.0)), bool(e.get("valid", True)))
        pool = []
        for p in all_paths:
            b = os.path.basename(p)
            if p in exclude or b in exclude:
                continue
            num = real_frame_number(p)
            if lo < num < hi:
                q = quality.get(b, (0.0, True))
                if q[1]:
                    pool.append((num, q[0], p))
        if not pool:
            return []
        edges = np.linspace(lo, hi, n_extra + 2)[1:-1]
        chosen = []
        for e in edges:
            near = [x for x in pool if abs(x[0] - e) <= (hi - lo) / (n_extra + 1)]
            if not near:
                continue
            best = max(near, key=lambda x: x[1])
            if best[2] not in chosen:
                chosen.append(best[2])
        return chosen

    def _stac_infer_bridges(self, planned):
        """One Omega pass per surviving candidate (model still loaded). Stores
        (item, prediction, meta) in loop_predict_list; the layout maps bridge
        frames to chunk frames."""
        from loop_utils.loop_bridges import bridge_layout, cfg_req
        lcfg = self._stac_loops_cfg()
        n_extra = int(cfg_req(lcfg, "bridge_extra_frames", "loops"))
        kf_set = set(self.img_list)
        for item, cand, gate in planned:
            ex1 = self._stac_extra_frames(item[1], n_extra, kf_set)
            ex2 = self._stac_extra_frames(item[3], n_extra, kf_set)
            pred = self.process_single_chunk(item[1], range_2=item[3], is_loop=True,
                                             extra_1=ex1, extra_2=ex2)
            layout = bridge_layout(item, self.chunk_indices, len(ex1), len(ex2))
            meta = {"candidate": cand, "gate": gate, "layout": layout}
            self.loop_predict_list.append((item, pred, meta))
            print(f"[loop-bridge] {cand['source']} {cand['i']}<->{cand['j']}: chunks "
                  f"{item[0]}<->{item[2]}, windows {item[1]}+{item[3]}, "
                  f"{len(ex1)}+{len(ex2)} extra frame(s)")
            torch.cuda.empty_cache()

    def _stac_ensure_bridge_anchors(self):
        """§4.1: plan `anchors_per_bridge` DA3 anchor frames inside EACH bridge
        window and extract the missing ones through the server's isolated DA3
        extractor (Model.metric_lock.anchor_extract, built by map_worker).
        Runs after the Omega model is released (the extractor needs the GPU)."""
        from loop_utils.loop_bridges import cfg_req
        from loop_utils.metric_lock import real_frame_number
        ml = self.config['Model'].get('metric_lock') or {}
        ae = ml.get('anchor_extract')
        lcfg = self._stac_loops_cfg()
        per = int(cfg_req(lcfg, "anchors_per_bridge", "loops"))
        anchor_dir = ml.get('anchor_dir')
        if not ae or not anchor_dir or per <= 0 or not self.loop_predict_list:
            if self.loop_predict_list:
                print("[loop-bridge] no anchor extractor configured — bridges take the "
                      "scale-graph scale of their parent chunks (recorded)")
            return
        wanted = []
        for item, _pred, _meta in self.loop_predict_list:
            for rng_ in (item[1], item[3]):
                span = rng_[1] - rng_[0]
                fr = [0.5] if per == 1 else [0.15 + 0.7 * i / (per - 1) for i in range(per)]
                for f in fr:
                    idx = rng_[0] + min(span - 1, int(round(f * (span - 1))))
                    wanted.append(self.img_list[idx])
        missing = sorted({os.path.basename(p) for p in wanted
                          if not os.path.exists(os.path.join(
                              anchor_dir, f"frame_{real_frame_number(p)}.npz"))})
        if not missing:
            print(f"[loop-bridge] bridge anchors: all {len(set(wanted))} planned frames "
                  f"already have DA3 depth")
            return
        mod = self._stac_server_module("reconstruction.da3_anchor")
        if mod is None:
            print("[loop-bridge] anchor extractor needs Model.loops.stac_server_dir — "
                  "bridges take the scale-graph scale (recorded)")
            return
        print(f"[loop-bridge] extracting DA3 depth for {len(missing)} bridge anchor frame(s)")
        mod.extract_anchor_depths(frames_dir=ae['frames_dir'], output_dir=ae['output_dir'],
                                  anchor_files=missing, model_id=ae['model_id'],
                                  python=ae['python'], log=print)

    def _stac_lock_bridges(self):
        """Metric-lock every bridge: own DA3 anchors when present, else the
        scale graph's value for its parent chunks (never 'the median') — and
        record which one it got."""
        from loop_utils.metric_lock import chunk_scale, apply_scale, real_frame_number
        ml = self.config['Model'].get('metric_lock') or {}
        anchor_dir = ml.get('anchor_dir')
        near_frac = float(ml.get('near_frac', 0.25))
        scales = getattr(self, '_stac_scales_applied', {}) or {}
        for li, (item, pred, meta) in enumerate(self.loop_predict_list):
            if pred.get('_stac_bridge_locked'):
                continue
            ka, (a0, a1), kb, (b0, b1) = item[0], item[1], item[2], item[3]
            lay = meta["layout"]
            nums = [None] * lay["n_frames"]
            for bl, g in zip(lay["bridge_a"], range(a0, a1)):
                nums[bl] = real_frame_number(self.img_list[g])
            for bl, g in zip(lay["bridge_b"], range(b0, b1)):
                nums[bl] = real_frame_number(self.img_list[g])
            nums = [n if n is not None else -1 for n in nums]
            s, n, ratios = (None, 0, [])
            if anchor_dir:
                s, n, ratios = chunk_scale(pred, nums, anchor_dir, near_frac=near_frac)
            if s is None:
                sa, sb = scales.get(ka), scales.get(kb)
                if sa is None or sb is None:
                    raise RuntimeError(f"[loop-bridge] bridge {ka}<->{kb}: no own anchors "
                                       f"and no graph scale for its parents — the scale "
                                       f"graph must have run before the bridges are locked")
                s = float(np.sqrt(sa * sb))
                src = "scale_graph_parents"
            else:
                src = "own_anchors"
            apply_scale(pred, s)
            pred['_stac_bridge_locked'] = True
            meta["lock"] = {"s": float(s), "n_anchors": int(n), "source": src,
                            "ratios": [float(r) for r in ratios]}
            print(f"[loop-bridge] {ka}<->{kb}: metric lock s={s:.4f} ({src}, {n} anchor(s))")

    def _stac_measure_loops(self, rigid):
        """Exact bridge↔chunk fits for every bridge (Sim3 → scale rows;
        rigid → SE(3) pose edges). Chunks are loaded two at a time."""
        from loop_utils.loop_bridges import measure_bridge
        lcfg = self._stac_loops_cfg()
        out = []
        for li, (item, pred, meta) in enumerate(self.loop_predict_list):
            ka, kb = item[0], item[2]
            da = self._stac_load_chunk(ka)
            db = self._stac_load_chunk(kb)
            meas = measure_bridge(pred, meta["layout"], da, db, lcfg, rigid=rigid)
            out.append(meas)
        self._stac_drop_chunk_cache()
        return out

    def _stac_scale_close(self, meas_sim3):
        """§5.1–5.3: re-solve the scale graph WITH the loop rows and apply the
        residual factor δ_k = s_v2/s_v1 to every chunk (and to bridges locked
        from their parents). Scale breaks are diagnosed, everything lands in
        scale_graph.json. Resume-safe (metric_lock.json + npy stamp)."""
        import json as _json
        from loop_utils.loop_bridges import cfg_req, loop_scale_row
        from loop_utils.metric_lock import (solve_scale_graph, scale_break_diagnosis,
                                            apply_scale, seam_residuals)
        scfg = self._stac_scale_cfg() or {}
        lcfg = self._stac_loops_cfg()
        ml = self.config['Model'].get('metric_lock') or {}
        inputs = getattr(self, '_stac_scale_inputs', None)
        s_v1 = getattr(self, '_stac_scales_applied', None)
        n_chunks = len(self.chunk_indices)
        sg_path = os.path.join(self.output_dir, "scale_graph.json")
        stamp_key = '_stac_loop_scale_applied'
        if inputs is None or s_v1 is None:
            print("[scale-graph] metric lock did not run this session (resume) — loop "
                  "rows cannot be added to a lock that was applied in a previous run; "
                  "the scale graph stays as locked")
            return
        sigma_loop = float(cfg_req(scfg, "sigma_loop", "scale"))
        tol_log = float(cfg_req(lcfg, "scale_tol_log", "loops"))
        sb_factor = float(cfg_req(lcfg, "scale_break_sigma_factor", "loops"))
        loop_rel, loop_rows = {}, []
        for li, ((item, pred, meta), meas) in enumerate(zip(self.loop_predict_list, meas_sim3)):
            ka, kb = int(item[0]), int(item[2])
            if not meas.get("ok") or ka == kb:
                continue
            # measured on v1-locked chunks → express in RAW terms for the joint solve
            log_r_meas = loop_scale_row(meas["s_ab"])
            log_r_raw = log_r_meas + np.log(s_v1[kb]) - np.log(s_v1[ka])
            sig = sigma_loop
            is_break = abs(log_r_meas) > tol_log
            if is_break:
                sig *= sb_factor
            key = (ka, kb)
            if key in loop_rel:           # several bridges between the same chunks: mean
                prev = loop_rel[key]
                loop_rel[key] = ((prev[0] + log_r_raw) / 2.0, min(prev[1], sig))
            else:
                loop_rel[key] = (log_r_raw, sig)
            loop_rows.append({"bridge": li, "chunks": [ka, kb], "s_ab": meas["s_ab"],
                              "log_r_measured": float(log_r_meas), "sigma": float(sig),
                              "scale_break": bool(is_break)})
        absolute = list(inputs.get("absolute") or [])
        s_v2 = solve_scale_graph(inputs["s_da3"], inputs["n_anchors"], inputs["seam_rel"],
                                 n_chunks, sigma_seam=inputs["sigma_seam"],
                                 sigma_anchor=inputs["sigma_anchor"],
                                 loop_rel=loop_rel, absolute=absolute)
        breaks = []
        for row in loop_rows:
            if row["scale_break"]:
                d = scale_break_diagnosis(inputs["s_da3"], inputs["n_anchors"], inputs["seam_rel"],
                                          n_chunks, loop_rel, tuple(row["chunks"]),
                                          inputs["sigma_seam"], inputs["sigma_anchor"],
                                          absolute=absolute,
                                          localisation_gap=float(cfg_req(
                                              scfg, "break_localisation_gap", "scale")))
                d["bridge"] = row["bridge"]
                breaks.append(d)
                print(f"[scale-graph] SCALE BREAK on loop {row['chunks']}: suspect seam "
                      f"{d['suspect_seam']}->{(d['suspect_seam'] or 0) + 1}, jump "
                      f"{d['jump_pct']:.2f}% "
                      f"({'localised' if d.get('localised') else 'AMBIGUOUS — one cycle, weak absolute witnesses'}; "
                      f"see scale_graph.json)")
        delta = {}
        for k in range(n_chunks):
            sv1 = s_v1.get(k)
            if sv1 is None or not np.isfinite(s_v2[k]) or s_v2[k] <= 0:
                delta[k] = 1.0
            else:
                delta[k] = float(s_v2[k] / sv1)
        # apply δ to the chunks (in place, stamped) and to parent-locked bridges
        n_moved = 0
        for k in range(n_chunks):
            path = os.path.join(self.result_unaligned_dir, f"chunk_{k}.npy")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True).item()
            if data.get(stamp_key):
                continue
            if abs(delta[k] - 1.0) > 1e-12:
                apply_scale(data, delta[k])
                n_moved += 1
            data[stamp_key] = float(delta[k])
            np.save(path, data)
        for item, pred, meta in self.loop_predict_list:
            if meta.get("lock", {}).get("source") == "scale_graph_parents":
                ka, kb = int(item[0]), int(item[2])
                f = float(np.sqrt(delta[ka] * delta[kb]))
                if abs(f - 1.0) > 1e-12:
                    apply_scale(pred, f)
                meta["lock"]["delta_parents"] = f
        self._stac_scales_applied = {k: float(s_v1.get(k, 1.0) * delta[k]) for k in range(n_chunks)}
        self._stac_scale_delta = delta
        if self._stac_authority_cfg():
            _acfg = self._stac_authority_cfg()
            _used = max(abs(float(np.log(v))) for v in delta.values()) if delta else 0.0
            _fr = _used / float(_acfg["scale_graph_max_log"])
            self._stac_authority_record("scale_graph", _fr,
                                        _fr > float(_acfg["saturation_warn"]),
                                        {"max_log": float(_acfg["scale_graph_max_log"]),
                                         "used_max_log": _used})
        # verifier bookkeeping: s_frames from the drift stage (when it ran)
        s_frames = getattr(self, '_stac_drift_frames', None)
        rep = {"version": 1,
               "n_chunks": n_chunks,
               "sigma_seam": inputs["sigma_seam"], "sigma_anchor": inputs["sigma_anchor"],
               "sigma_loop": sigma_loop,
               "s_v1_anchors_seams": {str(k): float(v) for k, v in s_v1.items()},
               "s_v2_with_loops": {str(k): float(s_v2[k]) for k in range(n_chunks)},
               "delta_applied": {str(k): float(v) for k, v in delta.items()},
               "loop_rows": loop_rows,
               "absolute_rows": [{"chunk": int(a[0]), "log_s": float(a[1]), "sigma": float(a[2]),
                                  "source": (a[3] if len(a) > 3 else "unspecified")}
                                 for a in absolute],
               "seam_residuals_log_v1": seam_residuals([s_v1.get(k, 1.0) for k in range(n_chunks)],
                                                        inputs["seam_rel"]),
               "seam_residuals_log_v2": seam_residuals(s_v2, inputs["seam_rel"]),
               "scale_breaks": breaks,
               "s_frames": ({str(k): [float(x) for x in v] for k, v in s_frames.items()}
                            if s_frames else None)}
        with open(sg_path, "w") as f:
            _json.dump(rep, f, indent=1)
        dmax = max(abs(np.log(v)) for v in delta.values()) if delta else 0.0
        print(f"[scale-graph] ✅ {len(loop_rows)} loop row(s) + {len(absolute)} absolute row(s) "
              f"closed the scale: {n_moved} chunk(s) re-scaled, max |δ| "
              f"{(np.exp(dmax) - 1) * 100:.2f}% → scale_graph.json")
        # metric_lock.json: record the loop stage for the resume guard
        _ml_path = os.path.join(self.output_dir, "metric_lock.json")
        if os.path.exists(_ml_path):
            try:
                _rep = _json.load(open(_ml_path))
            except Exception as _e:
                raise RuntimeError(f"metric_lock.json unreadable ({_e}) — cannot record the "
                                   f"loop scale stage; the lock state must be trustworthy")
            _rep["loop_stage"] = {"delta": {str(k): float(v) for k, v in delta.items()},
                                  "n_loop_rows": len(loop_rows)}
            for k, v in self._stac_scales_applied.items():
                _rep.setdefault("chunks", {}).setdefault(str(k), {})["s_applied"] = float(v)
            with open(_ml_path, "w") as f:
                _json.dump(_rep, f, indent=1)

    def _stac_verify_loops(self, meas_rigid):
        """§4.2 steps 1–4 on the SE(3) measurements; builds the pose-edge list
        (every edge carries σ) and the loop report. loop_enable_opt then depends
        on at least one edge with σ ≤ max_edge_sigma_m — the exact-seam skip is
        gone (claude_stac.txt DoD)."""
        import json as _json
        from loop_utils.loop_bridges import (verify_loop, attention_score, cfg_req,
                                             save_loop_report)
        lcfg = self._stac_loops_cfg()
        max_sig = float(cfg_req(lcfg, "max_edge_sigma_m", "loops"))
        starved_sig = float(cfg_req(lcfg, "starved_sigma_m", "loops"))
        sem_path = os.path.join(self.output_dir, "loop_semantics.json")
        semantics = None
        if os.path.exists(sem_path):
            semantics = _json.load(open(sem_path))
        from loop_utils.metric_lock import real_frame_number
        edges = list(getattr(self, '_stac_loops_rejected', []) or [])
        self.loop_sim3_list = []
        self._stac_loop_edges = []
        self._stac_loop_edges_kf = []
        gcfg = self._stac_graph_cfg()
        for li, ((item, pred, meta), meas) in enumerate(zip(self.loop_predict_list, meas_rigid)):
            ka, kb = int(item[0]), int(item[2])
            cand = meta["candidate"]
            sem = None
            if semantics is not None:
                fr = semantics.get("frames", {})
                ni, nj = real_frame_number(self.img_list[cand['i']]), \
                    real_frame_number(self.img_list[cand['j']])
                sa = fr.get(str(ni), {}).get("structural")
                sb = fr.get(str(nj), {}).get("structural")
                sem = {"a": sa, "b": sb}
            att = None
            if bool(cfg_req(lcfg, "attention_verify", "loops")):
                att = attention_score(pred.get("camera_and_register_tokens"), meta["layout"])
            v = verify_loop(meas, lcfg, semantic=sem, attention=att, spatial=meta.get("gate"))
            if not meas.get("ok"):
                # starved exact fit → the VENDOR coarse fit, recorded as low confidence
                coarse = self._stac_vendor_bridge_fit(item, pred, meta)
                if coarse is not None:
                    s_ab, R_ab, t_ab = coarse
                    v = {"status": "accepted", "sigma_m": starved_sig, "fallback": "vendor_point_map_fit",
                         "reasons": ["exact correspondences starved — vendor point-map fit, "
                                     f"σ={starved_sig} m (low confidence)"], "checks": v.get("checks", {})}
                    meas = dict(meas, ok=True, s_ab=float(s_ab), R_ab=np.asarray(R_ab).tolist(),
                                t_ab=np.asarray(t_ab).tolist())
            edge = {"bridge": li, "item": [ka, list(item[1]), kb, list(item[3])],
                    "candidate": cand, "gate": meta.get("gate"), "lock": meta.get("lock"),
                    "measurement": {k_: v_ for k_, v_ in meas.items() if k_ != "sides"},
                    "sides": meas.get("sides"), "verdict": v, "status": v["status"],
                    "stage": "verification", "intra_chunk": ka == kb}
            edges.append(edge)
            if v["status"] in ("accepted", "scale_break") and gcfg is not None:
                # KEYFRAME edge for the SE(3) graph (§4.3): the relative pose
                # between the two window centres as ONE Omega pass measured it
                # (the bridge's own metric-locked extrinsics); σ_t = the verified
                # exact-fit residual, σ_rot from config. Intra-chunk bridges
                # give edges too (no scale row, but a pose constraint).
                lay = meta["layout"]
                ext_b = np.asarray(pred['extrinsic'])
                if ext_b.ndim == 4:
                    ext_b = ext_b[0]
                la = lay["bridge_a"][len(lay["bridge_a"]) // 2]
                lb_ = lay["bridge_b"][len(lay["bridge_b"]) // 2]
                gi = int(item[1][0] + len(lay["bridge_a"]) // 2)
                gj = int(item[3][0] + len(lay["bridge_b"]) // 2)
                Ei = np.asarray(ext_b[la], np.float64)
                Ej = np.asarray(ext_b[lb_], np.float64)
                Z = np.linalg.inv(Ei) @ Ej
                kf_edge = {"i": gi, "j": gj, "Z": Z.tolist(), "sigma_m": float(v["sigma_m"]),
                           "sigma_deg": float(cfg_req(gcfg, "loop_sigma_rot_deg", "graph")),
                           "bridge": li, "status": v["status"], "chunks": [ka, kb]}
                self._stac_loop_edges_kf.append(kf_edge)
                edge["keyframe_edge"] = kf_edge
            if v["status"] in ("accepted", "scale_break") and ka != kb:
                self._stac_loop_edges.append({"a": ka, "b": kb, "R_ab": np.asarray(meas["R_ab"]),
                                              "t_ab": np.asarray(meas["t_ab"]),
                                              "sigma_m": float(v["sigma_m"]),
                                              "status": v["status"], "bridge": li})
                if float(v["sigma_m"]) <= max_sig:
                    self.loop_sim3_list.append((ka, kb, (1.0, np.asarray(meas["R_ab"]),
                                                         np.asarray(meas["t_ab"]))))
            tag = ("ACCEPTED" if v["status"] == "accepted" else v["status"].upper())
            print(f"[loop-verify] bridge {li} chunks {ka}<->{kb} ({cand['source']} "
                  f"{cand['i']}<->{cand['j']}): {tag} σ={v.get('sigma_m', float('nan')):.4f} m "
                  f"{'; '.join(v.get('reasons', []))}")
        n_good = sum(1 for e in self._stac_loop_edges if e["sigma_m"] <= max_sig)
        self.loop_enable_opt = bool(self.loop_enable and n_good > 0)
        gate_mod = self._stac_server_module("reconstruction.loops.spatial_gate")
        save_loop_report(os.path.join(self.output_dir, "loop_edges.json"), edges,
                         extra={"spatial_gate": "on" if gate_mod is not None else "off",
                                "attention_verify": bool(cfg_req(lcfg, "attention_verify", "loops")),
                                "semantics": semantics is not None,
                                "max_edge_sigma_m": max_sig,
                                "n_edges_usable": n_good,
                                "loop_enable_opt": self.loop_enable_opt,
                                "provisional_seams": getattr(self, '_stac_provisional_seams', None)})
        print(f"[loop-verify] {n_good} usable edge(s) with σ ≤ {max_sig} m → loop optimizer "
              f"{'ON' if self.loop_enable_opt else 'OFF (no trustworthy edge)'}; loop_edges.json")

    # ══════════════════════════════════════════════════════════════════════
    # STAC F2 — KEYFRAME SE(3) POSE GRAPH, INTRINSIC UNCERTAINTY, AUTHORITY
    # (claude_stac.txt §4.3, §4.7, §4.8). Replaces the vendor's per-chunk Sim3
    # loop optimizer in the vggtomega path (Model.graph present).
    # ══════════════════════════════════════════════════════════════════════
    def _stac_graph_cfg(self):
        return self.config['Model'].get('graph')

    def _stac_authority_cfg(self):
        return self.config['Model'].get('authority')

    def _stac_uncertainty(self):
        """§4.8: every overlap frame is predicted twice (chunk k and k+1). The
        per-pixel disagreement between the two ALIGNED copies is the model's
        own uncertainty for that frame — persisted as uncert/uncert_<frame>.npy
        (float32 metres) plus uncertainty.json (per-frame median / p90), and
        used as a σ component of the odometry edges (§4.3) and of the witness
        stage (F3). Frames predicted once carry the session median, declared
        as such. Resume-safe (uncertainty.json + files)."""
        import json as _json
        from loop_utils.metric_lock import real_frame_number
        udir = os.path.join(self.output_dir, "uncert")
        os.makedirs(udir, exist_ok=True)
        rep_path = os.path.join(self.output_dir, "uncertainty.json")
        if os.path.exists(rep_path):
            try:
                prev = _json.load(open(rep_path))
                if prev.get("chunk_indices") == [list(ci) for ci in self.chunk_indices]:
                    self._stac_uncert = {int(k): v for k, v in prev["frames"].items()}
                    self._stac_uncert_median = float(prev["session_median_m"])
                    print(f"[uncertainty] resume: {len(self._stac_uncert)} frame(s) from "
                          f"uncertainty.json")
                    return
            except (ValueError, KeyError) as _e:
                print(f"[uncertainty] uncertainty.json unreadable ({_e}) — re-measuring")
        per_frame = {}
        prev_tail = None
        for k, (start, end) in enumerate(self.chunk_indices):
            data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                           allow_pickle=True).item()
            wp = np.asarray(data['world_points']); wp = wp[0] if wp.ndim == 5 else wp
            cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
            if prev_tail is not None and prev_tail[0] == k - 1:
                for local, g in enumerate(range(start, end)):
                    if g not in prev_tail[1]:
                        continue
                    p_prev, c_prev = prev_tail[1][g]
                    ok = (c_prev > 1e-5) & (cf[local] > 1e-5)
                    d = np.full(wp.shape[1:3], np.nan, np.float32)
                    if ok.any():
                        d[ok] = np.linalg.norm(wp[local][ok].astype(np.float64)
                                               - p_prev[ok].astype(np.float64), axis=1)
                    num = real_frame_number(self.img_list[g])
                    np.save(os.path.join(udir, f"uncert_{num}.npy"), d)
                    vals = d[np.isfinite(d)]
                    per_frame[g] = {"frame": int(num), "median_m": float(np.median(vals)) if len(vals) else None,
                                    "p90_m": float(np.percentile(vals, 90)) if len(vals) else None,
                                    "n_px": int(len(vals)), "witnesses": 2}
            if k + 1 < len(self.chunk_indices):
                nxt0 = self.chunk_indices[k + 1][0]
                prev_tail = (k, {g: (wp[g - start].astype(np.float32), cf[g - start])
                                 for g in range(max(nxt0, start), end)})
            else:
                prev_tail = None
            del data
        meds = [v["median_m"] for v in per_frame.values() if v["median_m"] is not None]
        session_med = float(np.median(meds)) if meds else 0.0
        self._stac_uncert = per_frame
        self._stac_uncert_median = session_med
        with open(rep_path, "w") as f:
            _json.dump({"version": 1, "chunk_indices": [list(ci) for ci in self.chunk_indices],
                        "session_median_m": session_med, "n_shared_frames": len(per_frame),
                        "frames": {str(k): v for k, v in per_frame.items()},
                        "note": "frames predicted once carry the session median"}, f, indent=1)
        print(f"[uncertainty] {len(per_frame)} shared frame(s): two-copy disagreement median "
              f"{session_med * 100:.2f} cm → uncert/, uncertainty.json")

    def _stac_ensemble_uncertainty(self):
        """§4.8 (optional, Model.certify.ensemble_offset_frames > 0): a second
        Omega pass with chunk boundaries SHIFTED by that many frames gives a
        third prediction of every frame. Each ensemble chunk is glued to the
        main chain by an exact rigid fit on its frames' owner copies and the
        per-pixel disagreement joins the uncertainty files (witnesses: 3).
        Runs while the model is loaded; OFF by default (certification runs)."""
        cert = self.config['Model'].get('certify') or {}
        off = int(cert.get('ensemble_offset_frames', 0) or 0)
        if off <= 0 or len(self.chunk_indices) < 2:
            return
        from loop_utils.metric_lock import robust_rigid, real_frame_number, frame_owner
        edir = os.path.join(self.output_dir, "_tmp_results_ensemble")
        os.makedirs(edir, exist_ok=True)
        N = len(self.img_list)
        step = self.chunk_size - self.overlap
        ranges = []
        start = off
        while start + 2 < N:
            end = min(start + self.chunk_size, N)
            ranges.append((start, end))
            if end >= N:
                break
            start += step
        self._stac_ensemble_ranges = ranges
        for e_idx, (a, b) in enumerate(ranges):
            path = os.path.join(edir, f"chunk_{e_idx}.npy")
            if os.path.exists(path):
                continue
            pred = self.model.infer_chunk(self.img_list[a:b])
            for key in list(pred.keys()):
                if isinstance(pred[key], torch.Tensor):
                    pred[key] = pred[key].cpu().numpy().squeeze(0)
            pred['depth'] = np.squeeze(pred['depth'])
            pred['_stac_range'] = [int(a), int(b)]
            np.save(path, pred)
            torch.cuda.empty_cache()
        print(f"[uncertainty] ensemble pass: {len(ranges)} chunk(s) with boundaries shifted "
              f"by {off} frame(s) → {edir}")

    def _stac_ensemble_apply(self):
        """Second half of the ensemble witness (after the main chain exists):
        glue each ensemble chunk to the aligned main copies and merge the
        disagreement into uncert/ + uncertainty.json (witnesses: 3)."""
        cert = self.config['Model'].get('certify') or {}
        off = int(cert.get('ensemble_offset_frames', 0) or 0)
        ranges = getattr(self, '_stac_ensemble_ranges', None)
        if off <= 0 or not ranges:
            return
        import json as _json
        from loop_utils.metric_lock import (robust_rigid, real_frame_number, frame_owner,
                                            apply_scale)
        from loop_utils.loop_bridges import robust_sim3
        edir = os.path.join(self.output_dir, "_tmp_results_ensemble")
        udir = os.path.join(self.output_dir, "uncert")
        N = len(self.img_list)
        owner = frame_owner(self.chunk_indices, N)
        rep_path = os.path.join(self.output_dir, "uncertainty.json")
        rep = _json.load(open(rep_path)) if os.path.exists(rep_path) else {"frames": {}}
        n_frames = 0
        for e_idx, (a, b) in enumerate(ranges):
            pred = np.load(os.path.join(edir, f"chunk_{e_idx}.npy"), allow_pickle=True).item()
            wp = np.asarray(pred['world_points']); wp = wp[0] if wp.ndim == 5 else wp
            cf = np.asarray(pred['world_points_conf']).reshape(wp.shape[:3])
            # owner copies of the same frames under the FINAL chain
            P, Q = [], []
            owners = {}
            for local, g in enumerate(range(a, b)):
                k = int(owner[g])
                d = self._stac_load_chunk_aligned(k)
                wpk = np.asarray(d['world_points']); wpk = wpk[0] if wpk.ndim == 5 else wpk
                cfk = np.asarray(d['world_points_conf']).reshape(wpk.shape[:3])
                lk = g - self.chunk_indices[k][0]
                ok = (cf[local] > 1e-5) & (cfk[lk] > 1e-5)
                idx = np.flatnonzero(ok.reshape(-1))
                if len(idx) > 4000:
                    idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
                P.append(wp[local].reshape(-1, 3)[idx]); Q.append(wpk[lk].reshape(-1, 3)[idx])
                owners[g] = (wpk[lk].astype(np.float32), cfk[lk])
            fit = robust_sim3(np.concatenate(P), np.concatenate(Q), min_points=1000)
            if fit is None:
                print(f"[uncertainty] ensemble chunk {e_idx}: glue starved — skipped (declared)")
                continue
            s_, R_, t_, res, n_ = fit
            for local, g in enumerate(range(a, b)):
                wpk, cfk = owners[g]
                pe = s_ * (wp[local].reshape(-1, 3).astype(np.float64) @ R_.T) + t_
                pe = pe.reshape(wp.shape[1:])
                ok = (cf[local] > 1e-5) & (cfk > 1e-5)
                d3 = np.full(wp.shape[1:3], np.nan, np.float32)
                if ok.any():
                    d3[ok] = np.linalg.norm(pe[ok] - wpk[ok].astype(np.float64), axis=1)
                num = real_frame_number(self.img_list[g])
                fpath = os.path.join(udir, f"uncert_{num}.npy")
                if os.path.exists(fpath):
                    d2 = np.load(fpath)
                    both = np.isfinite(d2) & np.isfinite(d3)
                    merged = np.where(both, np.maximum(d2, d3), np.where(np.isfinite(d2), d2, d3))
                else:
                    merged = d3
                np.save(fpath, merged.astype(np.float32))
                vals = merged[np.isfinite(merged)]
                rep["frames"][str(g)] = {"frame": int(num),
                                         "median_m": float(np.median(vals)) if len(vals) else None,
                                         "p90_m": float(np.percentile(vals, 90)) if len(vals) else None,
                                         "n_px": int(len(vals)),
                                         "witnesses": 3 if str(g) in rep["frames"] else 2}
                n_frames += 1
        meds = [v["median_m"] for v in rep["frames"].values() if v.get("median_m") is not None]
        rep["session_median_m"] = float(np.median(meds)) if meds else 0.0
        rep["ensemble_offset_frames"] = off
        with open(rep_path, "w") as f:
            _json.dump(rep, f, indent=1)
        self._stac_uncert = {int(k): v for k, v in rep["frames"].items()}
        self._stac_uncert_median = float(rep["session_median_m"])
        print(f"[uncertainty] ensemble witness merged into {n_frames} frame(s)")

    def _stac_load_chunk_aligned(self, k):
        cache = getattr(self, '_stac_chunk_cache_aligned', None)
        if cache is None:
            cache = self._stac_chunk_cache_aligned = {}
        if k in cache:
            return cache[k]
        data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{k}.npy"),
                       allow_pickle=True).item()
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[k] = data
        return data

    def _stac_drop_aligned_cache(self):
        if getattr(self, '_stac_chunk_cache_aligned', None):
            self._stac_chunk_cache_aligned.clear()

    def _stac_holdout_pairs(self, gcfg):
        """Held-out surface pairs (frames f, f+d — d from graph.holdout_offsets,
        every graph.holdout_stride-th f) with exact-surface correspondences
        under the CURRENT chain: the judge of the pose graph (same strategy as
        chunk_field_verdict). Returns [(f, g, p[n,3], q[n,3])] in world coords."""
        from loop_utils.metric_lock import surface_pair_correspondences, frame_owner
        from loop_utils.loop_bridges import cfg_req
        offsets = [int(x) for x in cfg_req(gcfg, "holdout_offsets", "graph")]
        stride = int(cfg_req(gcfg, "holdout_stride", "graph"))
        n_samp = int(cfg_req(gcfg, "holdout_samples", "graph"))
        N = len(self.img_list)
        owner = frame_owner(self.chunk_indices, N)
        wanted = set()
        for f in range(0, N, max(stride, 1)):
            for d in offsets:
                if f + d < N:
                    wanted.add(f); wanted.add(f + d)
        cache = {}
        for k, (start, end) in enumerate(self.chunk_indices):
            need = [g for g in range(start, end) if g in wanted and owner[g] == k]
            if not need:
                continue
            data = self._stac_load_chunk_aligned(k)
            wp = np.asarray(data['world_points']); wp = wp[0] if wp.ndim == 5 else wp
            cf = np.asarray(data['world_points_conf']).reshape(wp.shape[:3])
            ext = np.asarray(data['extrinsic']); K = np.asarray(data['intrinsic'])
            for g in need:
                local = g - start
                c2w = self._stac_aligned_pose(k, local, ext[local])
                cache[g] = (wp[local].astype(np.float32), cf[local].astype(np.float32),
                            np.linalg.inv(c2w), K[local])
        self._stac_drop_aligned_cache()
        pairs = []
        for f in range(0, N, max(stride, 1)):
            for d in offsets:
                g = f + d
                if f not in cache or g not in cache:
                    continue
                pq = surface_pair_correspondences(cache[f][0], cache[f][1],
                                                  cache[g][0], cache[g][1],
                                                  cache[g][2], cache[g][3],
                                                  max_samples=n_samp)
                if pq is not None:
                    pairs.append((f, g, pq[0], pq[1]))
        return pairs

    def _stac_pose_graph(self):
        """§4.3: the keyframe SE(3) graph over the ALIGNED chunks — odometry
        from the chain (σ from seam residuals + §4.8 uncertainty), the verified
        loop edges (σ measured), a weak gravity prior; solved by
        loop_utils.pose_graph; §4.7 authority (veto of loop edges demanding
        more than the drift budget, saturation report); gated by loop gain and
        held-out surface pairs; applied as a RIGID move per frame (points and
        camera together, both copies of a shared frame) — depth per ray and
        provenance invariant. Resume-safe (pose_graph.json + npy stamp)."""
        gcfg = self._stac_graph_cfg()
        if gcfg is None or self._stac_loops_cfg() is None or len(self.chunk_indices) < 1:
            return
        import json as _json
        from loop_utils.loop_bridges import cfg_req
        from loop_utils.pose_graph import PoseGraph
        from loop_utils.metric_lock import frame_owner, se3_matrices
        from loop_utils.lie import se3_log, se3_inv
        N = len(self.img_list)
        owner = frame_owner(self.chunk_indices, N)
        pg_path = os.path.join(self.output_dir, "pose_graph.json")
        acfg = self._stac_authority_cfg() or {}
        X = None
        report = None
        if os.path.exists(pg_path):
            try:
                prev = _json.load(open(pg_path))
                if prev.get("chunk_indices") == [list(ci) for ci in self.chunk_indices] \
                        and prev.get("verdict") in ("APPLY", "IDENTITY"):
                    X = se3_matrices(np.asarray(prev["xi"], np.float64)) if prev["verdict"] == "APPLY" \
                        else np.tile(np.eye(4), (N, 1, 1))
                    print(f"[pose-graph] resume: {prev['verdict']} loaded from pose_graph.json")
            except (ValueError, KeyError) as _e:
                print(f"[pose-graph] pose_graph.json unreadable ({_e}) — re-solving")
        if X is None:
            edges_kf = list(getattr(self, '_stac_loop_edges_kf', []) or [])
            if not edges_kf and not bool(cfg_req(gcfg, "run_without_loops", "graph")):
                report = {"verdict": "IDENTITY", "reason": "no verified loop edge — nothing "
                                                            "closes; odometry alone would only "
                                                            "smooth what the seams already fixed",
                          "chunk_indices": [list(ci) for ci in self.chunk_indices],
                          "n_loop_edges": 0}
                with open(pg_path, "w") as f:
                    _json.dump(report, f, indent=1)
                print(f"[pose-graph] IDENTITY: {report['reason']}")
                return
            # 1) initial poses per frame (owner copy, current chain)
            T0 = np.zeros((N, 4, 4))
            for k, (start, end) in enumerate(self.chunk_indices):
                data = self._stac_load_chunk_aligned(k)
                ext = np.asarray(data['extrinsic'])
                for local, g in enumerate(range(start, end)):
                    if owner[g] == k:
                        T0[g] = self._stac_aligned_pose(k, local, ext[local])
            self._stac_drop_aligned_cache()
            # 2) σ per frame from the two-copy uncertainty
            unc = getattr(self, '_stac_uncert', {}) or {}
            unc_med = float(getattr(self, '_stac_uncert_median', 0.0) or 0.0)

            def sig_u(g):
                v = unc.get(g)
                return float(v["median_m"]) if v and v.get("median_m") is not None else unc_med

            seam_res = getattr(self, '_stac_seam_residuals', {}) or {}
            s_intra = float(cfg_req(gcfg, "sigma_odo_intra_m", "graph"))
            s_intra_deg = float(cfg_req(gcfg, "sigma_odo_intra_deg", "graph"))
            pg = PoseGraph(T0, gcfg)
            for g in range(N - 1):
                Z = se3_inv(T0[g]) @ T0[g + 1]
                s_t = np.sqrt(s_intra ** 2 + sig_u(g) ** 2 + sig_u(g + 1) ** 2)
                if owner[g] != owner[g + 1]:
                    s_t = np.sqrt(s_t ** 2 + float(seam_res.get(int(owner[g]), 0.0)) ** 2)
                pg.add_relative(g, g + 1, Z, s_intra_deg, s_t, huber=False, tag="odo")
            loop_ids = []
            for e in edges_kf:
                eid = pg.add_relative(int(e["i"]), int(e["j"]), np.asarray(e["Z"]),
                                      float(e["sigma_deg"]), float(e["sigma_m"]), huber=True,
                                      tag=f"loop:{e['bridge']}")
                loop_ids.append((eid, e))
            # 3) gravity prior: the consensus camera-down of the chain (orient's estimator)
            downs = T0[:, :3, 1] / (np.linalg.norm(T0[:, :3, 1], axis=1, keepdims=True) + 1e-12)
            g_down = downs.mean(0); g_down /= (np.linalg.norm(g_down) + 1e-12)
            s_grav = float(cfg_req(gcfg, "sigma_gravity_deg", "graph"))
            for g in range(N):
                pg.add_gravity(g, g_down, s_grav)
            # held-out judge BEFORE
            held = self._stac_holdout_pairs(gcfg)
            before_loop = pg.edge_residuals("loop")
            loop_before = float(np.sum([r["t_m"] for r in before_loop.values()])) if before_loop else 0.0
            # 4) solve with the §4.7 veto loop
            vetoed = []
            budget_cfg = (self._stac_loops_cfg() or {}).get("spatial") or {}
            centres = T0[:, :3, 3]
            for _round in range(len(loop_ids) + 1):
                pg.solve(log=print)
                Xc = pg.corrections()
                edge_res = pg.edge_residuals("loop")
                offenders = []
                for eid, e in loop_ids:
                    if not pg._edges[eid]["active"]:
                        continue
                    i, j = int(e["i"]), int(e["j"])
                    lo, hi = min(i, j), max(i, j)
                    L = float(np.linalg.norm(np.diff(centres[lo:hi + 1], axis=0), axis=1).sum())
                    delta = max(float(budget_cfg["drift_floor_m"]),
                                float(budget_cfg["drift_rate_m_per_m"]) * L)
                    # what the edge DEMANDS: the correction it obtained at its
                    # endpoints plus what it still asks for (its residual —
                    # Huber lets a liar keep asking without being obeyed)
                    need = max(float(np.linalg.norm(Xc[i][:3, 3])), float(np.linalg.norm(Xc[j][:3, 3])),
                               float(edge_res.get(eid, {}).get("t_m", 0.0)))
                    if need > delta:
                        offenders.append((need - delta, eid, e, need, delta))
                if not offenders:
                    break
                offenders.sort(reverse=True)
                _, eid, e, need, delta = offenders[0]
                pg.deactivate(eid)
                vetoed.append({"bridge": e["bridge"], "i": e["i"], "j": e["j"],
                               "correction_m": need, "budget_m": delta,
                               "reason": "loop edge demands a correction beyond the drift "
                                         "budget — not drift, a false loop (§4.7)"})
                print(f"[pose-graph] VETO loop edge {e['i']}<->{e['j']} (bridge {e['bridge']}): "
                      f"{need * 100:.0f} cm demanded > budget {delta * 100:.0f} cm — re-solving")
            Xc = pg.corrections()
            after_loop_res = pg.edge_residuals("loop")
            loop_after = float(np.sum([r["t_m"] for r in after_loop_res.values()])) if after_loop_res else 0.0
            # 5) gates
            def _held_median(Xm):
                vals = []
                for f_, g_, p, q in held:
                    p2 = p @ Xm[f_][:3, :3].T + Xm[f_][:3, 3]
                    q2 = q @ Xm[g_][:3, :3].T + Xm[g_][:3, 3]
                    vals.append(float(np.median(np.linalg.norm(p2 - q2, axis=1))))
                return float(np.median(vals)) if vals else float("nan")
            held_before = _held_median(np.tile(np.eye(4), (N, 1, 1)))
            held_after = _held_median(Xc)
            gain = (1.0 - loop_after / loop_before) if loop_before > 0 else 0.0
            min_gain = float(cfg_req(gcfg, "min_loop_gain", "graph"))
            max_deg = float(cfg_req(gcfg, "max_seam_degradation_m", "graph"))
            n_active = sum(1 for eid, _ in loop_ids if pg._edges[eid]["active"])
            ok_gain = (gain >= min_gain) if n_active > 0 else False
            ok_held = (not np.isfinite(held_before)) or (held_after <= held_before + max_deg)
            verdict = "APPLY" if (ok_gain and ok_held) else "IDENTITY"
            # §4.7 authority: fraction used vs the declared maximum
            t_mag = np.linalg.norm(Xc[:, :3, 3], axis=1)
            r_mag = np.array([np.degrees(np.linalg.norm(se3_log(M)[:3])) for M in Xc])
            a_max_m = float(cfg_req(acfg, "pose_graph_max_m", "authority"))
            a_max_deg = float(cfg_req(acfg, "pose_graph_max_deg", "authority"))
            frac = max(float(t_mag.max()) / a_max_m if a_max_m > 0 else 0.0,
                       float(r_mag.max()) / a_max_deg if a_max_deg > 0 else 0.0)
            saturated = frac > float(cfg_req(acfg, "saturation_warn", "authority"))
            if frac > 1.0:
                verdict = "IDENTITY"
            xi = np.array([se3_log(M) for M in Xc])
            report = {"chunk_indices": [list(ci) for ci in self.chunk_indices],
                      "verdict": verdict,
                      "gates": {"loop_gain": {"value": gain, "min": min_gain, "passed": ok_gain,
                                              "loop_residual_before_m": loop_before,
                                              "loop_residual_after_m": loop_after},
                                "holdout_surface_pairs": {"n_pairs": len(held),
                                                          "median_before_m": held_before,
                                                          "median_after_m": held_after,
                                                          "max_degradation_m": max_deg,
                                                          "passed": ok_held}},
                      "authority": {"stage": "pose_graph", "max_m": a_max_m, "max_deg": a_max_deg,
                                    "used_max_m": float(t_mag.max()), "used_max_deg": float(r_mag.max()),
                                    "fraction_used": frac, "saturated": bool(saturated),
                                    "exceeded": bool(frac > 1.0)},
                      "n_loop_edges": len(loop_ids), "n_loop_edges_active": n_active,
                      "vetoed": vetoed, "gravity_down": g_down.tolist(),
                      "solver": {k: v for k, v in pg.report.items() if k != "per_edge"},
                      "loop_edges_after": {str(k): v for k, v in after_loop_res.items()},
                      "xi": xi.tolist() if verdict == "APPLY" else None}
            with open(pg_path, "w") as f:
                _json.dump(report, f, indent=1)
            print(f"[pose-graph] loop residual {loop_before * 100:.1f} → {loop_after * 100:.1f} cm "
                  f"(gain {gain * 100:.0f}%, min {min_gain * 100:.0f}%) | held-out pairs "
                  f"{held_before * 100:.2f} → {held_after * 100:.2f} cm | authority "
                  f"{frac * 100:.0f}% of {a_max_m * 100:.0f} cm / {a_max_deg:.1f}° "
                  f"{'SATURATED ' if saturated else ''}→ {verdict}")
            X = Xc if verdict == "APPLY" else np.tile(np.eye(4), (N, 1, 1))
            self._stac_authority_record("pose_graph", frac, saturated)
        if not np.any([not np.allclose(M, np.eye(4), atol=1e-12) for M in X]):
            return
        # apply: compose into the per-frame fields (poses, depth graph, cap and the
        # writers all read _stac_elastic_corr) and move BOTH copies of every frame.
        # The composition happens ONCE per process (a resumed process starts with
        # empty fields and composes once; the npys are stamped separately).
        if not getattr(self, '_stac_pose_graph_composed', False):
            ecorr = getattr(self, '_stac_elastic_corr', None)
            if ecorr is None:
                ecorr = {k: np.tile(np.eye(4), (end - start, 1, 1))
                         for k, (start, end) in enumerate(self.chunk_indices)}
                self._stac_elastic_corr = ecorr
            for k, (start, end) in enumerate(self.chunk_indices):
                for local, g in enumerate(range(start, end)):
                    ecorr[k][local] = X[g] @ ecorr[k][local]
            self._stac_pose_graph_composed = True
        for k, (start, end) in enumerate(self.chunk_indices):
            path = os.path.join(self.result_aligned_dir, f"chunk_{k}.npy")
            data = np.load(path, allow_pickle=True).item()
            if data.get('_stac_pose_graph_applied'):
                print(f"[pose-graph] chunk {k}: already corrected — skipped")
                continue
            wp = np.asarray(data['world_points'])
            lead = wp.ndim == 5
            if lead:
                wp = wp[0]
            moved = 0
            for local, g in enumerate(range(start, end)):
                M = X[g]
                if np.allclose(M, np.eye(4), atol=1e-12):
                    continue
                p = wp[local].reshape(-1, 3).astype(np.float64)
                wp[local] = (p @ M[:3, :3].T + M[:3, 3]).reshape(wp[local].shape).astype(wp.dtype)
                moved += 1
            data['world_points'] = wp[None] if lead else wp
            data['_stac_pose_graph_applied'] = True
            np.save(path, data)
            print(f"[pose-graph] chunk {k}: {moved}/{end - start} frames moved")
        print("[pose-graph] ✅ keyframe corrections applied (rigid per frame: points + camera)")

    def _stac_authority_record(self, stage, fraction, saturated, extra=None):
        """§4.7: every corrective stage declares its authority and reports the
        fraction it used — authority.json accumulates them for the acta."""
        import json as _json
        path = os.path.join(self.output_dir, "authority.json")
        rep = {}
        if os.path.exists(path):
            try:
                rep = _json.load(open(path))
            except ValueError:
                rep = {}
        rep[stage] = {"fraction_used": float(fraction), "saturated": bool(saturated)}
        if extra:
            rep[stage].update(extra)
        with open(path, "w") as f:
            _json.dump(rep, f, indent=1)
        if saturated:
            print(f"[authority] ⚠ stage {stage} SATURATED: {fraction * 100:.0f}% of its declared "
                  f"authority used (limit {self._stac_authority_cfg().get('saturation_warn')})")

    def _stac_vendor_bridge_fit(self, item, pred, meta):
        """The vendor's coarse point-map fit for one bridge (used ONLY as the
        recorded low-confidence fallback when the exact fit starves)."""
        try:
            lay = meta["layout"]
            ka, kb = item[0], item[2]
            da = self._stac_load_chunk(ka)
            db = self._stac_load_chunk(kb)
            wp = np.asarray(pred['world_points']); wp = wp[0] if wp.ndim == 5 else wp
            cf = np.asarray(pred['world_points_conf']).reshape(wp.shape[:3])
            fits = []
            for side, d, bl, cl in (("a", da, lay["bridge_a"], lay["chunk_a_local"]),
                                    ("b", db, lay["bridge_b"], lay["chunk_b_local"])):
                wpc = np.asarray(d['world_points']); wpc = wpc[0] if wpc.ndim == 5 else wpc
                cfc = np.asarray(d['world_points_conf']).reshape(wpc.shape[:3])
                pm_c, c_c = wpc[cl], cfc[cl]
                pm_l, c_l = wp[bl], cf[bl]
                thr = min(np.median(c_c), np.median(c_l)) * 0.1 \
                    if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True) else -1.0
                fits.append(weighted_align_point_maps(pm_c, c_c, pm_l, c_l, None,
                                                      conf_threshold=thr, config=self.config))
            return compute_sim3_ab(fits[0], fits[1])
        except Exception as _e:  # the fallback itself failing is a recorded rejection
            print(f"[loop-verify] vendor fallback fit failed ({_e}) — edge rejected")
            return None
        finally:
            self._stac_drop_chunk_cache()

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

        _stac_loops = (self._stac_loops_cfg() is not None
                       and (self.config['Model'].get('metric_lock') or {}).get('enable'))
        if _stac_loops:
            self._stac_ensemble_uncertainty()      # §4.8 optional third witness (model loaded)
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            half = int(self.config['Model']['loop_chunk_size'] / 2)
            loop_results = process_loop_list(self.chunk_indices, self.loop_list, half_window=half)
            if _stac_loops:
                # STAC: keep the (i, j) candidate attached to its windows and KEEP
                # intra-chunk candidates (i, j in the same chunk) — they are pose
                # edges for the keyframe graph even though they carry no scale row.
                _paired = []
                _seen = set()
                for res, (i, j) in zip(loop_results, self.loop_list):
                    key = (res[0], res[2], res[1], res[3])
                    if key in _seen:
                        continue
                    _seen.add(key)
                    _paired.append(tuple(res) + ((int(i), int(j)),))
                _keep_intra = bool((self._stac_loops_cfg() or {}).get('intra_chunk_loops', True))
                loop_results = [r for r in _paired if _keep_intra or r[0] != r[2]]
            else:
                loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            if _stac_loops:
                # ── STAC F1 ── metric lock (anchors + seams) BEFORE the bridges, so
                # the spatial gate sees a provisional metric chain and no bridge is
                # spent on an implausible candidate (§4.2 step 0). The model stays
                # loaded: the lock is CPU work.
                self._stac_metric_lock()
                planned = self._stac_plan_bridges(loop_results)
                self._stac_infer_bridges(planned)
            else:
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
        if _stac_loops:
            if not self.loop_enable:
                self._stac_metric_lock()
            else:
                # bridges: own anchors (extracted now that the GPU is free) → lock →
                # Sim3 measurement → loop rows close the scale graph (§5)
                self._stac_ensure_bridge_anchors()
                self._stac_lock_bridges()
                meas_sim3 = self._stac_measure_loops(rigid=False)
                self._stac_scale_close(meas_sim3)
        else:
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
                    if not hasattr(self, '_stac_seam_residuals'):
                        self._stac_seam_residuals = {}
                    self._stac_seam_residuals[chunk_idx] = float(_res)   # σ of the seam odometry (§4.3)
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


        if self.loop_enable and _stac_loops:
            # ── STAC F1 ── the scale is closed: exact SE(3) edges + verification
            # (§4.2) on the re-scaled chunks; every edge carries its measured σ.
            meas_rigid = self._stac_measure_loops(rigid=True)
            self._stac_verify_loops(meas_rigid)
        elif self.loop_enable:
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
            # vendor loop path (no Model.loops): coarse point-map edges, no measured σ
            # — the optimizer runs as the vendor ships it. The former "all seams
            # exact → optimizer SKIPPED" rule is gone (claude_stac.txt P1): in the
            # STAC path the decision is σ-based inside _stac_verify_loops.
            self.loop_enable_opt = bool(self.loop_enable and len(self.loop_sim3_list) > 0)
        else:
            self.loop_enable_opt = False
        # STAC F2: in the vggtomega path the per-chunk Sim3 optimizer is REPLACED by
        # the keyframe SE(3) graph (_stac_pose_graph, after the chunks are aligned);
        # the vendor optimizer stays for the other backends.
        _stac_kf_graph = bool(_stac_loops and self._stac_graph_cfg() is not None)
        if self.loop_enable_opt and not _stac_kf_graph:
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
        # STAC: persist the accumulated per-chunk transforms — the post-hoc
        # stages (quality A/B harness, certification) rebuild _stac_aligned_pose
        # from the aligned npys + this file instead of re-running the seams.
        try:
            import json as _json
            with open(os.path.join(self.output_dir, "chunk_sim3.json"), "w") as _f:
                _json.dump({"chunk_indices": [list(ci) for ci in self.chunk_indices],
                            "sim3": [{"s": float(s_), "R": np.asarray(R_).tolist(),
                                      "t": np.asarray(t_).tolist()} for s_, R_, t_ in self.sim3_list]},
                           _f, indent=1)
        except OSError as _e:
            raise RuntimeError(f"could not persist chunk_sim3.json ({_e}) — the post-hoc "
                               f"stages need the accumulated chunk transforms")

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
                     or self.config['Model'].get('blend_copies')
                     or _stac_kf_graph)
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
        if _stac_kf_graph:
            # STAC F2: §4.8 intrinsic uncertainty (two-copy disagreement per shared
            # frame, + the optional ensemble witness) BEFORE the keyframe graph,
            # which consumes it as σ; the graph moves every frame rigidly and its
            # corrections flow into _stac_elastic_corr like the intra-chunk stage.
            self._stac_uncertainty()
            self._stac_ensemble_apply()
            self._stac_pose_graph()
        self._stac_elastic_seams()
        self._stac_intra_chunk()
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