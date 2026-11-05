"""STAC 2026-11-05 (USER: "una pasada de omega ahí ves cómo se distribuye todo … es recto, la covisibilidad es tanto, lo
puedo procesar en una pasada, o es un tramo con muchas vueltas"): ONE VGGT-Ω inference over a given list of frames.

  - the STRUCTURE PROBE: every keyframe of the walk at the resolution where they all fit the card — its cameras and
    depths say who sees whom across the whole walk (server/reconstruction/walk_structure.py);
  - a REVISITED ZONE: all its pieces of the walk inferred together, so the revisit is reconstructed jointly.

Saves per frame: camera-to-world (OpenCV), intrinsics on the inference grid, depth and confidence (the session's sky
mask applied — the same cached masks and pinned model as the chunks); with --salad also the SALAD similarity of every
pair (the loop detector's own model and preprocessing) and the appearance bar the loop detector would apply to it: the
bar calibrated on the session's geometric revisits (LoopModels.calibration, the run's Loop.SALAD.revisit_reference and
non-local band), else the configured similarity_threshold — said which. Deterministic: repro's STRICT torch settings,
one dtype. The config is read as the fork reads it (loop_utils.config_utils.load_config).

  python stac_frames_pass.py --config <run vggt_omega_config.yaml> --frames <list.txt> --resolution R --out <npz>
         [--salad] [--sky-cache <maplong_run/sky_masks>]"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_utils.stac_repro import repro as _stac_repro          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", required=True, help="a text file: one image path per line, in order")
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--salad", action="store_true")
    ap.add_argument("--sky-cache", default=None)
    a = ap.parse_args(argv)
    _stac_repro().enable_deterministic_torch(42)
    from loop_utils.config_utils import load_config
    cfg = load_config(a.config)
    cfg.setdefault("Model", {})["omega_resolution"] = int(a.resolution)
    paths = [l.strip() for l in open(a.frames) if l.strip()]
    if not paths:
        raise SystemExit("no frame to infer")
    from base_models.vggtomega_adapter import VGGTOmegaAdapter
    model = VGGTOmegaAdapter(cfg)
    model.load()
    pred = model.infer_chunk(paths)
    c2w = pred["extrinsic"][0].float().cpu().numpy()                  # (S, 4, 4)
    K = pred["intrinsic"][0].float().cpu().numpy()                    # (S, 3, 3) on the inference grid
    depth = pred["depth"][0].float().cpu().numpy()                    # (S, H, W)
    conf = (pred["depth_conf"][0] if pred.get("depth_conf") is not None
            else torch.ones_like(pred["depth"][0])).float().cpu().numpy()
    del pred, model
    torch.cuda.empty_cache()
    sky = None
    if a.sky_cache and bool(cfg.get("Model", {}).get("mask_sky", True)):
        from loop_utils.sky_mask import apply_sky_masks, skyseg_path
        from loop_utils.visual_util import segment_sky
        sky = apply_sky_masks(conf, paths, a.sky_cache, model_path=skyseg_path(), segment=segment_sky, log=print)
    out = {"c2w": c2w.astype(np.float32), "K": K.astype(np.float32), "depth": depth.astype(np.float32),
           "conf": conf.astype(np.float32), "paths": np.array(paths), "resolution": np.int32(a.resolution)}
    if a.salad:
        from LoopModels.LoopModel import LoopDetector
        det = LoopDetector(image_dir=os.path.dirname(paths[0]), output=os.devnull, config=cfg)
        det.image_paths = list(paths)
        D = det.extract_descriptors()
        D = D.float().cpu().numpy() if torch.is_tensor(D) else np.asarray(D, np.float32)
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        S = (D @ D.T).astype(np.float32)
        out["salad"] = S
        del det
        torch.cuda.empty_cache()
        # the bar LoopModel.find_loop_closures applies: calibrated on the geometric revisits over its own non-local
        # band (Loop.SALAD.min_gap / min_gap_frac, one bridge window), else the configured threshold
        sal = cfg["Loop"]["SALAD"]
        bar, src = float(sal["similarity_threshold"]), "configured"
        ref_path = sal.get("revisit_reference")
        if ref_path and os.path.exists(ref_path):
            from LoopModels.calibration import calibrate_threshold
            n = len(paths)
            bridge = int((cfg.get("Model") or {}).get("loop_chunk_size", 0) or 0)
            min_gap = max(int(sal.get("min_gap", 0) or 0), int(math.ceil(float(sal.get("min_gap_frac", 0) or 0) * n)),
                          bridge)
            idx = np.arange(n)
            local = np.abs(idx[:, None] - idx[None, :]) < min_gap
            frame_of = [int("".join(ch for ch in os.path.splitext(os.path.basename(q))[0] if ch.isdigit()))
                        for q in paths]
            with open(ref_path) as f:
                ref = json.load(f)
            cal, crep = calibrate_threshold(S.astype(np.float64), local, frame_of, ref)
            if cal is not None:
                bar, src = float(cal), f"calibrated on {crep['n_revisit_pairs']} geometric revisit pair(s)"
            else:
                src = f"configured (not calibrated: {crep.get('reason')})"
        out["salad_bar"] = np.float64(bar)
        out["salad_bar_source"] = np.array(src)
    tmp = a.out + ".part.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.out)
    print(f"[frames-pass] {len(paths)} frame(s) at {a.resolution} (grid {depth.shape[2]}x{depth.shape[1]}) → {a.out}"
          + (f"; sky masked ({sky['n_reused']} reused, {sky['n_computed']} computed)" if sky else "")
          + (f"; SALAD similarity of every pair, bar {float(out['salad_bar']):.3f} ({out['salad_bar_source']})"
             if a.salad else ""))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    main()
