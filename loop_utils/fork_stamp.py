"""THE FORK STAMP (docs/plan_determinismo.md point 9, audit omega-09).

Every resume artifact of the fork — the chunk and bridge npys, metric_lock.json,
chunk_health.json, scale_graph.json, uncertainty.json, elastic_seams.json, intra_chunk.json,
depth_graph.json, pose_graph.json, the ensemble npys — used to be reloaded whenever its
``chunk_indices`` matched: a verdict made by older code or another config was REPLAYED (a
pre-91cd1f7 pose-graph APPLY that today's judge refuses; every 'Reconstruir' with replace OFF
re-launches the fork over a finished session).

Now one stamp (server ``repro.stamp``) says what a run's products are made of:
- INPUTS: the keyframe images (and, when a loop bridge may add non-keyframe frames, every
  image of the frames directory), frame_quality.json (it picks those extra frames), the Omega
  weights, the DA3 anchors of the keyframes, the pinned sky segmenter, loop_semantics.json;
- CODE: the fork's own modules, the VGGT-Omega package it wraps, the server modules it imports
  by name (and their packages' __init__);
- CONFIG: the whole fork config, the frame list, the loop candidates and the numerics
  environment (card model, torch / CUDA / cuDNN, autocast dtype, torch numerics settings).

It is computed once at the start of :meth:`VGGT_Long.run`, held against ``fork_stamp.json``
and — ANY difference — every product of the old stamp is DELETED and the run recomputes from
the start (:func:`reconcile`): a product is resumed only by the code, config and inputs that
made it. Products cannot be recomputed one by one: the metric lock and the corrections act on
the chunk npys IN PLACE. The digest also travels inside every artifact
(``fork_stamp`` / ``_stac_fork_stamp``); an artifact of another digest reaching a stage after
the reconcile is a broken invariant and stops the run.

DA3 anchors the run extracts itself (bridge windows, ``_stac_ensure_bridge_anchors``) are its
PRODUCTS, not its inputs: recorded in ``fork_stamp.json`` (``run_products.anchors``), left out
of the stamp of a resume of the same run, and deleted with the other products when the stamp
changes — so a recomputed run starts from the inputs the map worker gave the first one.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from loop_utils.stac_repro import FORK_DIR, SERVER_DIR, repro

FORK_STAMP_NAME = "fork_stamp.json"
# what a new stamp leaves standing in the run directory: files that carry (and are checked
# against) their OWN stamp before the run stamp is computed — the SALAD candidates and their
# calibration (point 8), the content-keyed sky-mask cache (point 20) — the launch's inputs
# (the frame list this launch wrote, the config the map worker wrote, a server-written
# loop_semantics.json) and the stamp file itself. The same set map_worker keeps across a new
# chunk plan (_PLAN_INDEPENDENT_MAPLONG), plus the post-hoc candidates, loop_semantics.json and
# fork_stamp.json.
KEEP_ON_NEW_STAMP = frozenset({"loop_closures.txt", "loop_closures_posthoc.txt",
                               "salad_calibration.json", "sky_masks", "frame_list.json",
                               "vggt_omega_config.yaml", "loop_semantics.json",
                               FORK_STAMP_NAME})
# the run directories the fork creates at start (recreated after a wipe)
RUN_SUBDIRS = ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd")
# the server modules the fork imports by name (VGGT_Long._stac_server_module)
SERVER_MODULES = ("reconstruction/__init__.py", "reconstruction/loops/__init__.py",
                  "reconstruction/loops/spatial_gate.py", "reconstruction/da3_anchor.py",
                  "reconstruction/vio_scale.py", "repro.py")
_IMAGE_EXT = (".jpg", ".png")


def fork_code_files(fork_dir: str = FORK_DIR, server_dir: str = SERVER_DIR) -> List[Path]:
    """The code whose bits make the fork's products: every module of the fork outside its
    vendored third-party trees and tests (vggt_long.py, loop_utils/, base_models/
    base_model.py + vggtomega_adapter.py, LoopModels/ — the SALAD detector), the VGGT-Omega
    package it wraps, and the server modules it imports by name."""
    fork = Path(fork_dir)
    files = [fork / "vggt_long.py"]
    files += sorted((fork / "loop_utils").glob("*.py"))
    files += [fork / "base_models" / "base_model.py",
              fork / "base_models" / "vggtomega_adapter.py"]
    files += sorted(p for p in (fork / "LoopModels").rglob("*.py") if "__pycache__" not in p.parts)
    omega = fork.parent / "vggt-omega" / "vggt_omega"
    files += sorted(p for p in omega.rglob("*.py") if "__pycache__" not in p.parts)
    files += [Path(server_dir) / m for m in SERVER_MODULES]
    return [p for p in files if p.is_file()]


def _real_frame_number(path: str) -> int:
    from loop_utils.metric_lock import real_frame_number
    return int(real_frame_number(path))


def run_stamp(*, img_list: Sequence[str], img_dir: str, config: dict,
              loop_cands: Optional[Sequence[dict]] = None,
              output_dir: Optional[str] = None, skyseg: Optional[str] = None,
              exclude_anchors: Iterable[str] = (), environment: Optional[dict] = None,
              code_files: Optional[Sequence[Path]] = None) -> Dict[str, object]:
    """repro.stamp of everything a run's products are made of (see the module docstring).
    ``skyseg``: the pinned model's path when the run masks the sky; ``exclude_anchors``: anchor
    file names this run extracted itself (its products — :func:`run_products`);
    ``environment``: what the numerics depend on beyond code / config / inputs (the card model,
    torch / CUDA / cuDNN, the autocast dtype, the torch numerics settings — plan point 12)."""
    R = repro()
    model = config.get("Model") or {}
    inputs: Dict[str, str] = {f"frames/{os.path.basename(p)}": p for p in img_list}
    loops = model.get("loops") or {}
    if int(loops.get("bridge_extra_frames", 0) or 0) > 0 and bool(model.get("loop_enable")):
        # a bridge window may add NON-keyframe frames (_stac_extra_frames): any image of the
        # directory can enter a bridge
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(_IMAGE_EXT):
                inputs[f"frames/{fn}"] = os.path.join(img_dir, fn)
        fq = os.path.join(img_dir, "frame_quality.json")
        if os.path.isfile(fq):
            inputs["frames/frame_quality.json"] = fq
    wcfg = config.get("Weights") or {}
    name = wcfg.get("model")
    wpath = wcfg.get(name) if name else None
    if name == "VGGTOmega" and not (wpath and os.path.isfile(str(wpath))):
        raise RuntimeError(f"Weights.{name} {wpath!r} is not a local file — the run cannot be "
                           f"stamped (plan point 9)")
    if wpath and os.path.isfile(str(wpath)):
        inputs[f"weights/{name}"] = str(wpath)
    # (a non-production backend whose weights are a hub reference is stamped by that
    # reference, inside the fork config — its bytes cannot be read here)
    anchor_dir = (model.get("metric_lock") or {}).get("anchor_dir")
    excluded = set(exclude_anchors or ())
    if anchor_dir:
        for p in img_list:
            fn = f"frame_{_real_frame_number(p)}.npz"
            a = os.path.join(anchor_dir, fn)
            if fn not in excluded and os.path.isfile(a):
                inputs[f"anchors/{fn}"] = a
    if skyseg:
        inputs["skyseg.onnx"] = skyseg
    if output_dir:
        sem = os.path.join(output_dir, "loop_semantics.json")
        if os.path.isfile(sem):
            inputs["loop_semantics.json"] = sem
    cands = [{"i": int(c["i"]), "j": int(c["j"]),
              "sim": (None if c.get("sim") is None else float(c["sim"])),
              "source": str(c.get("source", "salad"))} for c in (loop_cands or [])]
    cfg = {"fork_config": config, "frame_list": [os.path.basename(p) for p in img_list],
           "loop_candidates": cands}
    if environment is not None:
        cfg["environment"] = environment
    return R.stamp(inputs=inputs,
                   code=(fork_code_files() if code_files is None else list(code_files)),
                   config=cfg)


def read_saved(output_dir: str) -> Optional[dict]:
    """fork_stamp.json (None: absent or unreadable)."""
    path = os.path.join(output_dir, FORK_STAMP_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def run_products(saved: Optional[dict]) -> Dict[str, List[str]]:
    """What a saved stamp's run produced outside its directory (``anchors``: DA3 anchor file
    names it extracted itself)."""
    rp = (saved or {}).get("run_products") or {}
    return {"anchors": sorted(str(a) for a in (rp.get("anchors") or []))}


def _write(output_dir: str, doc: dict) -> None:
    path = os.path.join(output_dir, FORK_STAMP_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def record_run_products(output_dir: str, *, anchors: Sequence[str]) -> None:
    """Add DA3 anchor files the run extracted itself to fork_stamp.json (``run_products``) —
    the stamp of a resume of this run leaves them out, a new stamp deletes them."""
    doc = read_saved(output_dir)
    if doc is None:
        raise RuntimeError(f"{os.path.join(output_dir, FORK_STAMP_NAME)} is missing — the run "
                           f"stamp is written before any product (plan point 9)")
    rp = dict(doc.get("run_products") or {})
    rp["anchors"] = sorted(set(rp.get("anchors") or []) | {os.path.basename(a) for a in anchors})
    doc["run_products"] = rp
    _write(output_dir, doc)


def reconcile(output_dir: str, now: dict, *, anchor_dir: Optional[str] = None,
              saved: Optional[dict] = None,
              log: Callable[[str], object] = print) -> Dict[str, object]:
    """Hold this run's stamp ``now`` against fork_stamp.json. The same stamp: the products on
    disk are this run's — resumed. Any difference (no stamp, another code / config / input):
    every entry of ``output_dir`` outside :data:`KEEP_ON_NEW_STAMP` is deleted, with the
    anchors the old run extracted itself (from ``anchor_dir``), and the run starts over; the
    new stamp is written. Returns {'resumed', 'diffs', 'deleted'}."""
    R = repro()
    if saved is None:
        saved = read_saved(output_dir)
    diffs = R.check_stamp(saved, now)
    deleted: List[str] = []
    if not diffs:
        return {"resumed": True, "diffs": [], "deleted": []}
    for name in sorted(os.listdir(output_dir)):
        if name in KEEP_ON_NEW_STAMP:
            continue
        p = os.path.join(output_dir, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
        deleted.append(name)
    if anchor_dir:
        for fn in run_products(saved)["anchors"]:
            a = os.path.join(anchor_dir, fn)
            if os.path.isfile(a):
                os.remove(a)
                deleted.append(f"anchors/{fn}")
    for d in RUN_SUBDIRS:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    _write(output_dir, dict(now))
    if deleted:
        log("[STAC stamp] the products on disk were made under another stamp — "
            + "; ".join(diffs[:8]) + (f"; … {len(diffs) - 8} more" if len(diffs) > 8 else "")
            + f" — {len(deleted)} product(s) deleted, the run recomputes from the start "
              f"(plan point 9)")
    return {"resumed": False, "diffs": diffs, "deleted": deleted}


def check_artifact(doc, now_digest: Optional[str], what: str, error=RuntimeError) -> None:
    """An artifact (a dict carrying 'fork_stamp' / '_stac_fork_stamp', or the digest itself)
    of ANOTHER run stamp raises ``error``. ``now_digest`` None (a stage driven alone, e.g. by
    a unit test) has nothing to compare and passes."""
    if now_digest is None:
        return
    saved = doc.get("fork_stamp", doc.get("_stac_fork_stamp")) if isinstance(doc, dict) else doc
    if saved is not None and str(saved) == str(now_digest):
        return
    why = ("it carries no fork stamp" if saved is None else
           f"its fork stamp {str(saved)[:12]}… is not this run's {str(now_digest)[:12]}…")
    raise error(f"{what}: {why} — produced by another fork code / config / input, after the "
                f"run stamp was reconciled (a broken invariant: plan point 9 never resumes it)")
