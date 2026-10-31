"""The VGGT-SLAM 2.0 sections of the fork config (claude_stac_slam2.txt, phase 0).

Two optional sections reach the fork from server/config.yaml through the map worker:
``Model.seams`` (idea 1, phase 2: one calibration for every Ω inference — the seams and the
loops measure on the same K) and ``Model.loops.attention_gate`` (idea 2, phase 3: the
training-free loop verifier that reads the backbone's global attention over a candidate pair).

THE RULE (prompt rule 5): ABSENT = OFF = f576cbf bit for bit. PRESENT = every subkey mandatory
(``cfg_req``, no default in code), the enumerations checked by name, before any GPU work. The
map worker writes a section only when the server config turns it on, so the reference's fork
config never changes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loop_utils.loop_bridges import cfg_req

SEAMS_PATH = "seams"
GATE_PATH = "loops.attention_gate"
CALIBRATION_PATH = "loops.attention_gate.calibration"

FOCAL_MODELS = ("none", "common_K")
COMMON_K_SOURCES = ("session_median", "intake_probe")
GATE_MODES = ("off", "gate")
CALIBRATION_SELECTS = ("max_auc_lower_ci",)


def _choice(section: dict, key: str, path: str, allowed) -> str:
    v = cfg_req(section, key, path)
    if not isinstance(v, str) or v not in allowed:
        raise RuntimeError(f"config key Model.{path}.{key} must be one of {'|'.join(allowed)}, "
                           f"got {v!r}")
    return v


def _number(section: dict, key: str, path: str, lo=None, hi=None, integer=False,
            lo_excl=False):
    v = cfg_req(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RuntimeError(f"config key Model.{path}.{key} must be a number, got {v!r}")
    if integer and int(v) != v:
        raise RuntimeError(f"config key Model.{path}.{key} must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise RuntimeError(f"config key Model.{path}.{key}={v} is below the minimum {lo}")
    if hi is not None and v > hi:
        raise RuntimeError(f"config key Model.{path}.{key}={v} is above the maximum {hi}")
    return int(v) if integer else float(v)


def seams_cfg(config: dict) -> Optional[Dict[str, Any]]:
    """``Model.seams`` validated — None when the section is absent (off)."""
    model = (config or {}).get("Model") or {}
    se = model.get("seams")
    if se is None:
        return None
    if not isinstance(se, dict):
        raise RuntimeError(f"config section Model.{SEAMS_PATH} must be a mapping (or absent = off)")
    return {"focal_model": _choice(se, "focal_model", SEAMS_PATH, FOCAL_MODELS),
            "common_K_source": _choice(se, "common_K_source", SEAMS_PATH, COMMON_K_SOURCES)}


def attention_gate_cfg(config: dict) -> Optional[Dict[str, Any]]:
    """``Model.loops.attention_gate`` validated (the calibration block included) — None when the
    section is absent (off)."""
    model = (config or {}).get("Model") or {}
    loops = model.get("loops") or {}
    ag = loops.get("attention_gate")
    if ag is None:
        return None
    if not isinstance(ag, dict):
        raise RuntimeError(f"config section Model.{GATE_PATH} must be a mapping (or absent = off)")
    sst = cfg_req(ag, "salad_similarity_threshold", GATE_PATH)
    if sst is not None:
        sst = _number(ag, "salad_similarity_threshold", GATE_PATH, lo=-1.0, hi=1.0)
    ca = cfg_req(ag, "calibration", GATE_PATH)
    if not isinstance(ca, dict):
        raise RuntimeError(f"config section Model.{CALIBRATION_PATH} must be a mapping")
    grid = cfg_req(ca, "salad_grid", CALIBRATION_PATH)
    if (not isinstance(grid, (list, tuple)) or not grid
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and -1.0 <= x <= 1.0 for x in grid)):
        raise RuntimeError(f"config key Model.{CALIBRATION_PATH}.salad_grid must be a non-empty "
                           f"list of similarities in [-1, 1]")
    calibration = {
        "select": _choice(ca, "select", CALIBRATION_PATH, CALIBRATION_SELECTS),
        "target_precision": _number(ca, "target_precision", CALIBRATION_PATH, lo=0.0, hi=1.0),
        "min_labels": _number(ca, "min_labels", CALIBRATION_PATH, lo=1, integer=True),
        "n_negatives": _number(ca, "n_negatives", CALIBRATION_PATH, lo=0, integer=True),
        "n_boot": _number(ca, "n_boot", CALIBRATION_PATH, lo=1, integer=True),
        "confidence": _number(ca, "confidence", CALIBRATION_PATH, lo=0.5, hi=1.0, lo_excl=True),
        "salad_grid": [float(x) for x in grid],
    }
    return {"mode": _choice(ag, "mode", GATE_PATH, GATE_MODES),
            "layer": _number(ag, "layer", GATE_PATH, lo=0, integer=True),
            "min_score": _number(ag, "min_score", GATE_PATH, lo=0.0, hi=1.0),
            "top_fraction": _number(ag, "top_fraction", GATE_PATH, lo=0.0, hi=1.0, lo_excl=True),
            "salad_similarity_threshold": sst,
            "calibration": calibration}


def validate(config: dict) -> Dict[str, Optional[Dict[str, Any]]]:
    """Both sections at once — called before any GPU work so a malformed section fails by name
    first. {'seams': ..., 'attention_gate': ...}, each None when absent."""
    return {"seams": seams_cfg(config), "attention_gate": attention_gate_cfg(config)}
