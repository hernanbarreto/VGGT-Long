"""claude_stac_slam2.txt, phase 0 (fork side): the optional sections ``Model.seams`` and
``Model.loops.attention_gate`` — ABSENT = off = f576cbf bit for bit; PRESENT = every subkey
mandatory, refused by name before any GPU work. No GPU, no model."""
import copy
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
sys.path.insert(0, str(_FORK / "tests"))

from slam2_scenario import GATE_OFF, SEAMS_OFF, compare_products          # noqa: E402

REFERENCE_COMMIT = "f576cbf"       # the fork pin of stac-build main when the branch was opened


def _cfg(seams=None, gate=None):
    cfg = {"Model": {"loops": {"anchors_per_bridge": 3}}}
    if seams is not None:
        cfg["Model"]["seams"] = copy.deepcopy(seams)
    if gate is not None:
        cfg["Model"]["loops"]["attention_gate"] = copy.deepcopy(gate)
    return cfg


def test_absent_sections_are_off():
    from loop_utils.slam2_config import attention_gate_cfg, seams_cfg, validate
    assert seams_cfg(_cfg()) is None and attention_gate_cfg(_cfg()) is None
    assert validate(_cfg()) == {"seams": None, "attention_gate": None}
    assert validate({"Model": {}}) == {"seams": None, "attention_gate": None}


def test_present_sections_are_read_whole():
    from loop_utils.slam2_config import validate
    seams_on = dict(SEAMS_OFF, focal_model="common_K")
    gate_on = dict(GATE_OFF, mode="gate", salad_similarity_threshold=0.60)
    v = validate(_cfg(seams=seams_on, gate=gate_on))
    assert v["seams"] == seams_on
    assert v["attention_gate"]["mode"] == "gate" and v["attention_gate"]["layer"] == 4
    assert v["attention_gate"]["salad_similarity_threshold"] == 0.60
    assert v["attention_gate"]["calibration"]["salad_grid"] == [0.55, 0.60, 0.65, 0.70]
    off = validate(_cfg(seams=SEAMS_OFF, gate=GATE_OFF))
    assert off["seams"]["focal_model"] == "none" and off["attention_gate"]["mode"] == "off"


@pytest.mark.parametrize("seams, gate, named", [
    ({"focal_model": "common_K"}, None, "Model.seams.common_K_source"),
    ({"common_K_source": "session_median"}, None, "Model.seams.focal_model"),
    ({"focal_model": "affine", "common_K_source": "session_median"}, None, "Model.seams.focal_model"),
    ({"focal_model": "common_K", "common_K_source": "stray"}, None, "Model.seams.common_K_source"),
    (None, {k: v for k, v in GATE_OFF.items() if k != "min_score"}, "Model.loops.attention_gate.min_score"),
    (None, {k: v for k, v in GATE_OFF.items() if k != "layer"}, "Model.loops.attention_gate.layer"),
    (None, {k: v for k, v in GATE_OFF.items() if k != "salad_similarity_threshold"},
     "Model.loops.attention_gate.salad_similarity_threshold"),
    (None, {k: v for k, v in GATE_OFF.items() if k != "calibration"}, "Model.loops.attention_gate.calibration"),
    (None, dict(GATE_OFF, calibration={k: v for k, v in GATE_OFF["calibration"].items() if k != "n_boot"}),
     "Model.loops.attention_gate.calibration.n_boot"),
    (None, dict(GATE_OFF, mode="verify"), "Model.loops.attention_gate.mode"),
    (None, dict(GATE_OFF, top_fraction=0.0), "Model.loops.attention_gate.top_fraction"),
    (None, dict(GATE_OFF, calibration=dict(GATE_OFF["calibration"], select="max_auc")),
     "Model.loops.attention_gate.calibration.select"),
])
def test_a_present_section_with_a_missing_or_bad_subkey_fails_by_name(seams, gate, named):
    from loop_utils.slam2_config import validate
    with pytest.raises(RuntimeError) as e:
        validate(_cfg(seams=seams, gate=gate))
    assert named in str(e.value), str(e.value)


def test_the_runner_validates_the_sections_before_any_inference():
    """A malformed section fails by name at the top of process_long_sequence — before the first
    Omega pass — and the runner caches the validated sections."""
    import vggt_long as vl
    src = (_FORK / "vggt_long.py").read_text()
    body = src[src.index("    def process_long_sequence(self):"):]
    first_stmt = body.splitlines()[1].strip()
    assert first_stmt.startswith("self._stac_slam2_cfg()"), first_stmt
    r = object.__new__(vl.VGGT_Long)
    r.config = _cfg(seams={"focal_model": "common_K"})
    with pytest.raises(RuntimeError, match="Model.seams.common_K_source"):
        r._stac_slam2_cfg()
    r2 = object.__new__(vl.VGGT_Long)
    r2.config = _cfg()
    assert r2._stac_slam2_cfg() == {"seams": None, "attention_gate": None}
    assert r2._stac_slam2_cfg() is r2._stac_slam2_cfg()


# ── bit for bit against f576cbf ─────────────────────────────────────────────────────────────

def _env():
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    env.pop("PYTHONPATH", None)
    return env


def _reference_tree(tmp_path: Path) -> Path:
    """The fork at REFERENCE_COMMIT laid out as <tmp>/vendor/VGGT-Long beside <tmp>/server (the
    same relative layout the fork's imports resolve by: loop_utils.stac_repro, tests/conftest)."""
    try:
        head = subprocess.run(["git", "-C", str(_FORK), "rev-parse", "--verify", f"{REFERENCE_COMMIT}^{{commit}}"],
                              capture_output=True, text=True)
    except OSError as e:
        pytest.skip(f"git is not available ({e}) — the reference tree cannot be extracted")
    if head.returncode != 0:
        pytest.skip(f"commit {REFERENCE_COMMIT} is not in this fork clone — the reference tree "
                    f"cannot be extracted ({head.stderr.strip()})")
    root = tmp_path / "ref"
    fork = root / "vendor" / "VGGT-Long"
    fork.mkdir(parents=True)
    tar = tmp_path / "ref.tar"
    with open(tar, "wb") as f:
        subprocess.run(["git", "-C", str(_FORK), "archive", REFERENCE_COMMIT], stdout=f, check=True)
    with tarfile.open(tar) as t:
        t.extractall(fork)
    os.symlink(_SERVER, root / "server")
    # the scenario file travels with the test (f576cbf predates it); it imports only what both
    # trees have
    shutil.copy(_FORK / "tests" / "slam2_scenario.py", fork / "tests" / "slam2_scenario.py")
    return fork


def _run(fork: Path, out: Path, *flags) -> Path:
    r = subprocess.run([sys.executable, str(fork / "tests" / "slam2_scenario.py"), str(out), *flags],
                       cwd=str(fork), env=_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"scenario failed in {fork}:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
    return out / "maplong_run"


@pytest.fixture(scope="module")
def reference_products(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("slam2_ref")
    fork = _reference_tree(tmp)
    return _run(fork, tmp / "out")


def test_products_with_the_sections_absent_are_f576cbf_bit_for_bit(tmp_path, reference_products):
    now = _run(_FORK, tmp_path / "now")
    assert {p.name for p in now.glob("*.json")} >= {"chunk_health.json", "loop_edges.json",
                                                     "metric_lock.json", "scale_graph.json"}
    diffs = compare_products(reference_products, now)
    assert diffs == [], "\n".join(diffs)


def test_products_with_the_sections_present_but_off_are_f576cbf_bit_for_bit(tmp_path, reference_products):
    now = _run(_FORK, tmp_path / "now_off", "--seams-none", "--gate-off")
    diffs = compare_products(reference_products, now)
    assert diffs == [], "\n".join(diffs)


def test_two_runs_of_this_tree_agree_bit_for_bit(tmp_path):
    a = _run(_FORK, tmp_path / "a")
    b = _run(_FORK, tmp_path / "b")
    assert compare_products(a, b) == []
