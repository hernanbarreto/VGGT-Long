"""pytest bootstrap for the STAC fork tests: the fork root (loop_utils), the
STAC server (reconstruction.loops) and its tests dir (synthetic generator)
importable."""
import sys
from pathlib import Path

_FORK = Path(__file__).resolve().parents[1]
_SERVER = _FORK.parents[1] / "server"
for p in (_FORK, _SERVER, _SERVER / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
