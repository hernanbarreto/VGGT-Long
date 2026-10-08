"""The STAC server's reproducibility primitives, reached from the fork (docs/plan_determinismo.md).

The fork is vendored at <repo>/vendor/VGGT-Long; the server's ``repro.py`` (stamps, the
deterministic torch settings, the environment record, lossless pose text) lives at
<repo>/server. ONE implementation for the whole pipeline — the fork imports it, it never
carries a copy. A missing ``repro.py`` FAILS: a fork run that cannot stamp, record or write
exactly is not a run of this pipeline."""

from __future__ import annotations

import os
import sys

FORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(os.path.dirname(os.path.dirname(FORK_DIR)), "server")


def repro():
    """The server's ``repro`` module (its directory put on sys.path when it is not there)."""
    if not os.path.isfile(os.path.join(SERVER_DIR, "repro.py")):
        raise RuntimeError(f"{SERVER_DIR}/repro.py not found — the fork needs the STAC server's "
                           f"reproducibility primitives (docs/plan_determinismo.md)")
    if SERVER_DIR not in sys.path:
        sys.path.append(SERVER_DIR)
    import repro as _repro
    return _repro
