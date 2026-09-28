"""SALAD's appearance bar calibrated on the session's own geometric revisits
(STAC 2026-09-28) — numpy only, so it can be tested without the model."""

import numpy as np


def calibrate_threshold(sims, local, frame_of_index, reference):
    """The appearance bar that best separates GEOMETRIC revisits from everything
    else, measured on the session (STAC, 2026-09-28).

    ``sims``: the (n, n) cosine similarity of the detector's descriptors;
    ``local``: its odometry band mask; ``frame_of_index``: the frame number of each
    descriptor row; ``reference``: intake.walk.revisit_reference's document (the
    DA3-window trajectory). A non-local pair is a revisit when its cameras stand
    closer than ``dist_bar_m`` and look less than half the field of view apart
    (``cos_bar``); every other non-local pair is not. The threshold maximises
    Youden's J = TPR - FPR over the pairs' own similarities — no parameter.
    Returns (threshold or None, report)."""
    idx = {int(f): k for k, f in enumerate(reference["frames"])}
    rows = [i for i, f in enumerate(frame_of_index) if int(f) in idx]
    if len(rows) < 2:
        return None, {"reason": "no descriptor row maps onto the reference trajectory"}
    ref_k = np.array([idx[int(frame_of_index[i])] for i in rows])
    C = np.asarray(reference["centres"], np.float64)[ref_k]
    F = np.asarray(reference["forward"], np.float64)[ref_k]
    S = np.asarray(sims, np.float64)[np.ix_(rows, rows)]
    L = np.asarray(local, bool)[np.ix_(rows, rows)]
    iu = np.triu(~L, 1)
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    pos = iu & (D < float(reference["dist_bar_m"])) & ((F @ F.T) > float(reference["cos_bar"]))
    neg = iu & ~pos
    sp, sn = S[pos], S[neg]
    if sp.size == 0 or sn.size == 0:
        return None, {"reason": f"{int(sp.size)} revisit / {int(sn.size)} other pair(s) — "
                                f"nothing to separate", "n_revisit_pairs": int(sp.size)}
    cand = np.unique(np.concatenate([sp, sn]))
    tpr = 1.0 - np.searchsorted(np.sort(sp), cand, side="left") / sp.size
    fpr = 1.0 - np.searchsorted(np.sort(sn), cand, side="left") / sn.size
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(cand[k]), {"threshold": float(cand[k]), "youden_j": float(j[k]),
                            "tpr": float(tpr[k]), "fpr": float(fpr[k]),
                            "n_revisit_pairs": int(sp.size), "n_other_pairs": int(sn.size),
                            "revisit_similarity_median": float(np.median(sp)),
                            "other_similarity_median": float(np.median(sn)),
                            "dist_bar_m": float(reference["dist_bar_m"]),
                            "cos_bar": float(reference["cos_bar"])}
