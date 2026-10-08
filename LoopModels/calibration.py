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
    (``cos_bar``); every other non-local pair is not — except the pairs within
    ``error_factor`` x the bar's MEASURED error (``dist_bar_err_m`` / ``cos_bar_err``,
    intake.walk.revisit_reference) of either bar, which are in neither class (plan point
    71). The threshold maximises Youden's J = TPR - FPR over the pairs' own
    similarities — no parameter. Returns (threshold or None, report)."""
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
    FF = F @ F.T
    dist_bar, cos_bar = float(reference["dist_bar_m"]), float(reference["cos_bar"])
    # DECIDIDO (docs/plan_determinismo.md point 71, 2026-10-07): each bar carries the error
    # intake.walk.revisit_reference MEASURED for it (a fixed-key bootstrap of the median over
    # the I3 windows), and a pair within error_factor x that error of EITHER bar is in neither
    # class — it is undecidable at the bar's own resolution and must not flip the Youden
    # argmax from one run to the next (pccr 2408 logged dist_bar 1.41 m and 1.44 m).
    # error_factor is THE USER's factor of the rule (point 1: improvement >= 2 x the error),
    # carried by the reference from correction_graph.graph.improvement_error_factor.
    try:
        dist_err = float(reference["dist_bar_err_m"])
        cos_err = float(reference["cos_bar_err"])
        factor = float(reference["error_factor"])
    except KeyError as e:
        raise KeyError(f"the revisit reference carries no measured bar error / error factor "
                       f"({e}) — written by intake.walk.revisit_reference version >= 3; "
                       f"re-run I3 (the walk) on this session") from e
    undecided = iu & ((np.abs(D - dist_bar) < factor * dist_err)
                      | (np.abs(FF - cos_bar) < factor * cos_err))
    pos = iu & ~undecided & (D < dist_bar) & (FF > cos_bar)
    neg = iu & ~undecided & ~pos
    sp, sn = S[pos], S[neg]
    n_und = int(undecided.sum())
    if sp.size == 0 or sn.size == 0:
        return None, {"reason": f"{int(sp.size)} revisit / {int(sn.size)} other pair(s) — "
                                f"nothing to separate", "n_revisit_pairs": int(sp.size),
                      "n_undecided_pairs": n_und, "dist_bar_err_m": dist_err,
                      "cos_bar_err": cos_err, "error_factor": factor}
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
                            "dist_bar_m": dist_bar, "cos_bar": cos_bar,
                            # the bars' measured errors and the pairs left out of both classes
                            # for lying within error_factor x the error of either bar
                            "dist_bar_err_m": dist_err, "cos_bar_err": cos_err,
                            "error_factor": factor, "n_undecided_pairs": n_und}
