#!/usr/bin/env python3
"""MED-05 R4: admissible basal interval per patient from out/r4_<model>_scan.csv, using
the criteria fixed in r4_safe_basal.py. Writes out/r4_bounds.csv and
out/r4_bounds_posthoc_subsets.csv.

v2 (2026-10-26, after an external review):
  * a grid point is admissible only if ALL criteria hold there jointly
    (worst-case Gmin >= 80 mg/dL, scenario-mean TIR >= 70 %, worst-case Gmax <= 300 mg/dL);
  * grid points with a non-converged orbit in any scenario are treated as not admissible
    and counted in `n_unresolved`;
  * the admissible set must be contiguous on the grid (`contiguous`);
  * bounds are refined by linear interpolation of the binding quantity only toward an
    adjacent non-admissible grid point; widths below the grid step are flagged
    (`below_grid`), and r4_refine.py re-solves those intervals locally.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent / "out"
HYPO, TIR_T, GMAX_T = 80.0, 70.0, 300.0
STEP = 0.02

def summarize(d):
    w = d.groupby("f").agg(Gmin=("Gmin", "min"), Gmax=("Gmax", "max"), TIR=("TIR", "mean"),
                           conv=("converged", "all")).sort_index()
    w["ok"] = (w.Gmin >= HYPO) & (w.TIR >= TIR_T) & (w.Gmax <= GMAX_T) & w.conv
    return w

def _edge(w, i, j):
    """Interpolated crossing between admissible grid point i and non-admissible j."""
    fi, fj = w.index[i], w.index[j]
    cands = []
    for col, thr, ge in (("Gmin", HYPO, True), ("TIR", TIR_T, True), ("Gmax", GMAX_T, False)):
        vi, vj = w[col].iloc[i], w[col].iloc[j]
        bad_j = (vj < thr) if ge else (vj > thr)
        if bad_j and vi != vj:
            cands.append(fi + (vi - thr) / (vi - vj) * (fj - fi))
    if not w.conv.iloc[j] or not cands:
        return fi                                   # unresolved neighbour: keep the grid point
    return min(cands, key=lambda x: abs(x - fi))   # first criterion to fail

def bounds_for(d):
    w = summarize(d)
    idx = np.where(w.ok.values)[0]
    base = dict(n_unresolved=int((~w.conv).sum()),
                Gmin_worst_at_100=float(w.loc[1.0, "Gmin"]), TIR_mean_at_100=float(w.loc[1.0, "TIR"]),
                TIR_mean_at_90=float(w.loc[0.9, "TIR"]))
    if len(idx) == 0:
        return dict(base, f_min=np.nan, f_max=np.nan, width=np.nan, feasible=False,
                    contiguous=True, below_grid=False, at_grid_edge=False)
    lo, hi = idx.min(), idx.max()
    contiguous = bool(len(idx) == hi - lo + 1)
    fmin = w.index[lo] if lo == 0 else _edge(w, lo, lo - 1)
    fmax = w.index[hi] if hi == len(w) - 1 else _edge(w, hi, hi + 1)
    return dict(base, f_min=fmin, f_max=fmax, width=fmax - fmin, feasible=True, contiguous=contiguous,
                below_grid=bool(fmax - fmin < STEP), at_grid_edge=bool(lo == 0 or hi == len(w) - 1))

if __name__ == "__main__":
    res, ph = [], []
    subsets = {"nominal only": lambda s: (s.cho == 1.0) & (s.si == 1.0),
               "carbs only (si=1)": lambda s: s.si == 1.0,
               "sensitivity only (cho=1)": lambda s: s.cho == 1.0}
    for m in ("hovorka", "uva"):
        p = OUT / f"r4_{m}_scan.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        for pid, d in s.groupby("Patient"):
            res.append(dict(model=d.model.iloc[0], Patient=pid, **bounds_for(d)))
        for name, sel in subsets.items():   # POST HOC: subsets of the scenario set
            for pid, d in s[sel(s)].groupby("Patient"):
                ph.append(dict(model=d.model.iloc[0], subset=name, Patient=pid, **bounds_for(d)))
    r = pd.DataFrame(res); r.to_csv(OUT / "r4_bounds.csv", index=False)
    ph = pd.DataFrame(ph); ph.to_csv(OUT / "r4_bounds_posthoc_subsets.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(3).to_string())
    print(ph.groupby(["model", "subset"]).agg(n=("Patient", "size"), feasible=("feasible", "sum"),
          f_min_med=("f_min", "median"), f_max_med=("f_max", "median"), width_med=("width", "median"),
          noncontig=("contiguous", lambda x: int((~x).sum()))).round(3).to_string())
