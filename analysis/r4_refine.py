#!/usr/bin/env python3
"""MED-05 R4: local re-solution of admissible-interval bounds (pre-specified criteria, all nine
scenarios) by bisection on f, replacing grid interpolation. Applies to every feasible patient in
out/r4_bounds.csv whose bound is not at the grid edge. Tolerance 0.001 in f.
Writes out/r4_bounds_refined.csv."""
import sys, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import r4_safe_basal as R
from r4_bounds import HYPO, TIR_T, GMAX_T, STEP

TOL = 0.001

def make(model, pid):
    if model == "Hovorka":
        row = [r for r in R.R3.H.PATIENTS_ADULT if r[0] == pid][0]
        return (lambda f, c, s: R.HovS(row, f, c, s)), R.R3.HovorkaDay(row, 1.0, "ramadan").y_ss()
    return (lambda f, c, s: R.UvaS(pid, f, c, s)), R.R3.UVADay(pid, 1.0, "ramadan", 250.0).y_ss()

def admissible(mk, y0, f):
    gmin, gmax, tirs, conv = np.inf, -np.inf, [], True
    for c in R.CHO_F:
        for s in R.SI_F:
            sys_ = mk(f, c, s)
            y, ok = R.orbit_fwd(sys_, y0)
            _, _, g = sys_.flow(y, dense=True)
            conv &= ok; gmin = min(gmin, g.min()); gmax = max(gmax, g.max())
            tirs.append(np.mean((g >= 70) & (g <= 180)) * 100)
    return bool(conv and gmin >= HYPO and np.mean(tirs) >= TIR_T and gmax <= GMAX_T)

def bisect(mk, y0, f_in, f_out):
    while abs(f_out - f_in) > TOL:
        mid = 0.5 * (f_in + f_out)
        if admissible(mk, y0, mid):
            f_in = mid
        else:
            f_out = mid
    return f_in

def job(args):
    import os
    os.nice(15)
    model, pid, fmin, fmax = args
    mk, y0 = make(model, pid)
    grid = np.round(np.arange(0.50, 1.3001, STEP), 2)
    # admissible grid points at the ends of the interval
    lo_in = grid[np.searchsorted(grid, fmin - 1e-9)] if fmin > 0.5 + 1e-9 else 0.5
    out = dict(model=model, Patient=pid, f_min_grid_interp=fmin, f_max_grid_interp=fmax)
    out["f_min_refined"] = 0.5 if fmin <= 0.5 + 1e-9 else bisect(mk, y0, round(lo_in, 2), round(lo_in - STEP, 2))
    hi_in = grid[np.searchsorted(grid, fmax + 1e-9) - 1]
    out["f_max_refined"] = 1.3 if fmax >= 1.3 - 1e-9 else bisect(mk, y0, round(hi_in, 2), round(hi_in + STEP, 2))
    out["width_refined"] = out["f_max_refined"] - out["f_min_refined"]
    return out

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    b = pd.read_csv(OUT / "r4_bounds.csv")
    todo = [(r.model, r.Patient, r.f_min, r.f_max) for r in b[b.feasible].itertuples()]
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(job, todo))
    pd.DataFrame(res).to_csv(OUT / "r4_bounds_refined.csv", index=False)
    print(pd.DataFrame(res).round(4).to_string())
