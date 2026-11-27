#!/usr/bin/env python3
"""R8d (post hoc, revision 1): dominant Floquet multiplier for all 240 cases at tight tolerance.

Same cases as r3_periodic.py (40 patients x {pre-Ramadan day, f = 1.0, 0.9, 0.8, 0.7, 0.6}), same
shooting (orbit of each policy started from the pre-Ramadan orbit), but LSODA rtol 1e-12 / atol 1e-14
and a central-difference monodromy matrix with relative step 1e-5 (one-sided second order where the
backward step would leave the non-negative domain), as in Part A of r8_numerical_accuracy.py, which
showed agreement with power iteration. Replaces the forward-difference values (step 1e-6, rtol 1e-9)
of Tables S4/S5. Per-patient checkpoints in out/r8d_parts/, 3 workers at nice 15, 900 s per case.
Output: out/r8d_floquet_all.csv.
"""
import sys, os, time, signal, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"; PART = OUT / "r8d_parts"
sys.path.insert(0, str(HERE))
import r8_numerical_accuracy as A
R3 = A.R3
POL = [1.0, 0.9, 0.8, 0.7, 0.6]

def job(item):
    model, key = item
    os.nice(15); A.set_tol(1e-12, 1e-14)
    if model == "Hovorka":
        row = R3.H.PATIENTS_ADULT[key]; mk = lambda f, k: R3.HovorkaDay(row, f, k); label = "Hovorka"
    else:
        mk = lambda f, k: R3.UVADay(key, f, k, R3.CHO[key.split("#")[0]]); label = "UVA/Padova"
    rows = []
    pre = mk(1.0, "pre")
    def case(s, y0, pol):
        t = time.time(); r = dict(model=label, Patient=s.pid, policy=pol)
        try:
            def work():
                y, _, resid = R3.periodic_orbit(s, y0)
                mu = np.linalg.eigvals(A.mono_central(s, y, 1e-5)); mu = mu[np.argsort(-np.abs(mu))]
                return y, dict(resid=resid, mu1=float(abs(mu[0])), mu2=float(abs(mu[1])))
            y, out = A.guarded(work); r.update(out); r["status"] = "ok"
        except A.Timeout:
            y = None; r["status"] = "timeout"
        r["seconds"] = round(time.time() - t, 1); rows.append(r); print(r, flush=True)
        return y
    y_pre = case(pre, pre.y_ss(), "pre")
    for f in POL:
        s = mk(f, "ramadan")
        case(s, y_pre if y_pre is not None else s.y_ss(), f)
    pd.DataFrame(rows).to_csv(PART / f"{model}_{str(key).replace('#', '_')}.csv", index=False)
    return key

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    PART.mkdir(exist_ok=True)
    items = [("Hovorka", i) for i in range(10)] + [("uva", p) for p in R3.UVA_IDS]
    todo = [it for it in items if not (PART / f"{it[0]}_{str(it[1]).replace('#', '_')}.csv").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(3) as ex:
        for k in ex.map(job, todo):
            print(f"{k} saved ({time.time()-t0:.0f}s)", flush=True)
    d = pd.concat([pd.read_csv(p) for p in sorted(PART.glob("*.csv"))])
    d.to_csv(OUT / "r8d_floquet_all.csv", index=False)
    print("done", len(d), "rows", flush=True)
