#!/usr/bin/env python3
"""MED-05: sensitivity of the dominant Floquet multiplier to the finite-difference step
(h = 1e-5, 1e-6, 1e-7 relative) on selected patients and policies. Writes out/floquet_step_check.csv."""
import sys, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_periodic as R

def mono(sys_, y, rel):
    base = sys_.flow(y); n = len(y); M = np.zeros((n, n))
    for j in range(n):
        h = rel * max(abs(y[j]), 1e-2); e = np.zeros(n); e[j] = h
        M[:, j] = (sys_.flow(y + e) - base) / h
    return M

rows = []
cases = [("Hovorka", R.H.PATIENTS_ADULT[0], f) for f in (1.0, 0.8, 0.6)] + \
        [("Hovorka", R.H.PATIENTS_ADULT[9], f) for f in (1.0, 0.6)] + \
        [("UVA/Padova", p, f) for p in ("adult#001", "child#004", "adolescent#007") for f in (1.0, 0.8, 0.6)]
for model, who, f in cases:
    s = R.HovorkaDay(who, f, "ramadan") if model == "Hovorka" else R.UVADay(who, f, "ramadan", R.CHO[who.split("#")[0]])
    y, _, _ = R.periodic_orbit(s, s.y_ss())
    r = dict(model=model, Patient=s.pid, policy=f)
    for rel in (1e-5, 1e-6, 1e-7):
        r[f"mu1_h{rel:.0e}"] = float(np.max(np.abs(np.linalg.eigvals(mono(s, y, rel)))))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(HERE / "out" / "floquet_step_check.csv", index=False)
