#!/usr/bin/env python3
"""MED-05: compare the UVA/Padova day map (r3_periodic.UVADay) with simglucose's own
T1DPatient.step event logic (eating-episode detection, last_Qsto/last_foodtaken) for the
same 1-min inputs over 3 days. Writes out/simglucose_equivalence_check.csv."""
import sys, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_periodic as R
from simglucose.patient.t1dpatient import T1DPatient, Action

def inputs_per_minute(s, t):
    cho = sum(g / (e - st) for st, e, g in s.meals if st <= t < e)
    ins = s.basal * s.f + sum(U / (e - st) for st, e, U in s.bolus if st <= t < e)
    return cho, ins

rows = []
for pid, f in (("adult#001", 1.0), ("adult#002", 0.5), ("child#001", 0.8), ("adolescent#006", 0.9)):
    s = R.UVADay(pid, f, "ramadan", R.CHO[pid.split("#")[0]])
    z = s.y_ss()
    pat = T1DPatient.withName(pid)                     # default sample_time 1 min
    pat._state = z[:13].copy(); pat._odesolver.set_initial_value(z[:13].copy(), 0)
    pat._last_Qsto, pat._last_foodtaken = 0.0, s.ep_total_last
    pat._last_action = Action(CHO=0.0, insulin=0.0)
    gmap, gsim = [], []
    for day in range(3):
        z, _, g = s.flow(z, dense=True); gmap += list(g)
        for t in range(1440):
            cho, ins = inputs_per_minute(s, t)
            gsim.append(pat.state[3] / pat._params.Vg)
            pat.step(Action(CHO=cho, insulin=ins))
    d = np.abs(np.array(gmap) - np.array(gsim))
    rows.append(dict(Patient=pid, policy=f, max_abs_diff_mgdl=d.max(), max_abs_diff_day3_mgdl=d[2880:].max()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / "out" / "simglucose_equivalence_check.csv", index=False)
