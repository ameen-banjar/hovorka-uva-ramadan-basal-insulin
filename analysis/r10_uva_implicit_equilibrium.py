#!/usr/bin/env python3
"""MED-05 R10 (revision 1, Reviewer 4 minor 4; Reviewer 3 point 1): implicit scalar
characterization of the UVA/Padova fasting equilibrium and a check against the numerical
equilibria of r1_r2_analysis.py (out/uva_r1r2.csv).

At a fasting equilibrium (no meal, constant basal u = f u2ss) the insulin states solve a
linear system, which gives plasma insulin I* = It*, Id* = I*, X* = I* - Ib. Writing
V(f) = Vm0 + Vmx X* and E(Gp) for renal excretion, the two glucose equations reduce to
    h(Gp) := max{kp1 - kp2 Gp - kp3 I*, 0} - Fsnc - E(Gp) - V(f) Gt(Gp) / (Km0 + Gt(Gp)) = 0,
where Gt(Gp) > 0 is the positive root of  k2 Gt^2 + (k2 Km0 + V - k1 Gp) Gt - k1 Km0 Gp = 0.
If V(f) >= 0, Gt is increasing in Gp and h is strictly decreasing while EGP > 0, so the
root is unique; it exists iff kp1 - kp3 I* > Fsnc. Writes out/r10_uva_implicit.csv."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_periodic as R3
from simglucose.patient.t1dpatient import T1DPatient

POL = [1.0, 0.9, 0.8, 0.7, 0.6]
num = pd.read_csv(HERE / "out" / "uva_r1r2.csv").set_index("Patient")

def insulin_ss(q, f):
    u = f * q.u2ss                                   # pmol/kg/min (simglucose basal)
    x10 = u / (q.ka1 + q.kd); x11 = q.kd * x10 / q.ka2
    # x5, x9: -(m2+m4) x5 + m1 x9 + ka1 x10 + ka2 x11 = 0 ; -(m1+m30) x9 + m2 x5 = 0
    A = np.array([[-(q.m2 + q.m4), q.m1], [q.m2, -(q.m1 + q.m30)]])
    x5, x9 = np.linalg.solve(A, [-(q.ka1 * x10 + q.ka2 * x11), 0.0])
    return x5 / q.Vi

rows = []
for pid in R3.UVA_IDS:
    q = T1DPatient.withName(pid)._params
    for f in POL:
        I = insulin_ss(q, f); V = q.Vm0 + q.Vmx * (I - q.Ib)
        def Gt(Gp):
            b = q.k2 * q.Km0 + V - q.k1 * Gp
            return (-b + np.sqrt(b * b + 4 * q.k2 * q.k1 * q.Km0 * Gp)) / (2 * q.k2)
        E = lambda Gp: q.ke1 * (Gp - q.ke2) if Gp > q.ke2 else 0.0
        h = lambda Gp: max(q.kp1 - q.kp2 * Gp - q.kp3 * I, 0.0) - q.Fsnc - E(Gp) - V * Gt(Gp) / (q.Km0 + Gt(Gp))
        exists = (q.kp1 - q.kp3 * I) > q.Fsnc
        Gp = brentq(h, 1e-9, 1e6, xtol=1e-12) if (exists and V >= 0 and h(1e6) < 0) else np.nan
        ref = num.loc[pid, "G_ss_mgdl"] if f == 1.0 else num.loc[pid, f"Gstar_{int(f*100)}_mgdl"]
        rows.append(dict(Patient=pid, f=f, I_star=I, I_over_Ib=I / q.Ib, V_uptake=V, exists=exists,
                         G_implicit_mgdl=Gp / q.Vg, G_numerical_mgdl=ref,
                         rel_diff=abs(Gp / q.Vg - ref) / ref if np.isfinite(Gp) else np.nan))
r = pd.DataFrame(rows); r.to_csv(HERE / "out" / "r10_uva_implicit.csv", index=False)
print("I*/Ib range:", r.I_over_Ib.min(), r.I_over_Ib.max())
print("cases with V>=0:", int((r.V_uptake >= 0).sum()), "of", len(r), "; existence holds:", int(r.exists.sum()))
print("max rel diff (V>=0):", r.rel_diff.max())
print(r[r.V_uptake < 0][["Patient", "f", "V_uptake", "G_numerical_mgdl"]].round(2).to_string())
