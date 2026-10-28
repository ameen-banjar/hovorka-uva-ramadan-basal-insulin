#!/usr/bin/env python3
"""MED-05 robustness checks added on 2026-10-27 (all post hoc):
 (a) Elasticity of the Hovorka fasting equilibrium over a wide parameter population, from the
     closed form (Theorem 1, Corollary 1): 10,000 parameter sets, S_IT, S_ID, S_IE each scaled by
     a log-uniform factor in [0.5, 2]; EGP0, F01, k12, V_G by uniform factors in [0.7, 1.3];
     basal calibrated so that G* = 7.5 mmol/L (seed 1). Parameter sets with no calibrated
     basal (F01 >= EGP0) are counted and excluded.
 (b) Admissible-interval counts from the R4 scan under alternative criteria (grid resolution):
     hypoglycemia threshold {70, 75, 80} mg/dL x TIR threshold {60, 70} % x Gmax {300, 400} mg/dL.
 (c) Contraction bound of Theorem 2 against computed settling (Hovorka): the per-day factor
     exp(-24 c) from the basal-only rate c, against the observed ratio of consecutive
     end-of-day glucose deviations in the R3 transitions.
Writes out/r6_*.csv and out/r6_robustness.txt."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "code"))
import hovorka_corrected as H
import r4_bounds as B

lines = []
def say(s=""):
    print(s); lines.append(s)

# ---------------------------------------------------------------- (a) elasticity population
N = H.NOMINAL; rng = np.random.default_rng(1); n = 10000
SIT, SID, SIE = (H.SIT * np.exp(rng.uniform(np.log(.5), np.log(2), n)),
                 H.SID * np.exp(rng.uniform(np.log(.5), np.log(2), n)),
                 H.SIE * np.exp(rng.uniform(np.log(.5), np.log(2), n)))
EGP, F01, K12, VG = (N[k] * rng.uniform(.7, 1.3, n) for k in ("EGP0", "F01", "k12", "VG"))
G0 = 7.5
rows = []
for i in range(n):
    a, sit, sid, sie, egp, f01, k12, vg = None, SIT[i], SID[i], SIE[i], EGP[i], F01[i], K12[i], VG[i]
    if f01 >= egp:
        rows.append(dict(ok=False)); continue
    Imax = (1 - f01 / egp) / sie
    h = lambda I: (egp * (1 - sie * I) - f01) * (k12 + sid * I) - G0 * vg * sit * sid * I * I
    I = brentq(h, 1e-9, Imax * (1 - 1e-12), xtol=1e-14)
    t_d = sid * I / (k12 + sid * I); t_e = -sie * I * egp / (egp * (1 - sie * I) - f01)
    rows.append(dict(ok=True, I=I, eps=-2 + t_d + t_e, term_disposal=t_d, term_egp=t_e,
                     frac_of_umax=I / Imax))
E = pd.DataFrame(rows); E.to_csv(OUT / "r6_elasticity_population.csv", index=False)
e = E[E.ok]
say("(a) Hovorka elasticity over a parameter population (closed form, G* = 7.5 mmol/L)")
say(f"    sets: {n}; calibrated: {len(e)}; excluded (F01 >= EGP0): {n - len(e)}")
q = e.eps.quantile([0, .025, .25, .5, .75, .975, 1]).round(3).to_dict()
say(f"    eps: min {q[0]}, 2.5% {q[.025]}, Q1 {q[.25]}, median {q[.5]}, Q3 {q[.75]}, 97.5% {q[.975]}, max {q[1]}")
say(f"    eps < -1 in {int((e.eps < -1).sum())}/{len(e)}; eps < -2 in {int((e.eps < -2).sum())}/{len(e)}; "
    f"eps < -3 in {int((e.eps < -3).sum())}/{len(e)}")
say(f"    disposal term range {e.term_disposal.min():.3f}..{e.term_disposal.max():.3f}; "
    f"EGP term range {e.term_egp.min():.3f}..{e.term_egp.max():.3f}")
u = pd.read_csv(OUT / "uva_r1r2.csv")
say(f"    UVA/Padova cohort for comparison: eps > -1 in {int((u.eps > -1).sum())}/30")

# ---------------------------------------------------------------- (b) criteria sensitivity
say("\n(b) Patients with an admissible interval (all nine scenarios, grid resolution) under alternative criteria")
crit = []
for hypo in (70.0, 75.0, 80.0):
    for tir_t in (60.0, 70.0):
        for gmax in (300.0, 400.0):
            B.HYPO, B.TIR_T, B.GMAX_T = hypo, tir_t, gmax
            r = dict(hypo=hypo, TIR_min=tir_t, Gmax_max=gmax)
            for m, lab in (("hovorka", "Hovorka"), ("uva", "UVA/Padova")):
                s = pd.read_csv(OUT / f"r4_{m}_scan.csv")
                res = [B.bounds_for(d) for _, d in s.groupby("Patient")]
                r[f"{lab}_adm"] = sum(x["feasible"] for x in res)
                r[f"{lab}_n"] = len(res)
                r[f"{lab}_100_inside"] = sum(bool(x["feasible"] and x["f_min"] <= 1.0 <= x["f_max"]) for x in res)
            crit.append(r)
C = pd.DataFrame(crit); C.to_csv(OUT / "r6_criteria_sensitivity.csv", index=False)
say(C.to_string(index=False))
B.HYPO, B.TIR_T, B.GMAX_T = 80.0, 70.0, 300.0

# ---------------------------------------------------------------- (c) contraction bound vs settling
say("\n(c) Hovorka: contraction bound (basal insulin only) vs computed settling")
h = pd.read_csv(OUT / "hovorka_r1r2.csv")[["Patient", "contraction_rate_per_h"]]
t = pd.read_csv(OUT / "r3_hovorka_transient.csv")
w = t.pivot_table(index=["Patient", "policy"], columns="day", values="dev_mgdl")
# consecutive end-of-day ratios, only where both deviations exceed the numerical floor of the
# orbit computation (1e-5 mg/dL; the shooting tolerance leaves deviations near 1e-7 mg/dL)
FLOOR = 1e-5
fac = [(w[k + 1] / w[k]).where((w[k] > FLOOR) & (w[k + 1] > FLOOR)) for k in range(1, 9)]
ratio = pd.concat(fac, axis=1).max(axis=1).rename("observed_factor_max").reset_index()
ratio["n_pairs_above_floor"] = pd.concat(fac, axis=1).notna().sum(axis=1).values
ratio = ratio.merge(h, on="Patient")
ratio["bound_factor_per_day"] = np.exp(-24 * ratio.contraction_rate_per_h)
ratio.to_csv(OUT / "r6_contraction_vs_settling.csv", index=False)
say(f"    bound exp(-24c): {ratio.bound_factor_per_day.min():.3f}..{ratio.bound_factor_per_day.max():.3f}")
ok = ratio.dropna(subset=["observed_factor_max"])
say(f"    cases with at least one day pair above the 1e-5 mg/dL floor: {len(ok)}/{len(ratio)}")
say(f"    largest observed end-of-day factor per case: median {ok.observed_factor_max.median():.4f}, "
    f"max {ok.observed_factor_max.max():.4f}")
say(f"    observed <= bound in {int((ok.observed_factor_max <= ok.bound_factor_per_day).sum())}/{len(ok)} cases")
(OUT / "r6_robustness.txt").write_text("\n".join(lines) + "\n")
