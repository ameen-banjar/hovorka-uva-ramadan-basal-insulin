#!/usr/bin/env python3
"""MED-05 Supplementary Tables S1-S8 as LaTeX (analysis/out/supplementary_tables.tex)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DST = OUT / "supplementary_tables.tex"
sys.path.insert(0, str(HERE.parent / "code"))
import hovorka_corrected as H

def esc(s): return str(s).replace("#", r"\#").replace("_", r"\_")
def f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out   # no "-0"

def table(caption, label, cols, header, rows, size=r"\footnotesize"):
    out = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}", size,
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)

parts = []

# S1: Hovorka cohort and calibrated doses
s = pd.read_csv(HERE.parent / "results_corrected" / "summary_all_patients_policies.csv")
s = s[s.Period == "PreRamadan"].set_index("Patient")
rows = []
for pid, BW, isf, egp, f01 in H.PATIENTS_ADULT:
    r = s.loc[pid]
    rows.append([esc(pid), f"{BW}", f(isf), f(egp), f(f01), f(r.basal_U_day), f(r.ICR_g_per_U, 1), f(r.TDD_U, 1), f(r.CF_mgdl_per_U, 0)])
parts.append(table("Hovorka virtual cohort (10 adults): parameter multipliers and calibrated doses. Basal rate gives a fasting equilibrium of 135 mg/dL; ICR returns glucose to that equilibrium 300 min after a 60 g meal; CF by the 1800 rule on TDD = basal + 250 g/ICR.",
                   "tab:S1", "lrrrrrrrr",
                   ["Patient", "BW (kg)", "$S_I$ mult.", "$EGP_0$ mult.", "$F_{01}$ mult.", "Basal (U/day)", "ICR (g/U)", "TDD (U)", "CF (mg/dL/U)"], rows))

# S2: equilibrium, elasticity and stability
h = pd.read_csv(OUT / "hovorka_r1r2.csv")
rows = [[esc(r.Patient), f(r.eps), f(r.Gstar_90_mgdl, 0), f(r.Gstar_80_mgdl, 0), f(r.Gstar_60_mgdl, 0),
         f(r.G_max_zero_insulin_mgdl, 0), f(r.tau_slow_h, 1), f(r.tau_slow_h_at_80, 1), f(r.contraction_rate_per_h, 3)] for r in h.itertuples()]
parts.append(table("Hovorka: elasticity $\\varepsilon$ at 100\\% basal, fasting equilibrium $G^*$ (mg/dL) at 90/80/60\\% (equilibrium at 100\\% = 135 mg/dL), zero-insulin bound $G_{\\max}$, slow glucose time constant (h) at 100\\% and 80\\% (full glucose block including $d(G)$; at 80\\% the equilibrium is above the renal threshold), and weighted-$\\ell_1$ contraction rate (h$^{-1}$, $\\delta=S_{ID}I/(2k_{12})$).",
                   "tab:S2a", "lrrrrrrrr",
                   ["Patient", "$\\varepsilon$", "$G^*_{90}$", "$G^*_{80}$", "$G^*_{60}$", "$G_{\\max}$", "$\\tau_{100}$", "$\\tau_{80}$", "Rate"], rows))
u = pd.read_csv(OUT / "uva_r1r2.csv")
import simglucose
P = pd.read_csv(Path(simglucose.__file__).parent / "params" / "vpatient_params.csv").set_index("Name")
rows = [[esc(r.Patient), f(r.G_ss_mgdl, 0), f(r.eps), f(r.Gstar_90_mgdl, 0), f(r.Gstar_80_mgdl, 0), f(r.Gstar_60_mgdl, 0),
         f(r.tau_slow_h, 2), f(r.contraction_rate_per_h, 3), f(1 - P.loc[r.Patient, "Vm0"] / (P.loc[r.Patient, "Vmx"] * P.loc[r.Patient, "Ib"]))] for r in u.itertuples()]
parts.append(table("UVA/Padova: basal equilibrium, elasticity $\\varepsilon$, fasting equilibrium $G^*$ (mg/dL) at 90/80/60\\%, slow glucose time constant (h), $\\ell_1$ contraction rate (h$^{-1}$), and $f_{\\mathrm{crit}}$ (main text, Proposition~1).",
                   "tab:S2b", "lrrrrrrrr",
                   ["Patient", "$G^*_{100}$", "$\\varepsilon$", "$G^*_{90}$", "$G^*_{80}$", "$G^*_{60}$", "$\\tau$", "Rate", "$f_{\\mathrm{crit}}$"], rows, size=r"\scriptsize"))

# S3: Floquet multipliers
for m, lab, key in (("hovorka", "Hovorka", "S3a"), ("uva", "UVA/Padova", "S3b")):
    d = pd.read_csv(OUT / f"r3_{m}.csv"); d = d[d.policy != "pre"].copy(); d["policy"] = d.policy.astype(float)
    mu = d.pivot(index="Patient", columns="policy", values="mu1")
    tir = d.pivot(index="Patient", columns="policy", values="TIR_per")
    order = [p for p in (["adult#%03d" % i for i in range(1, 11)] if m == "hovorka" else
             [f"{r}#{n:03d}" for r in ("child", "adolescent", "adult") for n in range(1, 11)])]
    rows = [[esc(p)] + [f"{mu.loc[p, q]:.2e}" for q in (1.0, 0.9, 0.8, 0.7, 0.6)] + [f(tir.loc[p, q], 1) for q in (1.0, 0.9, 0.8, 0.7, 0.6)] for p in order]
    parts.append(table(f"{lab}: dominant Floquet multiplier $|\\mu_1|$ from a finite-difference monodromy matrix (relative step $10^{{-6}}$; values below about 0.01 are not resolved, see main text) and TIR (\\%) of the deterministic Ramadan periodic orbit, by basal policy.",
                       f"tab:{key}", "lrrrrrrrrrr",
                       ["Patient"] + [f"$|\\mu_1|$ {int(q*100)}\\%" for q in (1.0, 0.9, 0.8, 0.7, 0.6)] + [f"TIR {int(q*100)}\\%" for q in (1.0, 0.9, 0.8, 0.7, 0.6)],
                       rows, size=r"\scriptsize"))

# S4: safe intervals
b = pd.read_csv(OUT / "r4_bounds.csv"); ph = pd.read_csv(OUT / "r4_bounds_posthoc_subsets.csv")
rf = pd.read_csv(OUT / "r4_bounds_refined.csv").set_index(["model", "Patient"]) if (OUT / "r4_bounds_refined.csv").exists() else None
if rf is not None:
    for i, r in b.iterrows():
        if (r.model, r.Patient) in rf.index:
            b.loc[i, "f_min"] = rf.loc[(r.model, r.Patient), "f_min_refined"]
            b.loc[i, "f_max"] = rf.loc[(r.model, r.Patient), "f_max_refined"]
rows = []
for r in b.itertuples():
    n = ph[(ph.model == r.model) & (ph.Patient == r.Patient) & (ph.subset == "nominal only")].iloc[0]
    c = ph[(ph.model == r.model) & (ph.Patient == r.Patient) & (ph.subset == "carbs only (si=1)")].iloc[0]
    sv = ph[(ph.model == r.model) & (ph.Patient == r.Patient) & (ph.subset == "sensitivity only (cho=1)")].iloc[0]
    def iv(x): return f"[{f(x.f_min)}, {f(x.f_max)}]" if x.feasible else "none"
    rows.append([esc(r.model), esc(r.Patient), iv(r), iv(n), iv(c), iv(sv), f(r.Gmin_worst_at_100, 0), f(r.TIR_mean_at_100, 1)])
parts.append(table("Admissible basal interval $[f_{\\min},f_{\\max}]$ per patient: all nine fixed scenarios (pre-specified criteria; bounds re-solved by bisection to 0.001) and, post hoc, the nominal day, carbohydrate variation only, and sensitivity variation only. Last columns: worst-case minimum glucose (mg/dL) and scenario-mean TIR (\\%) at 100\\% basal.",
                   "tab:S4", "llllllrr",
                   ["Model", "Patient", "All nine", "Nominal", "Carbs only", "Sens. only", "$G_{\\min}$", "TIR"], rows, size=r"\scriptsize"))

# S7: criteria sensitivity (post hoc)
C = pd.read_csv(OUT / "r6_criteria_sensitivity.csv")
rows = [[f(r.hypo, 0), f(r.TIR_min, 0), f(r.Gmax_max, 0), f"{r.Hovorka_adm}/10", str(r.Hovorka_100_inside),
         f"{r.UVA_Padova_adm}/30", str(r.UVA_Padova_100_inside)] for r in C.rename(columns=lambda c: c.replace("/", "_")).itertuples()]
parts.append(table("Post hoc: number of patients with an admissible fixed basal fraction in 0.50--1.30 across all nine fixed scenarios, under alternative criteria (grid resolution 0.02, no bisection). Pre-specified criteria: 80 mg/dL, 70\\%, 300 mg/dL. 100\\% inside: patients whose interval contains the pre-Ramadan basal rate.",
                   "tab:S7", "rrrrrrr",
                   ["Min. glucose $\\ge$", "Mean TIR $\\ge$", "Max. glucose $\\le$", "Hovorka adm.", "Hovorka 100\\% inside", "UVA/Padova adm.", "UVA/Padova 100\\% inside"], rows))

# S8: day-to-day switching (post hoc design fixed before the run)
W = pd.read_csv(OUT / "r5_switching.csv")
g = W.groupby(["model", "Patient", "f", "f_tag"]).agg(sw_min=("sw_Gmin", "min"), sw_max=("sw_Gmax", "max"), sw_tir=("sw_TIR", "mean"),
                                                     env_min=("env_Gmin", "first"), env_max=("env_Gmax", "first"), env_tir=("env_TIR", "first")).reset_index()
g = g[g.f_tag == "midpoint"]
rows = [[esc(r.model), esc(r.Patient), f(r.f, 3), f(r.env_min, 1), f(r.sw_min, 1), f(r.env_max, 1), f(r.sw_max, 1), f(r.env_tir, 1), f(r.sw_tir, 1)] for r in g.itertuples()]
parts.append(table("Post hoc: day-to-day switching between the nine scenarios at the midpoint of each pre-specified admissible interval. Envelope: the nine fixed-scenario periodic orbits at the same $f$; switching: five 30-day sequences with the scenario drawn independently each day. Glucose in mg/dL, TIR in \\%.",
                   "tab:S8", "llrrrrrrr",
                   ["Model", "Patient", "$f$", "Env. min", "Switch. min", "Env. max", "Switch. max", "Env. TIR", "Switch. TIR"], rows))

DST.write_text("% generated by analysis/make_supplementary.py\n" + "\n\n".join(parts) + "\n")
print("written", DST)
