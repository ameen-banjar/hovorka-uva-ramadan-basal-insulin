#!/usr/bin/env python3
"""MED-05 manuscript figures from analysis/out (R1-R4). Writes ../figures/*.pdf|png."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE.parent / "figures"; FIG.mkdir(parents=True, exist_ok=True)
HOV, UVA = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated default palette)
# Grayscale printing (2026-10-27): the two models are also told apart without color:
# Hovorka = solid lines / filled circles, UVA/Padova = dashed lines / open triangles,
# and every series is labelled directly or in a legend.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False})
POL = [1.0, 0.9, 0.8, 0.7, 0.6]

def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)

# Figure 1: fasting equilibrium G*(f) (R1)
h = pd.read_csv(OUT / "hovorka_r1r2.csv"); u = pd.read_csv(OUT / "uva_r1r2.csv")
def gstar(df):
    cols = ["G_ss_mgdl"] + [f"Gstar_{int(f*100)}_mgdl" for f in POL[1:]]
    return df[cols].values
fig, ax = plt.subplots(figsize=(6.3, 3.2))
for row in gstar(h):
    ax.plot([p * 100 for p in POL], row, color=HOV, lw=1, alpha=0.45)
gu = gstar(u)
med, q1, q3 = np.median(gu, 0), np.percentile(gu, 25, 0), np.percentile(gu, 75, 0)
ax.fill_between([p * 100 for p in POL], q1, q3, facecolor=UVA, alpha=0.18, lw=0)
ax.plot([p * 100 for p in POL], q1, color=UVA, lw=0.8, ls=":"); ax.plot([p * 100 for p in POL], q3, color=UVA, lw=0.8, ls=":")
ax.plot([p * 100 for p in POL], med, color=UVA, lw=2, ls="--", marker="^", ms=5, mfc="white")
ax.plot([p * 100 for p in POL], np.median(gstar(h), 0), color=HOV, lw=2, marker="o", ms=4)
ax.text(99, 300, "Hovorka: 10 adults (thin solid lines);\nbold solid line with circles = median", color=INK, ha="left", va="top")
ax.text(76, 160, "UVA/Padova median (dashed, triangles)\nand IQR (dotted, 30 patients)", color=INK, ha="left")
ax.axhline(180, color=MUTED, lw=0.8, ls="--"); ax.text(60.5, 183, "180 mg/dL", color=MUTED, ha="right", fontsize=8)
ax.set_xlim(100.5, 59.5); ax.set_ylim(120, 320)
ax.set_xlabel("Basal rate (% of pre-Ramadan basal)"); ax.set_ylabel("Fasting equilibrium G* (mg/dL)")
save(fig, "fig1_equilibrium")

# Figure 2: computed transitions from the pre-Ramadan periodic orbit (R3)
# (replaces the earlier plot of -24/ln|mu1|: finite-difference multipliers below ~0.01 are not resolved)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
for ax, (m, c, lab, ls) in zip(axs, (("hovorka", HOV, "Hovorka", "-"), ("uva", UVA, "UVA/Padova", "--"))):
    t = pd.read_csv(OUT / f"r3_{m}_transient.csv")
    t = t[t.day <= 4]
    for (pid, pol), g in t.groupby(["Patient", "policy"]):
        ax.plot(g.day, np.maximum(g.dev_max_day_mgdl, 1e-4), color=c, lw=0.8, alpha=0.35, ls=ls)
    med = t.groupby("day").dev_max_day_mgdl.max()
    ax.plot(med.index, np.maximum(med.values, 1e-4), color=INK, lw=1.5, marker="s", ms=4, label="maximum over cases")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 500); ax.set_xticks([1, 2, 3, 4])
    ax.set_title(lab, fontsize=9, color=INK); ax.set_xlabel("Day after the schedule and basal change")
axs[0].set_ylabel("max over the day of |G - G$_{per}$| (mg/dL)")
axs[0].legend(loc="upper right", fontsize=7)
save(fig, "fig2_transitions")

# Figure 3: periodic-orbit TIR vs 30-day simulation TIR (R3)
fig, ax = plt.subplots(figsize=(3.6, 3.4))
sims = {"hovorka": HERE.parent / "results_corrected" / "summary_all_patients_policies.csv",
        "uva": HERE.parent / "data" / "med01_archived_uva_run" / "summary_all_patients_policies.csv"}
for m, c, lab, mk in (("hovorka", HOV, "Hovorka (filled circles)", "o"), ("uva", UVA, "UVA/Padova (open triangles)", "^")):
    d = pd.read_csv(OUT / f"r3_{m}.csv"); d = d[d.policy != "pre"].copy(); d["policy"] = d.policy.astype(float)
    s = pd.read_csv(sims[m]); s = s[s.Period == "Ramadan"]
    j = d.merge(s, left_on=["Patient", "policy"], right_on=["Patient", "Policy"])
    if mk == "o":
        ax.scatter(j.TIR, j.TIR_per, s=14, marker=mk, color=c, alpha=0.8, edgecolor=INK, linewidth=0.3, label=lab)
    else:
        ax.scatter(j.TIR, j.TIR_per, s=18, marker=mk, facecolor="white", edgecolor=c, linewidth=0.9, label=lab)
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
ax.set_xlabel("TIR, 30-day simulation (%)"); ax.set_ylabel("TIR, periodic orbit (%)"); ax.legend(loc="upper left", fontsize=7)
save(fig, "fig3_periodic_vs_simulation")

# Figure 4: safe basal intervals (R4)
b = pd.read_csv(OUT / "r4_bounds.csv"); ph = pd.read_csv(OUT / "r4_bounds_posthoc_subsets.csv")
if (OUT / "r4_bounds_refined.csv").exists():
    rf = pd.read_csv(OUT / "r4_bounds_refined.csv").set_index(["model", "Patient"])
    for i, r in b.iterrows():
        if (r.model, r.Patient) in rf.index:
            b.loc[i, "f_min"] = rf.loc[(r.model, r.Patient), "f_min_refined"]
            b.loc[i, "f_max"] = rf.loc[(r.model, r.Patient), "f_max_refined"]
nom = ph[ph.subset == "nominal only"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
for ax, (model, c) in zip(axs, (("Hovorka", HOV), ("UVA/Padova", UVA))):
    n = nom[nom.model == model].reset_index(drop=True); pre = b[b.model == model].set_index("Patient")
    for i, r in n.iterrows():
        if r.feasible:
            ax.plot([r.f_min * 100, r.f_max * 100], [i, i], color=c, lw=7, alpha=0.25, solid_capstyle="butt")
        q = pre.loc[r.Patient]
        if q.feasible:
            ax.plot([q.f_min * 100, q.f_max * 100], [i, i], color=INK, lw=2.2, solid_capstyle="butt",
                    marker="|", ms=7, mew=1.5)
    ax.axvline(100, color=MUTED, lw=0.8, ls="--"); ax.axvline(90, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(n))); ax.set_yticklabels(n.Patient, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlim(48, 132); ax.set_xlabel("Basal rate (% of pre-Ramadan)"); ax.set_title(model, fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
fig.text(0.5, 0.005, "Wide light bars: nominal day only (post hoc).  Narrow black bars with end ticks: all nine fixed scenarios (pre-specified).\n"
         "Dashed line: 100%; dotted line: 90%.", ha="center", va="bottom", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig4_safe_intervals.{ext}", dpi=300)
plt.close(fig)
print("figures written to", FIG)
