#!/usr/bin/env python3
"""
MED-05 — Hovorka (Cambridge) Ramadan basal-policy simulation, CORRECTED.

Derived from MED-01_Pre-Ramadan-TBR/hovorka_simulation/hovorka_ramadan.py
(left untouched). Same ODE structure, cohort (10 adults), meal protocol,
policies, safety rules and CGM noise. Three corrections (2026-10-23):

  C1  Insulin-action activation constants follow Hovorka et al. 2004:
      kb_i = S_I,i * ka_i with S_IT = 51.2e-4, S_ID = 8.2e-4, S_IE = 520e-4
      (per min per mU/L). The old values (0.0034, 0.006, 0.024) were
      15-110x too large and gave basal rates near 0.14 U/day.
  C2  Meal carbohydrate is converted from g to mmol (x 1000/180.16) before
      entering the gut compartments. The old code added grams to a glucose
      mass in mmol, so meals were 5.55x too small.
  C3  Boluses use real insulin units. The old code scaled boluses by
      u_b / REAL_UB_REF, a factor tuned to reach a target TIR. Here each
      patient's insulin-to-carbohydrate ratio (ICR) is computed from the
      model itself, before any outcome is seen: the ICR for which glucose
      returns to the fasting steady state 300 min after a 60 g meal.
      The correction factor uses the 1800 rule on the implied total
      daily dose (basal + 250 g / ICR).
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from pathlib import Path

sys.dont_write_bytecode = True
OUT_DIR = Path(__file__).resolve().parent.parent / "results_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "run.log"
MMOL_PER_G = 1000.0 / 180.16

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

# ============================================================
# HOVORKA 2004 NOMINAL PARAMETERS
# ============================================================
SIT, SID, SIE = 51.2e-4, 8.2e-4, 520e-4
NOMINAL = dict(
    VG=0.16, F01=0.0097, EGP0=0.0161, k12=0.066,
    ka1=0.006, ka2=0.06, ka3=0.03,
    kb1=SIT * 0.006, kb2=SID * 0.06, kb3=SIE * 0.03,   # C1
    tau_s=55.0, ke=0.138, VI=0.12, Ag=0.8, tmax_G=40.0,
)

CHO_DAY = 250.0
PATIENTS_ADULT = [
    # (pid, BW_kg, isf_mult, egp_mult, f01_mult) — unchanged cohort
    ("adult#001", 62, 1.50, 0.80, 0.93),
    ("adult#002", 68, 1.30, 0.85, 0.95),
    ("adult#003", 72, 1.15, 0.90, 0.97),
    ("adult#004", 75, 1.05, 0.95, 0.98),
    ("adult#005", 78, 1.00, 0.98, 1.00),
    ("adult#006", 80, 0.95, 1.00, 1.00),
    ("adult#007", 82, 0.90, 1.02, 1.02),
    ("adult#008", 85, 0.85, 1.08, 1.04),
    ("adult#009", 88, 0.75, 1.15, 1.06),
    ("adult#010", 92, 0.60, 1.22, 1.08),
]

def make_params(row):
    pid, BW, isf_mult, egp_mult, f01_mult = row
    p = dict(NOMINAL)
    p.update(pid=pid, BW=float(BW), isf_mult=isf_mult)
    for k in ("kb1", "kb2", "kb3"):
        p[k] = NOMINAL[k] * isf_mult
    p["EGP0"] = NOMINAL["EGP0"] * egp_mult
    p["F01"] = NOMINAL["F01"] * f01_mult
    return p

# ============================================================
# STEADY STATE
# ============================================================
TARGET_G_SS = 7.5  # mmol/L

def _ss_residual(u_b, p, G_ss):
    BW = p["BW"]
    I = u_b / (p["ke"] * p["VI"] * BW)
    x1, x2, x3 = (p[f"kb{i}"] / p[f"ka{i}"] * I for i in (1, 2, 3))
    Q1 = G_ss * p["VG"] * BW
    Q2 = x1 * Q1 / (p["k12"] + x2)
    return -(p["F01"] * BW + x1 * Q1) + p["k12"] * Q2 + p["EGP0"] * BW * (1 - x3)

def find_basal(p):
    return brentq(_ss_residual, 1e-4, 200.0, args=(p, TARGET_G_SS), xtol=1e-9)

def ss_state(u_b, p):
    BW = p["BW"]
    I = u_b / (p["ke"] * p["VI"] * BW)
    x1, x2, x3 = (p[f"kb{i}"] / p[f"ka{i}"] * I for i in (1, 2, 3))
    Q1 = TARGET_G_SS * p["VG"] * BW
    Q2 = x1 * Q1 / (p["k12"] + x2)
    S = u_b * p["tau_s"]
    return np.array([Q1, Q2, x1, x2, x3, S, S, I, 0.0, 0.0])

# ============================================================
# ODE
# ============================================================
def hovorka_ode(t, y, p, u_b, meals, bolus):
    Q1, Q2, x1, x2, x3, S1, S2, I, D1, D2 = y
    BW = p["BW"]
    Q1 = max(Q1, 0.0); Q2 = max(Q2, 0.0); I = max(I, 0.0)
    D1 = max(D1, 0.0); D2 = max(D2, 0.0)
    G = Q1 / (p["VG"] * BW)
    F01c = p["F01"] * BW if G >= 4.5 else p["F01"] * BW * G / 4.5
    FR = 0.003 * (G - 9.0) * p["VG"] * BW if G > 9.0 else 0.0
    meal_rate = 0.0
    for ts, te, g in meals:
        if ts <= t < te:
            meal_rate += g * MMOL_PER_G / (te - ts)          # C2: mmol/min
    UG = D2 / p["tmax_G"]
    bolus_rate = 0.0
    for ts, te, mU in bolus:
        if ts <= t < te:
            bolus_rate += mU / (te - ts)
    u_total = u_b + bolus_rate
    EGP = max(p["EGP0"] * BW * (1 - x3), 0.0)
    return [
        -(F01c + x1 * Q1) + p["k12"] * Q2 - FR + EGP + UG,
        x1 * Q1 - (p["k12"] + x2) * Q2,
        -p["ka1"] * x1 + p["kb1"] * I,
        -p["ka2"] * x2 + p["kb2"] * I,
        -p["ka3"] * x3 + p["kb3"] * I,
        u_total - S1 / p["tau_s"],
        (S1 - S2) / p["tau_s"],
        S2 / (p["tau_s"] * p["VI"] * BW) - p["ke"] * I,
        p["Ag"] * meal_rate - D1 / p["tmax_G"],
        (D1 - D2) / p["tmax_G"],
    ]

# ============================================================
# MODEL-BASED ICR AND CORRECTION FACTOR (C3)
# ============================================================
ICR_MEAL_G, ICR_HORIZON = 60.0, 300.0

def _g_after_meal(icr, p, u_b):
    y0 = ss_state(u_b, p)
    meals = [(0.0, 15.0, ICR_MEAL_G)]
    bolus = [(0.0, 5.0, ICR_MEAL_G / icr * 1000.0)]
    sol = solve_ivp(hovorka_ode, [0, ICR_HORIZON], y0, args=(p, u_b, meals, bolus),
                    method="RK45", max_step=1.0, rtol=1e-6, atol=1e-8)
    return sol.y[0, -1] / (p["VG"] * p["BW"]) - TARGET_G_SS

def find_icr(p, u_b):
    return brentq(_g_after_meal, 2.0, 200.0, args=(p, u_b), xtol=1e-4)

# ============================================================
# MEAL SCHEDULES (unchanged protocol; boluses in real units)
# ============================================================
RAMADAN_CHO_FACTORS = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30]
SUHOOR_OFFSETS_MIN = [60, 120, 180]

def bolus_mU(g, icr):
    return g / icr * 1000.0

def build_ramadan_schedule(n_days, icr):
    meals, bolus = [], []
    for day in range(n_days):
        base = day * 1440
        total = CHO_DAY * RAMADAN_CHO_FACTORS[day % 10]
        offset = SUHOOR_OFFSETS_MIN[(day // 10) % 3]
        fajr = base + 5 * 60 + 30
        su_s = fajr - offset - 60
        if1_s = base + 18 * 60 + 30
        sn_s = base + 22 * 60
        for ts, te, g in [(su_s, su_s + 60, 0.35 * total), (if1_s, if1_s + 15, 0.10 * total),
                          (if1_s + 15, if1_s + 60, 0.40 * total), (sn_s, sn_s + 40, 0.15 * total)]:
            meals.append((ts, te, g))
            bolus.append((ts, ts + 5, bolus_mU(g, icr)))
    return meals, bolus

def build_preramadan_schedule(n_days, icr):
    meals, bolus = [], []
    for day in range(n_days):
        base = day * 1440
        for ts, te, g in [(base + 480, base + 510, 0.30 * CHO_DAY), (base + 780, base + 810, 0.35 * CHO_DAY),
                          (base + 1140, base + 1170, 0.30 * CHO_DAY), (base + 930, base + 950, 0.05 * CHO_DAY)]:
            meals.append((ts, te, g))
            bolus.append((ts, ts + 5, bolus_mU(g, icr)))
    return meals, bolus

# ============================================================
# SIMULATION ENGINE (unchanged rules; correction bolus in real units)
# ============================================================
STEP, NOISE, N_DAYS, CGM_SEED_BASE = 5, 5.0, 30, 42

def simulate(p, u_b, meals, bolus, n_days, y0, cgm_seed, cf):
    rng = np.random.default_rng(cgm_seed)
    y = y0.copy()
    records, terminal, last_rescue_t = [], False, -240
    for t0 in range(0, n_days * 1440, STEP):
        t1 = t0 + STEP
        G_true = max(y[0], 0.0) / (p["VG"] * p["BW"])
        G_mgdl = float(np.clip(G_true * 18.015 + rng.normal(0, NOISE), 20, 600))
        extra = []
        if G_mgdl >= 300.0 and (t0 - last_rescue_t) >= 240:
            corr_U = min(max((G_mgdl - 250.0) / cf, 0.0), 1.0)
            extra = [(t0, t0 + 5, corr_U * 1000.0)]
            last_rescue_t = t0
        u_eff = 0.0 if G_mgdl < 70.0 else u_b
        win_m = [m for m in meals if m[1] > t0 and m[0] < t1]
        win_b = [b for b in bolus + extra if b[1] > t0 and b[0] < t1]
        sol = solve_ivp(hovorka_ode, [float(t0), float(t1)], y, args=(p, u_eff, win_m, win_b),
                        method="RK45", max_step=1.0, rtol=1e-4, atol=1e-7)
        if sol.success:
            y = sol.y[:, -1]
        y = np.maximum(y, 0.0)
        if y[0] / (p["VG"] * p["BW"]) * 18.015 > 550.0:
            terminal = True
            break
        records.append({"min": t0, "day": t0 // 1440, "CGM_mgdL": G_mgdl, "G_true_mmol": float(G_true)})
    return pd.DataFrame(records), terminal

def outcomes(df):
    g = df["CGM_mgdL"].values; m = float(np.mean(g))
    return dict(TIR=float(np.mean((g >= 70) & (g <= 180)) * 100), TBR=float(np.mean(g < 70) * 100),
                TBR54=float(np.mean(g < 54) * 100), TAR=float(np.mean(g > 180) * 100),
                MeanCGM=m, GMI=3.31 + 0.02392 * m, CV=float(np.std(g) / m * 100), N_obs=len(g))

POLICIES = [1.00, 0.90, 0.80, 0.70, 0.60]

def run_patient(i, row):
    p = make_params(row)
    seed = CGM_SEED_BASE + i
    u_b = find_basal(p)
    icr = find_icr(p, u_b)
    tdd = u_b * 1440 / 1000 + CHO_DAY / icr
    cf = 1800.0 / tdd
    rows = []
    base = dict(Patient=p["pid"], BW=p["BW"], isf_mult=p["isf_mult"], u_b_mU_min=u_b,
                basal_U_day=u_b * 1440 / 1000, ICR_g_per_U=icr, TDD_U=tdd, CF_mgdl_per_U=cf)
    y0 = ss_state(u_b, p)
    m, b = build_preramadan_schedule(N_DAYS, icr)
    df, term = simulate(p, u_b, m, b, N_DAYS, y0, seed, cf)
    df.to_csv(OUT_DIR / f"{p['pid']}_PreRamadan_AllDays.csv", index=False)
    rows.append(dict(base, Period="PreRamadan", Policy=1.0, Terminal=term, **outcomes(df)))
    m, b = build_ramadan_schedule(N_DAYS, icr)
    for f in POLICIES:
        df, term = simulate(p, u_b * f, m, b, N_DAYS, y0.copy(), seed, cf)
        df.to_csv(OUT_DIR / f"{p['pid']}_Ramadan{int(f*100):03d}_AllDays.csv", index=False)
        rows.append(dict(base, Period="Ramadan", Policy=f, Terminal=term, **outcomes(df)))
    return rows

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    LOG.write_text("")
    log(f"MED-05 corrected Hovorka run — kb={NOMINAL['kb1']:.3e},{NOMINAL['kb2']:.3e},{NOMINAL['kb3']:.3e}")
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(run_patient, i, r) for i, r in enumerate(PATIENTS_ADULT)]
        allrows = []
        for fu in futs:
            rows = fu.result(); allrows += rows
            r0 = rows[0]
            log(f"{r0['Patient']}: basal={r0['basal_U_day']:.2f} U/day ICR={r0['ICR_g_per_U']:.1f} g/U "
                f"TDD={r0['TDD_U']:.1f} CF={r0['CF_mgdl_per_U']:.0f} | " +
                " ".join(f"{r['Period'][:3]}{int(r['Policy']*100)}:TIR{r['TIR']:.0f}/TBR{r['TBR']:.1f}/TAR{r['TAR']:.0f}" for r in rows))
    pd.DataFrame(allrows).to_csv(OUT_DIR / "summary_all_patients_policies.csv", index=False)
    log("done")
