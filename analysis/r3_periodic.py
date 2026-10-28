#!/usr/bin/env python3
"""
MED-05 R3: 24-h periodic solutions under a representative Ramadan day,
Floquet multipliers, and adaptation speed, for Hovorka (corrected) and
UVA/Padova (simglucose model equations).

Open-loop, noise-free deterministic day (no CGM noise, no suspension,
no safety bolus): the object of the theorem is the forced ODE itself.
Representative Ramadan day = carb factor 1.00, Suhoor offset 120 min
(02:30-03:30), Iftar 18:30 (15 min) + 18:45 (45 min), snack 22:00 (40 min).
Pre-Ramadan day: 08:00 (30), 13:00 (30), 15:30 (20), 19:00 (30).

Usage: r3_periodic.py [hovorka|uva|all]
Outputs: analysis/out/r3_<model>.csv, r3_<model>_transient.csv
"""
import sys, warnings, time
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "code"))
import hovorka_corrected as H

DAY = 1440.0
POL = [1.0, 0.9, 0.8, 0.7, 0.6]
MGDL = 18.015
RTOL, ATOL = 1e-9, 1e-11

RAMADAN_DAY = [(150, 60, 0.35), (1110, 15, 0.10), (1125, 45, 0.40), (1320, 40, 0.15)]  # (start, dur, frac)
PRE_DAY = [(480, 30, 0.30), (780, 30, 0.35), (930, 20, 0.05), (1140, 30, 0.30)]
BOLUS_MIN = 5.0

def segments(day, extra=()):
    pts = {0.0, DAY}
    for s, d, _ in day:
        pts |= {float(s), float(s + d), float(s + BOLUS_MIN)}
    pts |= set(extra)
    return sorted(p for p in pts if 0 <= p <= DAY)

# ------------------------------------------------------------------ Hovorka
class HovorkaDay:
    name = "hovorka"
    def __init__(self, row, f, kind):
        self.p = H.make_params(row); self.pid = self.p["pid"]
        self.u_b = H.find_basal(self.p); self.icr = H.find_icr(self.p, self.u_b)
        self.f, self.day = f, (RAMADAN_DAY if kind == "ramadan" else PRE_DAY)
        tot = H.CHO_DAY
        self.meals = [(s, s + d, fr * tot) for s, d, fr in self.day]
        self.bolus = [(s, s + BOLUS_MIN, fr * tot / self.icr * 1000.0) for s, d, fr in self.day]
        self.seg = segments(self.day)
        self.n = 10
        self.signed = []          # indices allowed to be negative
    def y_ss(self): return H.ss_state(self.u_b, self.p)
    def G(self, y): return y[0] / (self.p["VG"] * self.p["BW"]) * MGDL
    def flow(self, y0, dense=False, lastF0=None):
        y = np.array(y0, float); ts, gs = [], []
        for a, b in zip(self.seg[:-1], self.seg[1:]):
            mid = 0.5 * (a + b)
            m = [x for x in self.meals if x[0] <= mid < x[1]]
            bo = [x for x in self.bolus if x[0] <= mid < x[1]]
            sol = solve_ivp(H.hovorka_ode, [a, b], y, args=(self.p, self.u_b * self.f, m, bo),
                            method="LSODA", rtol=RTOL, atol=ATOL, dense_output=dense)
            y = sol.y[:, -1]
            if dense:
                tt = np.arange(np.ceil(a), b, 1.0); ts += list(tt); gs += list(self.G(sol.sol(tt)) if len(tt) else [])
        return (y, np.array(ts), np.array(gs)) if dense else y

# ------------------------------------------------------------------ UVA/Padova
class UVADay:
    name = "uva"
    _P = None
    def __init__(self, pid, f, kind, role_cho):
        from simglucose.patient.t1dpatient import T1DPatient, Action
        import simglucose
        if UVADay._P is None:
            UVADay._P = pd.read_csv(Path(simglucose.__file__).parent / "params" / "vpatient_params.csv").set_index("Name")
            from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST
            UVADay._Q = pd.read_csv(CONTROL_QUEST).set_index("Name")
        self.model, self.Action = T1DPatient.model, Action
        from types import SimpleNamespace
        row = UVADay._P.loc[pid]
        self.q0 = row
        self.q = SimpleNamespace(**{k: float(v) for k, v in row.items() if k != "patient_history"})
        self.pid, self.f = pid, f
        self.icr = float(UVADay._Q.loc[pid, "CR"])
        self.basal = self.q.u2ss * self.q.BW / 6000.0
        self.day = RAMADAN_DAY if kind == "ramadan" else PRE_DAY
        tot = role_cho
        self.meals = [(s, s + d, fr * tot) for s, d, fr in self.day]   # grams
        self.bolus = [(s, s + BOLUS_MIN, fr * tot / self.icr) for s, d, fr in self.day]  # U
        # Eating episodes as in simglucose: a meal that starts when the previous one ends
        # (Iftar 18:30-18:45 and 18:45-19:30) continues the same episode, so last_Qsto is not
        # re-snapshotted and last_foodtaken keeps accumulating. ep_base[i]: grams already eaten
        # in the episode before meal i; ep_start[i]: meal i opens a new episode. As in
        # T1DPatient.step (1-min steps), the grams eaten so far include the current minute.
        self.ep_start, self.ep_base = [], []
        for i, (ms, me, g) in enumerate(self.meals):
            cont = i > 0 and abs(self.meals[i - 1][1] - ms) < 1e-9
            self.ep_start.append(not cont)
            self.ep_base.append(self.ep_base[-1] + self.meals[i - 1][2] if cont else 0.0)
        self.ep_total_last = self.ep_base[-1] + self.meals[-1][2]
        self.seg = segments(self.day)
        self.n = 14             # 13 model states + gastric memory (Qsto at the start of the last meal)
        self.signed = [6]       # remote insulin action X = I - Ib (relative) may be negative
    def y_ss(self): return np.append(self.q0[[f"x0_{i:2d}" for i in range(1, 14)]].values.astype(float), 0.0)
    def G(self, y): return np.asarray(y)[3] / self.q.Vg
    def flow(self, y0, dense=False, lastF0=None):
        z = np.array(y0, float); y = z[:13].copy(); ts, gs = [], []
        # Gastric-emptying memory (simglucose last_Qsto, last_foodtaken) is carried exactly as a
        # 14th state: z[13] = Qsto at the start of the previous day's last meal, whose grams are
        # last_foodtaken at 00:00. (Before 2026-10-26 it was reset to 0 at midnight.)
        # lastF0: grams of the previous day's last meal when that day used another schedule
        # (first Ramadan day after a pre-Ramadan day); otherwise this schedule's last meal.
        dbar_state = {"lastQ": float(z[13]), "lastF": self.ep_total_last if lastF0 is None else lastF0}
        for a, b in zip(self.seg[:-1], self.seg[1:]):
            mid = 0.5 * (a + b)
            m = [x for x in self.meals if x[0] <= mid < x[1]]
            bo = [x for x in self.bolus if x[0] <= mid < x[1]]
            cho = sum(g / (e - s) for s, e, g in m)                 # g/min
            ins = self.basal * self.f + sum(U / (e - s) for s, e, U in bo)  # U/min
            act = self.Action(CHO=cho, insulin=ins)
            if m:
                ms, me, g = m[0]
                k = self.meals.index(m[0])
                if abs(a - ms) < 1e-9:                                # meal starts
                    if self.ep_start[k]:                              # new episode: Qsto snapshot
                        dbar_state["Q0"] = y[0] + y[1]
                    dbar_state["ms"] = ms; dbar_state["rate"] = g / (me - ms); dbar_state["base"] = self.ep_base[k]
                Q0, ms0, rate, base = dbar_state["Q0"], dbar_state["ms"], dbar_state["rate"], dbar_state["base"]
                rhs = lambda t, x: self.model(t, x, act, self.q, Q0, base + rate * (np.floor(t - ms0 + 1e-9) + 1.0))
            else:
                # between meals: last_Qsto and last_foodtaken keep their final values
                lastQ, lastF = dbar_state["lastQ"], dbar_state["lastF"]
                rhs = lambda t, x: self.model(t, x, act, self.q, lastQ, lastF)
            sol = solve_ivp(rhs, [a, b], y, method="LSODA", rtol=RTOL, atol=ATOL, dense_output=dense)
            y = sol.y[:, -1]
            if m and abs(b - m[0][1]) < 1e-9:                         # meal ends: freeze Dbar inputs
                dbar_state["lastQ"], dbar_state["lastF"] = dbar_state["Q0"], dbar_state["base"] + m[0][2]
            if dense:
                tt = np.arange(np.ceil(a), b, 1.0); ts += list(tt); gs += list(self.G(sol.sol(tt)) if len(tt) else [])
        y = np.append(y, dbar_state["lastQ"])
        return (y, np.array(ts), np.array(gs)) if dense else y

def clip_states(sys_, y):
    """Non-negativity only for states that are physically non-negative."""
    y = np.array(y, float); mask = np.ones(len(y), bool); mask[getattr(sys_, "signed", [])] = False
    y[mask] = np.maximum(y[mask], 0.0)
    return y

# ------------------------------------------------------------------ shooting + Floquet
def periodic_orbit(sys_, y_start, n_fwd=8, newton=3):
    y = np.array(y_start, float)
    for _ in range(n_fwd):
        y = sys_.flow(y)
    for _ in range(newton):
        Fy = sys_.flow(y) - y
        M = monodromy(sys_, y, base=Fy + y)
        try:
            dy = np.linalg.solve(M - np.eye(len(y)), -Fy)
        except np.linalg.LinAlgError:
            break
        y = clip_states(sys_, y + dy)
        if np.max(np.abs(dy) / (np.abs(y) + 1e-6)) < 1e-9:
            break
    resid = np.max(np.abs(sys_.flow(y) - y) / (np.abs(y) + 1e-3))
    M = monodromy(sys_, y)
    return y, M, resid

def monodromy(sys_, y, base=None):
    if base is None:
        base = sys_.flow(y)
    n = len(y); M = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(y[j]), 1e-2)
        e = np.zeros(n); e[j] = h
        M[:, j] = (sys_.flow(y + e) - base) / h
    return M

def tir_tbr(g):
    return float(np.mean((g >= 70) & (g <= 180)) * 100), float(np.mean(g < 70) * 100), float(np.mean(g > 180) * 100)

def analyse(make, label):
    """make(f, kind) -> day system. Returns rows for all policies + pre."""
    rows, trans = [], []
    pre = make(1.0, "pre")
    y_pre, M_pre, r_pre = periodic_orbit(pre, pre.y_ss())
    _, _, g_pre = pre.flow(y_pre, dense=True)
    for f in POL:
        s = make(f, "ramadan")
        y_per, M, resid = periodic_orbit(s, y_pre)
        mu = np.linalg.eigvals(M); mu = mu[np.argsort(-np.abs(mu))]
        _, _, g = s.flow(y_per, dense=True)
        TIR, TBR, TAR = tir_tbr(g)
        # transient: start Ramadan from the pre-Ramadan periodic state at 00:00; the first day
        # carries the pre-Ramadan last meal (amount and stomach content) as gastric memory.
        # dist: |G - G_per| at midnight after day k; dmax: max over day k of |G(t) - G_per(t)|.
        y = y_pre.copy(); dist, dmax = [], []
        for day in range(1, 11):
            y, _, gt = s.flow(y, dense=True, lastF0=getattr(pre, "ep_total_last", None) if day == 1 else None)
            dist.append(abs(s.G(y) - s.G(y_per)))
            dmax.append(float(np.max(np.abs(gt - g))))
        mu1 = abs(mu[0])
        n95 = np.log(0.05) / np.log(mu1) if 0 < mu1 < 1 else np.nan
        rows.append(dict(model=label, Patient=s.pid, policy=f, shooting_resid=resid,
                         mu1=mu1, mu2=abs(mu[1]), mu1_real=float(np.real(mu[0])),
                         tau_eff_h=-24 / np.log(mu1), days_to_95pct=n95,
                         day2_dev_mgdl=dist[1],
                         G_min=g.min(), G_max=g.max(), G_mean=g.mean(),
                         G_fajr_to_maghrib_min=g[(np.arange(len(g)) >= 330) & (np.arange(len(g)) < 1110)].min(),
                         TIR_per=TIR, TBR_per=TBR, TAR_per=TAR,
                         day1_dev_mgdl=dist[0], day3_dev_mgdl=dist[2], day7_dev_mgdl=dist[6]))
        trans += [dict(model=label, Patient=s.pid, policy=f, day=k + 1, dev_mgdl=v, dev_max_day_mgdl=w)
                  for k, (v, w) in enumerate(zip(dist, dmax))]
    TIRp, TBRp, TARp = tir_tbr(g_pre)
    rows.append(dict(model=label, Patient=pre.pid, policy="pre", shooting_resid=r_pre,
                     mu1=abs(np.linalg.eigvals(M_pre)).max(), TIR_per=TIRp, TBR_per=TBRp, TAR_per=TARp,
                     G_min=g_pre.min(), G_max=g_pre.max(), G_mean=g_pre.mean()))
    return rows, trans

def job_hov(i):
    row = H.PATIENTS_ADULT[i]
    return analyse(lambda f, k: HovorkaDay(row, f, k), "Hovorka")

CHO = {"child": 150.0, "adolescent": 220.0, "adult": 250.0}
UVA_IDS = [f"{r}#{n:03d}" for r in ("child", "adolescent", "adult") for n in range(1, 11)]
def job_uva(pid):
    return analyse(lambda f, k: UVADay(pid, f, k, CHO[pid.split("#")[0]]), "UVA/Padova")

def _low_priority():
    import os
    os.nice(15)

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, job, items in (("hovorka", job_hov, range(10)), ("uva", job_uva, UVA_IDS)):
        if which not in (name, "all"):
            continue
        t0 = time.time(); R, T = [], []
        with ProcessPoolExecutor(max_workers=3, initializer=_low_priority) as ex:
            for rows, trans in ex.map(job, items):
                R += rows; T += trans
                print(f"{name}: {rows[0]['Patient']} done  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(R).to_csv(OUT / f"r3_{name}.csv", index=False)
        pd.DataFrame(T).to_csv(OUT / f"r3_{name}_transient.csv", index=False)
