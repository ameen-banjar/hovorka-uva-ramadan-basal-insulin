#!/usr/bin/env python3
"""
MED-05 R4: safe basal interval [f_min, f_max] (fraction of the pre-Ramadan
basal) by continuation of the 24-h periodic orbit over f, with a margin for
day-to-day variation. Hovorka (corrected) and UVA/Padova (simglucose eqs).

CRITERIA — fixed on 2026-10-24 before the scan was run:
  Scenarios S (day-to-day variation), 9 per patient:
    daily carbohydrate factor  c in {0.70, 1.00, 1.30}  (boluses follow c:
                                                          carbs counted correctly)
    insulin sensitivity factor s in {0.80, 1.00, 1.20}  (unannounced; scales
       Hovorka kb1..kb3, UVA/Padova Vmx and kp3; basal and ICR stay nominal)
    Suhoor 02:30, Iftar 18:30/18:45, snack 22:00 (as in R3).
  Sensor margin m = 10 mg/dL (two SDs of the 5 mg/dL CGM noise used in the
    Hovorka simulations).
  Upper bound (hypoglycaemia):  f_max = largest f with
        min over S of min_t G(t) >= 70 + m = 80 mg/dL.
  Lower bound (hyperglycaemia): f_min = smallest f with
        mean over S of TIR >= 70 %  AND  max over S of max_t G(t) <= 300 mg/dL.
  Grid f = 0.50, 0.52, ..., 1.30; bounds refined by linear interpolation of
  the binding quantity between adjacent grid points. Open-loop, noise-free
  periodic orbits (same object as R3).

Usage: r4_safe_basal.py [hovorka|uva|all]   ->  out/r4_<model>_scan.csv
"""
import sys, warnings, time
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import r3_periodic as R3

F_GRID = np.round(np.arange(0.50, 1.3001, 0.02), 2)
CHO_F = [0.70, 1.00, 1.30]
SI_F = [0.80, 1.00, 1.20]

class HovS(R3.HovorkaDay):
    def __init__(self, row, f, c, s):
        super().__init__(row, f, "ramadan")          # u_b and ICR from nominal patient
        for k in ("kb1", "kb2", "kb3"):
            self.p[k] *= s
        tot = R3.H.CHO_DAY * c
        self.meals = [(a, a + d, fr * tot) for a, d, fr in self.day]
        self.bolus = [(a, a + R3.BOLUS_MIN, fr * tot / self.icr * 1000.0) for a, d, fr in self.day]

class _DayTimeout(Exception):
    pass

def _alarm(*_):
    raise _DayTimeout()

LSODA_DAY_LIMIT_S = 20
RK4_STEP_MIN = 0.25
N_RK4_FALLBACK = [0]
RK4_SCENARIOS = set()   # (pid, c, s) that already needed RK4: use it directly afterwards

class UvaS(R3.UVADay):
    """UVA/Padova day map with a fallback: LSODA can chatter for hours when gated
    states sit at exactly zero (child#003, c=1.0, s=1.2, f=1.3 stalled a run on
    2026-10-25). A day that takes more than LSODA_DAY_LIMIT_S seconds is redone with
    fixed-step RK4 (step RK4_STEP_MIN min, states clipped at 0 as in simglucose)."""
    def __init__(self, pid, f, c, s):
        super().__init__(pid, f, "ramadan", R3.CHO[pid.split("#")[0]] * c)
        self.q.Vmx *= s; self.q.kp3 *= s
        self.key = (pid, c, s)

    def flow(self, y0, dense=False, lastF0=None):
        import signal
        if self.key in RK4_SCENARIOS:
            return self._flow_rk4(y0, dense, lastF0)
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(LSODA_DAY_LIMIT_S)
        try:
            return super().flow(y0, dense, lastF0)
        except _DayTimeout:
            N_RK4_FALLBACK[0] += 1
            RK4_SCENARIOS.add(self.key)
            return self._flow_rk4(y0, dense, lastF0)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)

    def _flow_rk4(self, y0, dense, lastF0=None):
        # lastF0: grams of the previous day's last eating episode when that day used another
        # scenario (day-to-day switching, r5_switching.py); otherwise this scenario's own.
        z = np.array(y0, float); y = z[:13].copy(); ts, gs = [], []
        lastF = self.ep_total_last if lastF0 is None else lastF0
        lastQ, Q0, ms, rate, base = float(z[13]), 0.0, 0.0, 0.0, 0.0
        mask = np.ones(13, bool); mask[self.signed] = False      # X (index 6) may be negative
        for a, b in zip(self.seg[:-1], self.seg[1:]):
            mid = 0.5 * (a + b)
            m = [x for x in self.meals if x[0] <= mid < x[1]]
            bo = [x for x in self.bolus if x[0] <= mid < x[1]]
            act = self.Action(CHO=sum(g / (e - st) for st, e, g in m),
                              insulin=self.basal * self.f + sum(U / (e - st) for st, e, U in bo))
            if m and abs(a - m[0][0]) < 1e-9:
                k = self.meals.index(m[0])
                if self.ep_start[k]:
                    Q0 = y[0] + y[1]
                ms, rate, base = m[0][0], m[0][2] / (m[0][1] - m[0][0]), self.ep_base[k]
            if m:
                rhs = lambda t, x, Q0=Q0, ms=ms, rate=rate, base=base: self.model(t, x, act, self.q, Q0, base + rate * (np.floor(t - ms + 1e-9) + 1.0))
            else:
                rhs = lambda t, x, lq=lastQ, lf=lastF: self.model(t, x, act, self.q, lq, lf)
            t = a
            while t < b - 1e-12:
                h = min(RK4_STEP_MIN, b - t)
                k1 = rhs(t, y); k2 = rhs(t + h / 2, y + h / 2 * k1)
                k3 = rhs(t + h / 2, y + h / 2 * k2); k4 = rhs(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y[mask] = np.maximum(y[mask], 0.0); t += h
                if dense and abs(t - round(t)) < 1e-9:
                    ts.append(t); gs.append(self.G(y))
            if m and abs(b - m[0][1]) < 1e-9:
                lastQ, lastF = Q0, base + m[0][2]
        y = np.append(y, lastQ)
        return (y, np.array(ts), np.array(gs)) if dense else y

def orbit_fwd(sys_, y, tol=1e-7, max_days=15):
    for _ in range(max_days):
        y1 = sys_.flow(y)
        if np.max(np.abs(y1 - y) / (np.abs(y1) + 1e-3)) < tol:
            return y1, True
        y = y1
    return y, False

def scan(make, y_start, pid, label):
    rows = []
    for c in CHO_F:
        for s in SI_F:
            y = np.array(y_start, float)
            for f in F_GRID[::-1]:                     # continuation from high to low basal
                sys_ = make(f, c, s)
                y, ok = orbit_fwd(sys_, y)
                _, _, g = sys_.flow(y, dense=True)
                rows.append(dict(model=label, Patient=pid, cho=c, si=s, f=f, converged=ok,
                                 rk4_fallback_days=N_RK4_FALLBACK[0],
                                 Gmin=g.min(), Gmax=g.max(), Gmean=g.mean(),
                                 TIR=float(np.mean((g >= 70) & (g <= 180)) * 100),
                                 TBR=float(np.mean(g < 70) * 100), TAR=float(np.mean(g > 180) * 100)))
    return rows

def job_hov(i):
    row = R3.H.PATIENTS_ADULT[i]
    base = R3.HovorkaDay(row, 1.3, "ramadan")
    return scan(lambda f, c, s: HovS(row, f, c, s), base.y_ss(), base.pid, "Hovorka")

def job_uva(pid):
    base = R3.UVADay(pid, 1.3, "ramadan", 250.0)
    return scan(lambda f, c, s: UvaS(pid, f, c, s), base.y_ss(), pid, "UVA/Padova")

# Run settings (2026-10-25): two earlier runs with 8 workers kept every core of the
# 8-core M1 at 100% for over an hour and froze the machine. Use 3 low-priority workers,
# save each patient as it finishes, and skip patients already saved.
MAX_WORKERS = 3
PART = OUT / "r4_parts"

def _low_priority():
    import os
    os.nice(15)

def _patient_file(name, key):
    return PART / f"{name}_{str(key).replace('#', '_')}.csv"

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    PART.mkdir(exist_ok=True)
    for name, job, items in (("hovorka", job_hov, range(10)), ("uva", job_uva, R3.UVA_IDS)):
        if which not in (name, "all"):
            continue
        keys = [k for k in items if not _patient_file(name, k).exists()]
        t0 = time.time()
        print(f"{name}: {len(items) - len(keys)} saved, {len(keys)} to run", flush=True)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_low_priority) as ex:
            futs = {ex.submit(job, k): k for k in keys}
            for fu in as_completed(futs):
                k = futs[fu]
                pd.DataFrame(fu.result()).to_csv(_patient_file(name, k), index=False)
                print(f"{name}: {k} saved ({time.time()-t0:.0f}s)", flush=True)
        parts = [pd.read_csv(_patient_file(name, k)) for k in items]
        pd.concat(parts).to_csv(OUT / f"r4_{name}_scan.csv", index=False)
