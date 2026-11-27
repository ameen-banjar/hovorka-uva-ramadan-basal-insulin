#!/usr/bin/env python3
"""MED-05 R8 (revision 1, post hoc): numerical accuracy of periodic orbits, Floquet
multipliers and admissible-interval quantities.

Answers Reviewer 1 (major 3), Reviewer 3 (points 2 and 7) and Reviewer 4 (minor 2).

DESIGN — fixed on 2026-11-27 before the run:
  A. Floquet multipliers, same 14 cases as floquet_step_check.py (Hovorka adult#001 at
     f = 1.0/0.8/0.6, adult#010 at 1.0/0.6; UVA/Padova adult#001, child#004,
     adolescent#007 at 1.0/0.8/0.6). Periodic orbit recomputed with LSODA at
     rtol 1e-12, atol 1e-14. The dominant multiplier |mu1| = rho(M), M = DPhi(y_per(0)), is
     estimated three ways at that tolerance:
       (a) central differences, relative step 1e-4;
       (b) central differences, relative step 1e-5
           (one-sided second-order formula for states closer to zero than the step);
       (c) power iteration on the day map: 12 iterations of a perturbation of relative
           size 1e-5 about Phi(y_per), renormalised each day; |mu1| = geometric mean of
           the last three growth ratios.
  B. Orbit tolerance: for the same 14 cases, the periodic orbit at rtol 1e-7, 1e-9 (used
     in the paper) and 1e-11 (atol = rtol/100); minimum, maximum, TIR of the 1-min glucose
     profile, and the largest glucose difference from the 1e-11 profile.
  C. Admissible bounds: for the six UVA/Padova patients with a pre-specified admissible
     interval, at f_min and f_max (out/r4_bounds_refined.csv) and all nine scenarios,
     min glucose, max glucose and TIR at rtol 1e-9 and 1e-11; days that used the RK4
     fallback are flagged (they do not depend on rtol).
  Parts A and B: each case has a 900 s time guard (part C relies on the 20-s per-day
  guard of UvaS, which replaces the outer alarm); a case that exceeds it is recorded as "timeout".

Usage: r8_numerical_accuracy.py [A|B|C|all] [--smoke]
Outputs: out/r8_floquet.csv, out/r8_tolerance.csv, out/r8_bounds_tolerance.csv
Single low-priority process.
"""
import sys, warnings, time, signal, os
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import r4_safe_basal as R
R3 = R.R3
SMOKE = "--smoke" in sys.argv
CASE_LIMIT_S = 900

class Timeout(Exception):
    pass

def _alarm(*_):
    raise Timeout()

def set_tol(rtol, atol=None):
    R3.RTOL, R3.ATOL = rtol, (rtol / 100 if atol is None else atol)

def make(model, who, f):
    if model == "Hovorka":
        return R3.HovorkaDay(who, f, "ramadan")
    return R3.UVADay(who, f, "ramadan", R3.CHO[who.split("#")[0]])

CASES = [("Hovorka", R3.H.PATIENTS_ADULT[0], f) for f in (1.0, 0.8, 0.6)] + \
        [("Hovorka", R3.H.PATIENTS_ADULT[9], f) for f in (1.0, 0.6)] + \
        [("UVA/Padova", p, f) for p in ("adult#001", "child#004", "adolescent#007") for f in (1.0, 0.8, 0.6)]
if SMOKE:
    CASES = [CASES[0], CASES[5]]

def tir(g):
    return float(np.mean((g >= 70) & (g <= 180)) * 100)

def mono_central(s, y, rel):
    n = len(y); M = np.zeros((n, n)); F0 = s.flow(y)
    signed = set(getattr(s, "signed", []))
    for j in range(n):
        h = rel * max(abs(y[j]), 1e-2); e = np.zeros(n); e[j] = h
        if y[j] - h < 0 and j not in signed:          # stay in the non-negative domain
            M[:, j] = (-3 * F0 + 4 * s.flow(y + e) - s.flow(y + 2 * e)) / (2 * h)
        else:
            M[:, j] = (s.flow(y + e) - s.flow(y - e)) / (2 * h)
    return M

def mu_power(s, y, eps=1e-5, iters=12):
    w = np.maximum(np.abs(y), 1e-2); F0 = s.flow(y)
    rng = np.random.default_rng(0); d = np.abs(rng.standard_normal(len(y))); d *= eps / np.linalg.norm(d)
    ratios = []
    for _ in range(iters):
        d1 = (s.flow(y + w * d) - F0) / w
        r = np.linalg.norm(d1) / np.linalg.norm(d); ratios.append(r)
        d = d1 * eps / max(np.linalg.norm(d1), 1e-300)
    return float(np.exp(np.mean(np.log(np.maximum(ratios[-3:], 1e-300))))), ratios

def guarded(fn, *a):
    old = signal.signal(signal.SIGALRM, _alarm); signal.alarm(CASE_LIMIT_S)
    try:
        return fn(*a)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def part_A():
    rows = []
    for model, who, f in CASES:
        s = make(model, who, f); r = dict(model=model, Patient=s.pid, policy=f)
        t = time.time()
        try:
            def work():
                set_tol(1e-12, 1e-14)
                y, _, resid = R3.periodic_orbit(s, s.y_ss())
                out = dict(resid=resid)
                for rel in (1e-4, 1e-5):
                    out[f"mu1_central_h{rel:.0e}"] = float(np.max(np.abs(np.linalg.eigvals(mono_central(s, y, rel)))))
                out["mu1_power"], rs = mu_power(s, y)
                out["power_ratios"] = " ".join(f"{x:.3g}" for x in rs)
                return out
            r.update(guarded(work)); r["status"] = "ok"
        except Timeout:
            r["status"] = "timeout"
        r["seconds"] = round(time.time() - t, 1); rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "r8_floquet.csv", index=False)

def part_B():
    rows = []
    for model, who, f in CASES:
        s = make(model, who, f); prof = {}
        for rtol in (1e-11, 1e-9, 1e-7):
            r = dict(model=model, Patient=s.pid, policy=f, rtol=rtol); t = time.time()
            try:
                def work():
                    set_tol(rtol)
                    y, _, resid = R3.periodic_orbit(s, s.y_ss())
                    _, _, g = s.flow(y, dense=True)
                    return y, g, resid
                y, g, resid = guarded(work); prof[rtol] = g
                r.update(status="ok", resid=resid, Gmin=g.min(), Gmax=g.max(), TIR=tir(g),
                         max_abs_diff_vs_1e11=float(np.max(np.abs(g - prof[1e-11]))) if 1e-11 in prof else np.nan)
            except Timeout:
                r["status"] = "timeout"
            r["seconds"] = round(time.time() - t, 1); rows.append(r); print(r, flush=True)
    set_tol(1e-9)
    pd.DataFrame(rows).to_csv(OUT / "r8_tolerance.csv", index=False)

def part_C():
    ref = pd.read_csv(HERE / "out" / "r4_bounds_refined.csv")
    ref = ref[ref.model == "UVA/Padova"]
    rows = []
    for p in ref.itertuples():
        for tag, f in (("f_min", p.f_min_refined), ("f_max", p.f_max_refined)):
            scen = [(c, s) for c in R.CHO_F for s in R.SI_F]
            if SMOKE:
                scen = scen[:1]
            for c, sfac in scen:
                r = dict(Patient=p.Patient, bound=tag, f=f, cho=c, si=sfac); t = time.time()
                for rtol in (1e-9, 1e-11):
                    try:
                        def work():
                            set_tol(rtol)
                            R.N_RK4_FALLBACK[0] = 0; R.RK4_SCENARIOS.clear()
                            sys_ = R.UvaS(p.Patient, f, c, sfac)
                            y, ok = R.orbit_fwd(sys_, sys_.y_ss(), max_days=20)
                            _, _, g = sys_.flow(y, dense=True)
                            return ok, g, R.N_RK4_FALLBACK[0]
                        ok, g, nrk = guarded(work)
                        k = f"{rtol:.0e}"
                        r.update({f"conv_{k}": ok, f"Gmin_{k}": g.min(), f"Gmax_{k}": g.max(),
                                  f"TIR_{k}": tir(g), f"rk4_days_{k}": nrk})
                    except Timeout:
                        r[f"status_{rtol:.0e}"] = "timeout"
                r["seconds"] = round(time.time() - t, 1); rows.append(r); print(r, flush=True)
            if SMOKE:
                break
        if SMOKE:
            break
    set_tol(1e-9)
    pd.DataFrame(rows).to_csv(OUT / "r8_bounds_tolerance.csv", index=False)

if __name__ == "__main__":
    os.nice(15)
    which = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
    if SMOKE:
        OUT = HERE / "out" / "r8_smoke"; OUT.mkdir(exist_ok=True)
    for name, fn in (("A", part_A), ("B", part_B), ("C", part_C)):
        if which in (name, "all"):
            t = time.time(); fn(); print(f"part {name} done ({time.time() - t:.0f}s)", flush=True)
