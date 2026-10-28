#!/usr/bin/env python3
"""MED-05 R5: day-to-day switching between the nine fixed scenarios of R4.

DESIGN — fixed on 2026-10-27 before the run (not preregistered):
  For every patient (10 Hovorka, 30 UVA/Padova) and basal fractions f in {1.0, 0.9, 0.8},
  plus the midpoint of the patient's pre-specified admissible interval when one exists
  (out/r4_bounds_refined.csv), the day-to-day scenario (c, s) of R4 is drawn independently
  and uniformly from the nine scenarios on each of 30 days. Five sequences (seeds 0-4) are
  shared by all patients. Each run starts at 00:00 on the nominal-scenario periodic orbit at f.
  The state, including the UVA/Padova gastric memory (the grams of the previous day's last
  eating episode), is carried across days; the sensitivity factor changes at midnight.
  Glucose is evaluated at 1-minute intervals.
  Outcomes per (patient, f, seed): minimum and maximum glucose over the 30 days, mean daily
  TIR; compared with the envelope of the nine fixed-scenario periodic orbits at the same f
  (minimum of their minima, maximum of their maxima, mean of their TIRs).
  Admissible under switching: the R4 criteria applied over all five sequences jointly
  (min >= 80 mg/dL, mean TIR >= 70 %, max <= 300 mg/dL).

Usage: r5_switching.py [hovorka|uva|all]  ->  out/r5_switching.csv
Runs with 3 low-priority workers and saves each patient as it finishes (out/r5_parts/).
"""
import sys, warnings, time
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import r4_safe_basal as R

F_LIST = [1.0, 0.9, 0.8]
N_DAYS, SEEDS = 30, range(5)
SCEN = [(c, s) for c in R.CHO_F for s in R.SI_F]
SEQ = {sd: np.random.default_rng(sd).integers(0, len(SCEN), N_DAYS) for sd in SEEDS}
PART = OUT / "r5_parts"
MAX_WORKERS = 3

def tir(g):
    return float(np.mean((g >= 70) & (g <= 180)) * 100)

def run_f(mk, y_start, f, label, pid, tag):
    systems = [mk(f, c, s) for c, s in SCEN]
    nom = systems[SCEN.index((1.0, 1.0))]
    y_nom, ok_nom = R.orbit_fwd(nom, np.array(y_start, float), max_days=20)
    env_min, env_max, env_tir, env_ok = np.inf, -np.inf, [], True
    for sys_ in systems:
        y, ok = R.orbit_fwd(sys_, y_nom)
        _, _, g = sys_.flow(y, dense=True)
        env_ok &= ok; env_min = min(env_min, g.min()); env_max = max(env_max, g.max()); env_tir.append(tir(g))
    rows = []
    for sd in SEEDS:
        y, prev = y_nom.copy(), nom
        gmin, gmax, tirs, below = np.inf, -np.inf, [], 0
        for k in SEQ[sd]:
            sys_ = systems[k]
            y, _, g = sys_.flow(y, dense=True, lastF0=getattr(prev, "ep_total_last", None))
            gmin = min(gmin, g.min()); gmax = max(gmax, g.max()); tirs.append(tir(g))
            below += int(g.min() < env_min - 1.0)
            prev = sys_
        rows.append(dict(model=label, Patient=pid, f=f, f_tag=tag, seed=sd,
                         sw_Gmin=gmin, sw_Gmax=gmax, sw_TIR=float(np.mean(tirs)),
                         env_Gmin=env_min, env_Gmax=env_max, env_TIR=float(np.mean(env_tir)),
                         env_converged=bool(env_ok and ok_nom), days_below_env_min_1mgdl=below))
    return rows

def job(args):
    model, key, mid = args
    if model == "hovorka":
        row = R.R3.H.PATIENTS_ADULT[key]
        base = R.R3.HovorkaDay(row, 1.0, "ramadan")
        mk, pid, label = (lambda f, c, s: R.HovS(row, f, c, s)), base.pid, "Hovorka"
    else:
        base = R.R3.UVADay(key, 1.0, "ramadan", 250.0)
        mk, pid, label = (lambda f, c, s: R.UvaS(key, f, c, s)), key, "UVA/Padova"
    fl = [(f, "fixed") for f in F_LIST] + ([(round(mid, 4), "midpoint")] if mid is not None else [])
    rows = []
    for f, tag in fl:
        rows += run_f(mk, base.y_ss(), f, label, pid, tag)
    return rows

def _low_priority():
    import os
    os.nice(15)

def _file(model, key):
    return PART / f"{model}_{str(key).replace('#', '_')}.csv"

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    PART.mkdir(exist_ok=True)
    ref = pd.read_csv(OUT / "r4_bounds_refined.csv")
    mids = {r.Patient: 0.5 * (r.f_min_refined + r.f_max_refined) for r in ref.itertuples()}
    items = []
    if which in ("hovorka", "all"):
        items += [("hovorka", i, None) for i in range(10)]
    if which in ("uva", "all"):
        items += [("uva", p, mids.get(p)) for p in R.R3.UVA_IDS]
    todo = [it for it in items if not _file(it[0], it[1]).exists()]
    t0 = time.time()
    print(f"{len(items) - len(todo)} saved, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_low_priority) as ex:
        futs = {ex.submit(job, it): it for it in todo}
        for fu in as_completed(futs):
            it = futs[fu]
            pd.DataFrame(fu.result()).to_csv(_file(it[0], it[1]), index=False)
            print(f"{it[0]} {it[1]} saved ({time.time() - t0:.0f}s)", flush=True)
    pd.concat([pd.read_csv(_file(m, k)) for m, k, _ in items]).to_csv(OUT / "r5_switching.csv", index=False)
    print("done", flush=True)
