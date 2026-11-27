#!/usr/bin/env python3
"""MED-05 R7 (revision 1, post hoc): temporally correlated, continuous day-to-day variation.

Answers Reviewer 1 (major 2) and Reviewer 3 (point 5): variation that is continuous,
serially correlated or asymmetric, with more sequences than the five of R5.

DESIGN — fixed on 2026-11-27 before the run (post hoc, not preregistered):
  Daily factors are continuous. A latent AR(1) process z_k = rho z_{k-1} + sqrt(1-rho^2) e_k,
  e_k ~ N(0,1), z_0 ~ N(0,1) (stationary), is drawn separately for carbohydrate and for
  sensitivity; each is mapped through the normal CDF to a uniform marginal:
      c_k = 0.70 + 0.60 Phi(z^c_k)            (same range as the R4 grid, 0.70-1.30)
      s_k = 0.80 + 0.40 Phi(z^s_k)            (0.80-1.20)
  Variants:
      iid      rho = 0.0
      ar05     rho = 0.5
      ar08     rho = 0.8
      ar08_up  rho = 0.8, asymmetric sensitivity s_k = 1.00 + 0.20 Phi(z^s_k)
               (sensitivity only rises: the hypoglycaemia direction)
  Factors are rounded to 0.01 (c) and 0.005 (s) so that day maps can be cached.
  Eight sequences (seeds 100-107) per variant, shared by all patients; 30 days.
  Patients and f exactly as R5: f in {1.0, 0.9, 0.8} and the midpoint of each pre-specified
  admissible interval. Start on the nominal periodic orbit at f; full state, including the
  UVA/Padova gastric memory, carried across midnight. Boluses follow c (as in R4).
  Envelope = the nine fixed-scenario periodic orbits at the same f, taken from
  out/r5_switching.csv (computed there with the same code).
  Admissible under a variant: R4 criteria over the eight sequences jointly
  (min >= 80 mg/dL, mean TIR >= 70 %, max <= 300 mg/dL).

Usage: r7_correlated.py [hovorka|uva|all] [--smoke]  ->  out/r7_correlated.csv
3 low-priority workers; each patient saved as it finishes (out/r7_parts/).
"""
import sys, warnings, time
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import r4_safe_basal as R

N_DAYS = 30
SEEDS = range(100, 108)
VARIANTS = {"iid": (0.0, False), "ar05": (0.5, False), "ar08": (0.8, False), "ar08_up": (0.8, True)}
PART = OUT / "r7_parts"
MAX_WORKERS = 3
SMOKE = "--smoke" in sys.argv

def ar1(rng, rho, n):
    z = np.empty(n); z[0] = rng.standard_normal()
    for k in range(1, n):
        z[k] = rho * z[k - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return z

def sequence(variant, seed):
    rho, up = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    zc, zs = ar1(rng, rho, N_DAYS), ar1(rng, rho, N_DAYS)
    c = np.round(0.70 + 0.60 * norm.cdf(zc), 2)
    s = 1.00 + 0.20 * norm.cdf(zs) if up else 0.80 + 0.40 * norm.cdf(zs)
    s = np.round(s / 0.005) * 0.005
    return list(zip(c, np.round(s, 3)))

SEQ = {(v, sd): sequence(v, sd) for v in VARIANTS for sd in SEEDS}

def tir(g):
    return float(np.mean((g >= 70) & (g <= 180)) * 100)

def run_f(mk, y_start, f, label, pid, tag, env):
    cache = {}
    def get(c, s):
        if (c, s) not in cache:
            cache[(c, s)] = mk(f, c, s)
        return cache[(c, s)]
    nom = get(1.0, 1.0)
    y_nom, ok_nom = R.orbit_fwd(nom, np.array(y_start, float), max_days=20)
    rows = []
    variants = ["ar08"] if SMOKE else list(VARIANTS)
    seeds = list(SEEDS)[:1] if SMOKE else list(SEEDS)
    for v in variants:
        for sd in seeds:
            y, prev = y_nom.copy(), nom
            gmin, gmax, tirs, below, days_lt70 = np.inf, -np.inf, [], 0, 0
            seq = SEQ[(v, sd)][:3] if SMOKE else SEQ[(v, sd)]
            for c, s in seq:
                sys_ = get(float(c), float(s))
                y, _, g = sys_.flow(y, dense=True, lastF0=getattr(prev, "ep_total_last", None))
                gmin = min(gmin, g.min()); gmax = max(gmax, g.max()); tirs.append(tir(g))
                below += int(g.min() < env["env_Gmin"] - 1.0); days_lt70 += int(g.min() < 70)
                prev = sys_
            rows.append(dict(model=label, Patient=pid, f=f, f_tag=tag, variant=v, seed=sd,
                             cv_Gmin=gmin, cv_Gmax=gmax, cv_TIR=float(np.mean(tirs)),
                             cv_TIR_worst_day=float(np.min(tirs)),
                             env_Gmin=env["env_Gmin"], env_Gmax=env["env_Gmax"], env_TIR=env["env_TIR"],
                             nominal_converged=bool(ok_nom), days_below_env_min_1mgdl=below,
                             days_min_below_70=days_lt70))
    return rows

def job(args):
    model, key, envs = args
    if model == "hovorka":
        row = R.R3.H.PATIENTS_ADULT[key]
        base = R.R3.HovorkaDay(row, 1.0, "ramadan")
        mk, pid, label = (lambda f, c, s: R.HovS(row, f, c, s)), base.pid, "Hovorka"
    else:
        base = R.R3.UVADay(key, 1.0, "ramadan", 250.0)
        mk, pid, label = (lambda f, c, s: R.UvaS(key, f, c, s)), key, "UVA/Padova"
    rows = []
    for (f, tag), env in envs.items():
        rows += run_f(mk, base.y_ss(), f, label, pid, tag, env)
        if SMOKE:
            break
    return rows

def _low_priority():
    import os
    os.nice(15)

def _file(model, key):
    return PART / f"{model}_{str(key).replace('#', '_')}.csv"

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    which = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "all"
    PART.mkdir(exist_ok=True)
    r5 = pd.read_csv(OUT / "r5_switching.csv")
    # Key on the model too: both cohorts have patients named adult#001..#010 (bug fixed 2026-11-27;
    # the first run keyed on Patient only, corrected afterwards by r7b_fix_envelope.py).
    env_tab = r5.groupby(["model", "Patient", "f", "f_tag"])[["env_Gmin", "env_Gmax", "env_TIR"]].first()
    def envs_for(model_label, pid):
        d = env_tab.loc[(model_label, pid)]
        return {(float(f), tag): d.loc[(f, tag)].to_dict() for f, tag in d.index}
    items = []
    if which in ("hovorka", "all"):
        for i in range(10):
            items.append(("hovorka", i, envs_for("Hovorka", R.R3.HovorkaDay(R.R3.H.PATIENTS_ADULT[i], 1.0, "ramadan").pid)))
    if which in ("uva", "all"):
        items += [("uva", p, envs_for("UVA/Padova", p)) for p in R.R3.UVA_IDS]
    if SMOKE:
        items = [items[0], items[-1]] if which == "all" else items[:1]
        t0 = time.time()
        for it in items:
            rows = job(it); print(pd.DataFrame(rows).round(2).to_string(), f"\n{time.time()-t0:.0f}s", flush=True)
        sys.exit()
    todo = [it for it in items if not _file(it[0], it[1]).exists()]
    t0 = time.time()
    print(f"{len(items) - len(todo)} saved, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_low_priority) as ex:
        futs = {ex.submit(job, it): it for it in todo}
        for fu in as_completed(futs):
            it = futs[fu]
            pd.DataFrame(fu.result()).to_csv(_file(it[0], it[1]), index=False)
            print(f"{it[0]} {it[1]} saved ({time.time() - t0:.0f}s)", flush=True)
    pd.concat([pd.read_csv(_file(m, k)) for m, k, _ in items]).to_csv(OUT / "r7_correlated.csv", index=False)
    print("done", flush=True)
