#!/usr/bin/env python3
"""Correct the envelope columns of out/r7_correlated.csv (first run, 2026-11-27).

The first run looked up the fixed-scenario envelope of R5 by patient name only. Both cohorts have
patients named adult#001..#010, so (i) UVA/Padova adults received Hovorka envelopes at f = 1.0/0.9/0.8,
and (ii) Hovorka adult#002/#003/#007 were also run at the UVA/Padova interval midpoints.
The simulated quantities (cv_*) do not depend on the envelope and are unaffected.
This script: drops the Hovorka midpoint runs, re-attaches env_* from out/r5_switching.csv keyed on
(model, Patient, f, f_tag), and blanks days_below_env_min_1mgdl for the rows whose envelope was wrong
(that per-day count cannot be recomputed without re-running). Output: out/r7_correlated_fixed.csv.
"""
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
d = pd.read_csv(OUT / "r7_correlated.csv")
d = d[~((d.model == "Hovorka") & (d.f_tag == "midpoint"))].copy()
r5 = pd.read_csv(OUT / "r5_switching.csv")
env = r5.groupby(["model", "Patient", "f", "f_tag"])[["env_Gmin", "env_Gmax", "env_TIR"]].first().reset_index()
d["f_r"] = d.f.round(6); env["f_r"] = env.f.round(6)
old = d[["env_Gmin", "env_Gmax", "env_TIR"]].copy()
d = d.drop(columns=["env_Gmin", "env_Gmax", "env_TIR"]).merge(
    env.drop(columns="f"), on=["model", "Patient", "f_r", "f_tag"], how="left", validate="many_to_one")
assert d.env_Gmin.notna().all(), "missing envelope"
changed = ~np.isclose(old.env_Gmin.values, d.env_Gmin.values)
d.loc[changed, "days_below_env_min_1mgdl"] = np.nan
d = d.drop(columns="f_r")
d.to_csv(OUT / "r7_correlated_fixed.csv", index=False)
print(f"rows {len(d)}; envelope corrected in {changed.sum()} rows")
