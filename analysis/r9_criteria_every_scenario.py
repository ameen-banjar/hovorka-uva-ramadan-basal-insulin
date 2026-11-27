#!/usr/bin/env python3
"""MED-05 R9 (revision 1, post hoc, no new simulation): admissibility when TIR >= 70 % is
required in EVERY one of the nine scenarios instead of on the scenario mean (Reviewer 4,
major 3). Reads out/r4_<model>_scan.csv (grid 0.50:0.02:1.30). Also checks on the grid that
each admissible set is one connected run of grid points. Writes out/r9_every_scenario.csv."""
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
rows = []
for m in ("hovorka", "uva"):
    d = pd.read_csv(OUT / f"r4_{m}_scan.csv"); d["f"] = d.f.round(2)
    for p, g in d.groupby("Patient"):
        a = g.groupby("f").agg(Gmin=("Gmin", "min"), Gmax=("Gmax", "max"), TIRmean=("TIR", "mean"),
                               TIRmin=("TIR", "min"), conv=("converged", "all")).sort_index()
        for rule, tircol in (("mean", "TIRmean"), ("every", "TIRmin")):
            ok = (a.Gmin >= 80) & (a.Gmax <= 300) & (a[tircol] >= 70) & a.conv
            runs = int(((ok.astype(int).diff().fillna(ok.astype(int)) == 1)).sum())
            fs = a.index[ok]
            rows.append(dict(model=g.model.iloc[0], Patient=p, rule=rule, admissible=bool(ok.any()),
                             f_min_grid=fs.min() if len(fs) else np.nan, f_max_grid=fs.max() if len(fs) else np.nan,
                             connected_runs=runs))
r = pd.DataFrame(rows); r.to_csv(OUT / "r9_every_scenario.csv", index=False)
print(r.groupby(["model", "rule"]).agg(admissible=("admissible", "sum"), max_runs=("connected_runs", "max")))
print(r[r.admissible].pivot_table(index="Patient", columns="rule", values=["f_min_grid", "f_max_grid"]))
