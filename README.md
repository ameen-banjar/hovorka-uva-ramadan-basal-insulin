# Equilibrium, contraction and periodic entrainment of two type 1 diabetes models under Ramadan time-restricted feeding — code and outputs

Code and derived outputs for the article *Equilibrium, Contraction, and Periodic Entrainment of Two Type 1 Diabetes Models under Ramadan Time-Restricted Feeding: A Mathematical Analysis of Basal Insulin Reduction* (Ameen Banjar, University of Jeddah; submitted to *Mathematics*, MDPI).

## Contents

| Folder / file | What it holds |
|---|---|
| `code/hovorka_corrected.py` | Hovorka model with corrected parameters (insulin-action constants $k_{b,i}=S_i k_{a,i}$; meal grams→mmol; model-based ICR), 30-day Ramadan simulation of 10 virtual adults × 5 basal policies + pre-Ramadan |
| `results_corrected/` | Output of the script above: per-day CGM traces (60 files), `summary_all_patients_policies.csv`, `run.log` |
| `data/med01_archived_uva_run/` | Archived June 2026 UVA/Padova run of the code of an earlier study by the author (Banjar, *Front. Endocrinol.* 2026, 17, 1922149): `simglucose_ramadan.py`, `run.log`, summary CSV. Not the published run; used only to check the analysis (article §2.3). Kept unchanged, so it contains local paths of the original machine and is not meant to be rerun |
| `analysis/r1_r2_analysis.py` | Fasting equilibrium, elasticity, Jacobians, contraction; SymPy checks |
| `analysis/r3_periodic.py` | 24-h periodic orbits by shooting, Floquet multipliers, adaptation transients |
| `analysis/r4_safe_basal.py`, `analysis/r4_bounds.py`, `analysis/r4_refine.py` | Continuation scan over the basal rate under 9 fixed scenarios; admissible intervals (pre-specified criteria in the script header) and bisection refinement |
| `analysis/r5_switching.py` | Post hoc: day-to-day switching between the nine scenarios over 30 days (design fixed in the script header before the run) |
| `analysis/r6_robustness.py` | Post hoc: Hovorka elasticity over 10,000 parameter sets (closed form); admissible-interval counts under 12 alternative criteria; contraction bound vs computed settling |
| `analysis/simglucose_equivalence_check.py`, `analysis/floquet_step_check.py` | Checks: day map vs simglucose stepping; Floquet finite-difference step sensitivity |
| `analysis/make_figures.py`, `analysis/make_supplementary.py` | Figures 1–4 (`figures/`) and Supplementary Tables S1–S8 (`analysis/out/supplementary_tables.tex`) |
| `analysis/out/` | All derived outputs (CSV, text checks) |
| `figures/` | Figures 1–4 of the article (PDF and PNG) |

## Reproduce

Python 3.11 with `requirements.txt`. Run from the repository root, in this order. Steps 3 and 4 take about 10 and 30 minutes on an Apple M1 (3 low-priority worker processes; each patient is saved as it finishes and the run resumes from saved patients).

```bash
python3 -m pip install -r requirements.txt
```

1. Hovorka 30-day simulations (corrected model):
```bash
python3 code/hovorka_corrected.py
```

2. Equilibrium, elasticity, Jacobian and contraction, with SymPy checks:
```bash
python3 analysis/r1_r2_analysis.py
```

3. Periodic orbits, Floquet multipliers and transitions:
```bash
python3 analysis/r3_periodic.py all
```

4. Admissible-interval scan, bounds and bisection refinement:
```bash
python3 analysis/r4_safe_basal.py all && python3 analysis/r4_bounds.py && python3 analysis/r4_refine.py
```

5. Checks: equality of the UVA/Padova day map with simglucose's own stepping, and Floquet step sensitivity:
```bash
python3 analysis/simglucose_equivalence_check.py && python3 analysis/floquet_step_check.py
```

6. Post-hoc robustness: switching (about 1 h on an Apple M1, 3 low-priority workers, resumable) and the closed-form/criteria/contraction checks:
```bash
python3 analysis/r5_switching.py all && python3 analysis/r6_robustness.py
```

7. Figures 1–4 and Supplementary Tables S1–S8:
```bash
python3 analysis/make_figures.py && python3 analysis/make_supplementary.py
```

What is reused rather than recomputed: the archived UVA/Padova 30-day run in `data/med01_archived_uva_run/` (used only for comparison).

## Notes
- The Hovorka cohort and protocol derive from the code of the earlier study; an earlier implementation had insulin-action constants 15–110 times too large (article, Remark 1). All results here use the corrected implementation.
- The UVA/Padova model is used through the open-source `simglucose` package (J. Xie), which is a dependency and is not redistributed here.
- AI assistance (Claude Code, Anthropic; model Claude Opus 5.5, October 2026) is disclosed in the article (§2.5 and Acknowledgments).

## Citation
See `CITATION.cff`. Please cite the article and this repository (DOI from Zenodo).

## License
MIT (see `LICENSE`).
