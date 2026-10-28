#!/usr/bin/env python3
"""
MED-05 R1/R2: fasting equilibrium, basal elasticity and stability of the
Hovorka (corrected parameters) and UVA/Padova (simglucose) models.

Outputs (analysis/out/):
  hovorka_r1r2.csv, uva_r1r2.csv, link_to_simulation.csv, sympy_checks.txt
"""
import sys, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, sympy as sp
from scipy.optimize import brentq, root
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "code"))
import hovorka_corrected as H

POL = [1.0, 0.9, 0.8, 0.7, 0.6]
MGDL = 18.015

# ---------------------------------------------------------------- SymPy
def sympy_checks():
    I, EGP0, F01, SIT, SID, SIE, k12, VG = sp.symbols("I EGP0 F01 S_IT S_ID S_IE k12 V_G", positive=True)
    x1, x2, x3 = SIT * I, SID * I, SIE * I
    G = sp.Symbol("G", positive=True)
    Q1 = G * VG                          # per kg
    Q2 = x1 * Q1 / (k12 + x2)
    res = -(F01 + x1 * Q1) + k12 * Q2 + EGP0 * (1 - x3)
    Gstar = sp.solve(sp.Eq(res, 0), G)[0]
    closed = (EGP0 * (1 - SIE * I) - F01) * (k12 + SID * I) / (VG * SIT * SID * I**2)
    eps = sp.simplify(sp.diff(sp.log(closed), I) * I)
    eps_claim = -2 + SID * I / (k12 + SID * I) - SIE * I * EGP0 / (EGP0 * (1 - SIE * I) - F01)
    A = sp.Matrix([[-x1, k12], [x1, -(k12 + x2)]])
    lines = [
        f"G* from solve == closed form: {sp.simplify(Gstar - closed) == 0}",
        f"elasticity == claimed formula: {sp.simplify(eps - eps_claim) == 0}",
        f"trace(A) = {sp.simplify(A.trace())}",
        f"det(A)   = {sp.factor(A.det())}",
        f"discriminant - [(x1-x2)^2 + k12^2 + 2 k12 (x1+x2)] = "
        f"{sp.simplify(A.trace()**2 - 4*A.det() - ((x1-x2)**2 + k12**2 + 2*k12*(x1+x2)))}",
    ]
    # Weighted-l1 contraction: weights (1+d, 1); column measures
    d = sp.Symbol("delta", positive=True)
    W = sp.diag(1 + d, 1)
    M = W * A * W.inv()
    col = [sp.simplify(M[j, j] + sum(abs(M[i, j]) for i in range(2) if i != j)) for j in range(2)]
    lines.append(f"weighted-l1 column measures (need <0): {col}")
    (OUT / "sympy_checks.txt").write_text("\n".join(lines) + "\n")
    return lines

# ---------------------------------------------------------------- Hovorka
def hov_equilibrium(p, u):
    """General fasting equilibrium (includes F01c below 4.5 and renal above 9)."""
    BW = p["BW"]; I = u / (p["ke"] * p["VI"] * BW)
    x1, x2, x3 = (p[f"kb{i}"] / p[f"ka{i}"] * I for i in (1, 2, 3))
    rhs = max(p["EGP0"] * BW * (1 - x3), 0.0)
    def lhs(G):
        F01c = p["F01"] * BW * (1 if G >= 4.5 else G / 4.5)
        FR = 0.003 * (G - 9.0) * p["VG"] * BW if G > 9 else 0.0
        return x1 * x2 / (p["k12"] + x2) * G * p["VG"] * BW + F01c + FR - rhs
    G = brentq(lhs, 1e-6, 200.0)
    return G, I, (x1, x2, x3)

def hov_closed(p, I):
    S = [p[f"kb{i}"] / p[f"ka{i}"] for i in (1, 2, 3)]
    return (p["EGP0"] * (1 - S[2] * I) - p["F01"]) * (p["k12"] + S[1] * I) / (p["VG"] * S[0] * S[1] * I**2)

def hov_d(p, G):
    """d(G) = d(F01c + FR)/dQ1 at glucose G (mmol/L): F01/(4.5 VG) below 4.5, 0.003 above 9."""
    return (p["F01"] / (4.5 * p["VG"]) if G < 4.5 else 0.0) + (0.003 if G > 9.0 else 0.0)

def hov_eps_terms(p, I):
    S = [p[f"kb{i}"] / p[f"ka{i}"] for i in (1, 2, 3)]
    return (-2.0, S[1] * I / (p["k12"] + S[1] * I),
            -S[2] * I * p["EGP0"] / (p["EGP0"] * (1 - S[2] * I) - p["F01"]))

def hov_eps(p, I):
    S = [p[f"kb{i}"] / p[f"ka{i}"] for i in (1, 2, 3)]
    return -2 + S[1] * I / (p["k12"] + S[1] * I) - S[2] * I * p["EGP0"] / (p["EGP0"] * (1 - S[2] * I) - p["F01"])

def num_jac(f, y, h=1e-6):
    y = np.asarray(y, float); f0 = np.asarray(f(y)); J = np.zeros((len(f0), len(y)))
    for j in range(len(y)):
        e = np.zeros_like(y); e[j] = h * max(1.0, abs(y[j]))
        J[:, j] = (np.asarray(f(y + e)) - np.asarray(f(y - e))) / (2 * e[j])
    return J

def hovorka_table():
    rows = []
    for r in H.PATIENTS_ADULT:
        p = H.make_params(r); u_b = H.find_basal(p)
        G1, I1, (x1, x2, x3) = hov_equilibrium(p, u_b)
        row = dict(Patient=p["pid"], isf_mult=p["isf_mult"], basal_U_day=u_b * 1.44,
                   I_ss=I1, x3=x3, G_ss_mgdl=G1 * MGDL,
                   closed_form_err=abs(hov_closed(p, I1) - G1), eps=hov_eps(p, I1))
        t0, t1, t2 = hov_eps_terms(p, I1)
        row.update(eps_term_const=t0, eps_term_disposal=t1, eps_term_egp=t2)
        for f in POL[1:]:
            row[f"Gstar_{int(f*100)}_mgdl"] = hov_equilibrium(p, f * u_b)[0] * MGDL
        u_max = p["ke"] * p["VI"] * p["BW"] * (1 - p["F01"] / p["EGP0"]) / (p["kb3"] / p["ka3"])
        row["u_max_U_day"] = u_max * 1.44
        row["G_max_zero_insulin_mgdl"] = (9 + (p["EGP0"] - p["F01"]) / (0.003 * p["VG"])) * MGDL
        # Glucose block and full Jacobian (no meals, constant basal)
        A = np.array([[-x1 - hov_d(p, G1), p["k12"]], [x1, -(p["k12"] + x2)]])
        lam = np.sort(np.linalg.eigvals(A).real)
        y0 = H.ss_state(u_b, p)
        J = num_jac(lambda y: H.hovorka_ode(0.0, y, p, u_b, [], []), y0)
        lam_full = np.linalg.eigvals(J)
        row.update(lam_glu_fast=lam[0], lam_glu_slow=lam[1], tau_slow_h=-1 / lam[1] / 60,
                   lam_full_max_real=lam_full.real.max(), lam_full_max_imag=abs(lam_full.imag).max(),
                   tau_slow_h_at_80=None)
        # slow mode at 80% basal
        G8, _, (y1, y2, _) = hov_equilibrium(p, 0.8 * u_b)
        # full glucose block, including d(G) (renal excretion is active when G* > 9 mmol/L)
        A8 = np.array([[-y1 - hov_d(p, G8), p["k12"]], [y1, -(p["k12"] + y2)]])
        row["G8_mmol"] = G8
        row["d_at_80"] = hov_d(p, G8)
        A8b = np.array([[-y1, p["k12"]], [y1, -(p["k12"] + y2)]])          # band-only (old) value
        row["tau_slow_h_at_80_bandonly"] = -1 / np.sort(np.linalg.eigvals(A8b).real)[1] / 60
        row["tau_slow_h_at_80"] = -1 / np.sort(np.linalg.eigvals(A8).real)[1] / 60
        delta = 0.5 * x2 / p["k12"]
        row["contraction_rate_per_h"] = min(delta * x1 / (1 + delta), x2 - delta * p["k12"]) * 60
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- UVA/Padova
def uva_table():
    from simglucose.patient.t1dpatient import T1DPatient, Action
    import simglucose
    P = pd.read_csv(Path(simglucose.__file__).parent / "params" / "vpatient_params.csv")
    rows = []
    for _, q in P.iterrows():
        basal = q.u2ss * q.BW / 6000.0
        def F(x, f):
            return T1DPatient.model(0, x, Action(CHO=0, insulin=f * basal), q, 0.0, 0.0)
        x0 = q[[f"x0_{i:2d}" for i in range(1, 14)]].values.astype(float)
        eq = {}
        for f in POL:
            sol = root(lambda x: F(x, f), x0, method="hybr", tol=1e-12)
            eq[f] = sol.x if sol.success else np.full(13, np.nan)
        x1 = eq[1.0]
        Gp = lambda x: x[3] / q.Vg * 1.0   # mg/dL
        # elasticity d ln Gp / d ln f at f=1 (central difference)
        h = 1e-4
        xp = root(lambda x: F(x, 1 + h), x1, method="hybr", tol=1e-12).x
        xm = root(lambda x: F(x, 1 - h), x1, method="hybr", tol=1e-12).x
        eps = (np.log(xp[3]) - np.log(xm[3])) / (np.log(1 + h) - np.log(1 - h))
        J = num_jac(lambda x: F(x, 1.0), x1)
        lam = np.linalg.eigvals(J)
        Vmt = q.Vm0 + q.Vmx * x1[6]
        Up = Vmt * q.Km0 / (q.Km0 + x1[4])**2
        Ep = q.ke1 if x1[3] > q.ke2 else 0.0
        Ag = np.array([[-(q.kp2 + q.k1 + Ep), q.k2], [q.k1, -(q.k2 + Up)]])
        lg = np.sort(np.linalg.eigvals(Ag).real)
        row = dict(Patient=q.Name, G_ss_mgdl=Gp(x1), Gb_param=q.Gb, eps=eps,
                   kp2=q.kp2, U_prime=Up, lam_glu_slow=lg[1], tau_slow_h=-1 / lg[1] / 60,
                   lam_full_max_real=lam.real.max(),
                   contraction_rate_per_h=min(q.kp2 + Ep, Up) * 60,
                   hovorka_like_det_share=(q.k1 * Up) / ((q.kp2 + Ep) * (q.k2 + Up) + q.k1 * Up))
        for f in POL[1:]:
            row[f"Gstar_{int(f*100)}_mgdl"] = Gp(eq[f])
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- link to simulations
def link(hov, uva):
    hs = pd.read_csv(HERE.parent / "results_corrected" / "summary_all_patients_policies.csv")
    us = pd.read_csv(HERE.parent / "data" / "med01_archived_uva_run" / "summary_all_patients_policies.csv")
    out = []
    for name, eqt, sim in (("Hovorka", hov, hs), ("UVA/Padova", uva, us)):
        r = sim[sim.Period == "Ramadan"]
        piv = r.pivot_table(index="Patient", columns="Policy", values="TIR")
        mc = r.pivot_table(index="Patient", columns="Policy", values="MeanCGM")
        d = eqt.set_index("Patient")
        common = d.index.intersection(piv.index)
        for f in POL[1:]:
            dG = d.loc[common, f"Gstar_{int(f*100)}_mgdl"] - d.loc[common, "G_ss_mgdl"]
            dTIR = piv.loc[common, f] - piv.loc[common, 1.0]
            dMC = mc.loc[common, f] - mc.loc[common, 1.0]
            out.append(dict(model=name, policy=f, n=len(common),
                            mean_dGstar_mgdl=dG.mean(), mean_dMeanCGM_sim=dMC.mean(),
                            mean_dTIR_sim=dTIR.mean(),
                            spearman_dGstar_vs_dTIR=spearmanr(dG, dTIR)[0],
                            spearman_dGstar_vs_dMeanCGM=spearmanr(dG, dMC)[0]))
    return pd.DataFrame(out)

if __name__ == "__main__":
    for l in sympy_checks(): print(l)
    hov = hovorka_table(); hov.to_csv(OUT / "hovorka_r1r2.csv", index=False)
    uva = uva_table(); uva.to_csv(OUT / "uva_r1r2.csv", index=False)
    lk = link(hov, uva); lk.to_csv(OUT / "link_to_simulation.csv", index=False)
    pd.set_option("display.width", 250)
    print(hov.round(4).to_string()); print(uva.round(4).describe().T.to_string()); print(lk.round(3).to_string())
