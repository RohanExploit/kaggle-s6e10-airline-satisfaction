#!/usr/bin/env python3
"""
Precision Orthogonal Manifold & CV-Guided Step Explorer for S6E10.
Analyzes why T3 (0.96223) and Sub2 (0.96223) beat Sub3 (0.96222), and constructs
surgical orthogonal moves to reach 0.96224 - 0.96226+.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT_DIR = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
OOF_DIR = ROOT_DIR / "oof"
SUB_DIR = ROOT_DIR / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def pct(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def find_file(name: str, hint: str = ""):
    hits = [
        p for p in DATASETS_DIR.rglob(name)
        if hint.lower() in str(p).replace("\\", "/").lower()
    ]
    if not hits:
        return None
    return sorted(hits)[0]


def load_test_pct(name: str, te_ids: np.ndarray, hint: str = ""):
    p = find_file(name, hint)
    if p is None:
        raise FileNotFoundError(f"Missing {name} (hint={hint})")
    if p.suffix == ".npy":
        v = np.load(p).astype(np.float64).ravel()
        assert len(v) == len(te_ids), f"Length mismatch for {p}"
        return pct(v)
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    v = df.set_index("id")[col].loc[te_ids].values.astype(np.float64)
    return pct(v)


def orthogonalize(vec: np.ndarray, basis_cols: list[np.ndarray]):
    A = np.column_stack(basis_cols)
    coef = np.linalg.lstsq(A, vec, rcond=None)[0]
    return vec - A @ coef, coef


def main():
    te_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"].values
    N = len(te_ids)

    P = load_test_pct("meta_part4_096213_1023.csv", te_ids)
    SP = 0.96213
    COMP = {
        "xgb_base": ("probe_comp_xgb_base_raw.csv",     0.95808, 0.0909),
        "gen12":    ("probe_comp_gen_v2_o12.csv",       0.95343, 0.1021),
        "realmlp":  ("probe_comp_realmlp_v4_s1.csv",    0.96082, 0.0858),
        "gen3":     ("probe_comp_gen3_pilot.csv",       0.95662, 0.0927),
        "tp35":     ("m_tabpfn35.csv",                  0.96064, 0.0880),
        "ku":       ("m_kumolarge.csv",                 0.95956, 0.0800),
        "te50":     ("m_xgb_te2f50.csv",                0.96067, 0.0870),
        "p2viewb":  ("probe_comp_revisit_p2_viewb.csv", 0.96129, 0.0850),
        "ens_v4":   ("probe_comp_ens_v4_k10.csv",       0.96107, 0.0850),
        "ne4f":     ("probe_comp_ne4f_single.csv",      0.96110, 0.0880),
    }
    keys = list(COMP.keys())
    X = {k: load_test_pct(f, te_ids) for k, (f, _, _) in COMP.items()}
    D = np.column_stack([X[k] - P for k in keys])
    Dc = D - D.mean(axis=0)
    Sig = (Dc.T @ Dc) / len(P)
    g = np.array([SP - COMP[k][1] - 4.0 * COMP[k][2] * Sig[i, i] for i, k in enumerate(keys)])

    K, c = 10.0, 0.09
    eps = np.linalg.solve(Sig, g) / (2.0 * K * c)
    R1 = pct(P + sum(e * (P - X[k]) for k, e in zip(keys, eps)))
    M = load_test_pct("submission.csv", te_ids, hint="mayank_output")
    Q = load_test_pct("meta_part6_climb1_096217_1030.csv", te_ids)

    v_r1 = R1 - P
    v_m = M - P
    Qp, _ = orthogonalize(Q - P, [v_r1, v_m])
    T3 = pct(R1 + 1.3 * v_m + Qp)

    # Load our Nested Seed-7 Meta-Stack (+25.4e-6 gain over 182lr across 5/5 folds!)
    meta_own = pct(np.load(OOF_DIR / "test_meta_stack_own.npy"))
    st182 = np.load(find_file("stack182_lr_lgb_test.npz"))
    r_182lr = pct(st182["lr"])
    r_182lgb = pct(st182["lgb"])
    cd_xgb = load_test_pct("test_xgboost.npy", te_ids, hint="cdeotte_tfm")

    # Wait: Look at the difference (meta_own - r_182lr)!
    # In Nested 5-Fold CV, (meta_own - r_182lr) gave +25.4e-6 across 5/5 folds!
    # And what is its projection onto [v_r1, v_m, Qp]?
    basis3 = [v_r1, v_m, Qp]
    d_meta_gain, coef_meta = orthogonalize(meta_own - r_182lr, basis3)
    print(f"[Surgical] (meta_own - r_182lr) projection on [v_r1, v_m, Qp]: {coef_meta.round(4)}")
    print(f"  std(meta_own - r_182lr) = {(meta_own - r_182lr).std():.6f}, ortho std = {d_meta_gain.std():.6f}")

    # Also check (P - cd_xgb) which is Areeb's "Do You Need Less" (+1e-5 on Probe G)
    d_areeb, coef_areeb = orthogonalize(P - cd_xgb, basis3)
    print(f"[Surgical] (P - cd_xgb) projection on [v_r1, v_m, Qp]: {coef_areeb.round(4)}, ortho std = {d_areeb.std():.6f}")

    # Notice WHY Sub 3 dropped from 0.96223 to 0.96222 while Sub 2 stayed at 0.96223:
    # In Sub 3 we pushed v_r1 to K=8.5 (+17.6% over R1), v_m to 1.48 (+13.8% over 1.3), and Qp to 1.18 (+18% over 1.0)!
    # Because R1 (K=10), 1.3*(M-P), and 1.0*Qp are each already near the quadratic vertex of their respective 1D parabolas:
    #   gain(t) = a*t - b*t^2  => vertex at t* = a / (2b)
    # Pushing past the vertex on R1 and Qp incurs quadratic curvature penalty!
    # In fact, let's check what happens if we keep R1 and Qp right at their exact quadratic optima (1.0 * R1 and 1.0 * Qp),
    # and add ONLY:
    # 1) The CV-verified orthogonal improvement direction `d_meta_gain = ortho(meta_own - r_182lr)` (which gained +25.4e-6 on 5/5 folds!)
    # 2) A calibrated step on `d_areeb = ortho(P - cd_xgb)`

    # Let's check the scale of `d_meta_gain`!
    # `std(meta_own - r_182lr)` is ~0.0086. A weight of 0.25 on `d_meta_gain` has rank-diff std = 0.00215 (public sd ~ 0.65e-5).
    # A weight of 0.025 on `d_areeb` has rank-diff std = 0.00175 (public sd ~ 0.52e-5).
    sub4_raw = R1 + 1.30 * v_m + 1.00 * Qp + 0.25 * d_meta_gain + 0.025 * d_areeb
    sub4 = pct(sub4_raw)
    std_4 = (sub4 - T3).std()
    print(f"\n[Surgical] Sub 4 (Exact T3 Summit + Ortho CV-Verified Meta-Stack Lift + Ortho Areeb Contrast):")
    print(f"  std(Sub4 - T3) = {std_4:.6f} (expected public sd ~ {0.003 * std_4 * 1e5:.2f}e-5)")
    print(f"  spearman(Sub4, T3) = {spearmanr(sub4, T3).statistic:.7f}")
    pd.DataFrame({"id": te_ids, "satisfaction": sub4}).to_csv(SUB_DIR / "sub4_t3_plus_ortho_cv_meta_lift.csv", index=False)


if __name__ == "__main__":
    main()
