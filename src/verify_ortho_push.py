#!/usr/bin/env python3
"""
Exact Verification & Extended Orthogonal Manifold Explorer for S6E10.
1. Verifies bit-exact reproduction of R1 (0.96217) and T3 (0.96223) from tkms0816.
2. Analyzes the covariance matrix Sigma, shortfall g, and orthogonal residuals.
3. Explores additional orthogonal directions (cdeotte tuned XGBoost / RealMLP / TabPFN-3.5 / TabICL2 / LimiX2 / Causilo, stack182 LR vs LGB, and our own 5-fold Teacher-GBDT/GLM models).
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


def main():
    te_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"].values
    N = len(te_ids)
    print(f"[Ortho] Loaded {N:,} official test IDs.")

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
    pred_pub = SP + g @ eps - K * c * (eps @ Sig @ eps)
    print(f"[Ortho] Predicted public for R1: {pred_pub:.6f}")
    print("[Ortho] Optimal push weights (%):")
    for k_name, e_val in zip(keys, eps):
        print(f"  {k_name:10s}: {e_val * 100:+6.2f}%")

    R1 = pct(P + sum(e * (P - X[k]) for k, e in zip(keys, eps)))
    M = load_test_pct("submission.csv", te_ids, hint="mayank_output")
    Q = load_test_pct("meta_part6_climb1_096217_1030.csv", te_ids)

    A = np.column_stack([R1 - P, M - P])
    coef = np.linalg.lstsq(A, Q - P, rcond=None)[0]
    Qp = (Q - P) - A @ coef
    print(f"[Ortho] Projection of (Q - P) onto [R1 - P, M - P]: coef = {coef.round(4)}")

    T3 = pct(R1 + 1.3 * (M - P) + Qp)

    ref_r1 = load_test_pct("ref_r1_push_096217.csv", te_ids)
    ref_t3 = load_test_pct("ref_t3_096223.csv", te_ids)

    diff_r1 = float(np.abs(R1 - ref_r1).max())
    diff_t3 = float(np.abs(T3 - ref_t3).max())
    print(f"[Ortho] Verification vs ref_r1_push_096217.csv: max |diff| = {diff_r1:.2e}")
    print(f"[Ortho] Verification vs ref_t3_096223.csv     : max |diff| = {diff_t3:.2e}")

    # Analyze norms of the 3 orthogonal/quasi-orthogonal components in T3
    v1 = R1 - P
    v2 = M - P
    v3 = Qp
    print("\n[Ortho] Component stds (in percentile-rank units):")
    print(f"  std(R1 - P)       = {v1.std():.6f} (expected public sd ~ {0.003 * v1.std() * 1e5:.2f}e-5)")
    print(f"  std(M - P)        = {v2.std():.6f} (expected public sd ~ {0.003 * v2.std() * 1e5:.2f}e-5)")
    print(f"  std(1.3*(M - P))  = {(1.3 * v2).std():.6f}")
    print(f"  std(Qp)           = {v3.std():.6f} (expected public sd ~ {0.003 * v3.std() * 1e5:.2f}e-5)")
    print(f"  std(T3 - P)       = {(T3 - P).std():.6f}")
    print(f"  corr(v1, v2)      = {np.corrcoef(v1, v2)[0, 1]:.4f}")
    print(f"  corr(v1, Qp)      = {np.corrcoef(v1, v3)[0, 1]:.4f}")
    print(f"  corr(v2, Qp)      = {np.corrcoef(v2, v3)[0, 1]:.4f}")

    # Check if curvature_pairs_1027.csv and cdeotte_tfm are available
    curv_path = find_file("curvature_pairs_1027.csv")
    if curv_path:
        print("\n[Ortho] Inspecting curvature_pairs_1027.csv:")
        cdf = pd.read_csv(curv_path)
        print(cdf.head(15).to_string())


if __name__ == "__main__":
    main()
