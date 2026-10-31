#!/usr/bin/env python3
"""
Synthesize Candidate 19: Multi-Manifold Logit Simplex Synthesis (Top 10 Horizon).
Combines:
1. Primary Anchor: Sub 15 (Our Verified Peak 0.96226, 0 ties).
2. Deep Foundation Transformer: TabPFN-3.5 (Prior Labs Rank 8).
3. Continuous Neural Manifold: Cloud RepLeaf PLR SuperStack (5-fold OOF 0.962228).
4. Deep Neural Network: PyTabKit RealMLP (Gray Robertson & Demidov).
5. Fine-Grained Tree Foundation: High-Bin GPU XGBoost (Depth 10, k10).
6. Target Encoding Foundation: CatBoost GPU CTR.
7. Blending Domain: Logit Space z = log(p / (1 - p)) with Hierarchical Lexical Operator (eps=1e-7).
"""
import os
import sys
from pathlib import Path

ROOT = Path(r"E:\projects\playground-series-s6e10")
GP_DIR = ROOT / "datasets" / "goodpjw_selected"
SUB_DIR = ROOT / "submissions"
CLOUD_OUT = ROOT / "kaggle_repleaf_output"

TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def pct(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def to_logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def main():
    print("=" * 75)
    print("CANDIDATE 19: MULTI-MANIFOLD LOGIT SIMPLEX SYNTHESIS")
    print("=" * 75)

    # 1. Load Primary Anchor: Sub 15 (0.96226)
    sub15_path = SUB_DIR / "sub15_top5_hybrid_breakthrough.csv"
    assert sub15_path.exists(), f"Sub 15 not found at {sub15_path}"
    df15 = pd.read_csv(sub15_path)
    s15 = df15["satisfaction"].values
    N = len(s15)
    r15 = pct(s15)
    z15 = to_logit(r15)
    print(f"[Loaded] Sub 15 (Peak 0.96226): {N:,} rows, {len(np.unique(s15)):,} unique")

    # 2. Load Diverse Deep Foundation Arrays
    tabpfn_path = GP_DIR / "test_tabpfn_ne4.npy"
    repleaf_path = CLOUD_OUT / "test_cloud_repleaf_superstack.npy"
    realmlp_path = GP_DIR / "test_realmlp_v4fa_auxfull.npy"
    xgb_path = GP_DIR / "test_xgb_d10_k10.npy"
    cat_path = GP_DIR / "test_cat_ctr_k.npy"

    assert tabpfn_path.exists(), f"TabPFN missing at {tabpfn_path}"
    assert repleaf_path.exists(), f"RepLeaf missing at {repleaf_path}"
    assert realmlp_path.exists(), f"RealMLP missing at {realmlp_path}"
    assert xgb_path.exists(), f"XGBoost missing at {xgb_path}"
    assert cat_path.exists(), f"CatBoost missing at {cat_path}"

    z_tabpfn = to_logit(pct(np.load(tabpfn_path)))
    z_repleaf = to_logit(pct(np.load(repleaf_path)))
    z_realmlp = to_logit(pct(np.load(realmlp_path)))
    z_xgb = to_logit(pct(np.load(xgb_path)))
    z_cat = to_logit(pct(np.load(cat_path)))

    print("[Loaded] 5 Diverse Deep Foundation Arrays:")
    print(f"  TabPFN-3.5 Transformer: {len(z_tabpfn):,} rows")
    print(f"  RepLeaf PLR SuperStack: {len(z_repleaf):,} rows")
    print(f"  PyTabKit RealMLP:       {len(z_realmlp):,} rows")
    print(f"  High-Bin GPU XGBoost:   {len(z_xgb):,} rows")
    print(f"  CatBoost GPU CTR:       {len(z_cat):,} rows")

    # 3. Multi-Manifold Logit Simplex Weighting
    w_sub15   = 0.9400
    w_tabpfn  = 0.0210
    w_repleaf = 0.0180
    w_realmlp = 0.0090
    w_xgb     = 0.0060
    w_cat     = 0.0060

    total_w = w_sub15 + w_tabpfn + w_repleaf + w_realmlp + w_xgb + w_cat
    assert np.isclose(total_w, 1.0), f"Weights sum to {total_w}, expected 1.0"

    print("\n[Logit Simplex Weights]:")
    print(f"  Sub 15 Anchor:    {w_sub15:.4f} (94.0%)")
    print(f"  TabPFN-3.5:       {w_tabpfn:.4f} (2.1%)")
    print(f"  RepLeaf PLR:      {w_repleaf:.4f} (1.8%)")
    print(f"  PyTabKit RealMLP: {w_realmlp:.4f} (0.9%)")
    print(f"  High-Bin XGBoost: {w_xgb:.4f} (0.6%)")
    print(f"  CatBoost CTR:     {w_cat:.4f} (0.6%)")

    z_blend = (
        w_sub15 * z15
        + w_tabpfn * z_tabpfn
        + w_repleaf * z_repleaf
        + w_realmlp * z_realmlp
        + w_xgb * z_xgb
        + w_cat * z_cat
    )

    # 4. Strict Hierarchical Lexical Operator (eps = 1e-7)
    eps = 1e-7
    print(f"\n[Tie-Resolution] Applying Hierarchical Lexical Operator (eps={eps:.1e})...")
    p_perturbed = z_blend + eps * z_repleaf
    strict_ranks = np.argsort(np.argsort(p_perturbed)).astype(np.float64)
    p_cand19 = (strict_ranks + 0.5) / float(N)

    # 5. Rigorous Mathematical Audit
    n_unique_19 = len(np.unique(p_cand19))
    n_ties_19 = N - n_unique_19
    delta_r = p_cand19 - r15
    std_delta = np.std(delta_r)
    max_delta = np.max(np.abs(delta_r))
    rho_15 = spearmanr(p_cand19, r15).statistic

    print("\n" + "=" * 70)
    print("MATHEMATICAL INVARIANT AUDIT REPORT: CANDIDATE 19")
    print("=" * 70)
    print(f"Total Rows Evaluated:           {N:,}")
    print(f"Total Unique Values:            {n_unique_19:,}")
    print(f"Tied Pairs:                     {n_ties_19} (Exact Target: 0)")
    print(f"Rank displacement std vs Sub 15:{std_delta:.6f} (Target: 0.0020 - 0.0030)")
    print(f"Max absolute delta vs Sub 15:   {max_delta:.6f}")
    print(f"Spearman correlation vs Sub 15: {rho_15:.7f} (Target: 0.99994 - 0.99997)")
    print(f"Min / Median / Max Prediction:  {p_cand19.min():.9f} / {np.median(p_cand19):.9f} / {p_cand19.max():.9f}")
    print("=" * 70)

    # Invariant Assertions
    assert n_ties_19 == 0, f"FAILED: {n_ties_19} ties found!"
    assert 0.0018 <= std_delta <= 0.0035, f"FAILED: std={std_delta:.6f} outside target bounds [0.0018, 0.0035]"
    assert rho_15 >= 0.99990, f"FAILED: rho={rho_15:.7f} < 0.99990"
    assert not np.isnan(p_cand19).any(), "FAILED: NaNs found!"
    assert not np.isinf(p_cand19).any(), "FAILED: Infs found!"
    assert p_cand19.min() > 0.0 and p_cand19.max() < 1.0, "FAILED: Range out of bounds!"

    print(">> ALL MATHEMATICAL INVARIANTS SATISFIED & SIGNED OFF.")

    # 6. Save Candidate 19 Submission File
    out_file = SUB_DIR / "sub19_multimanifold_logit_simplex.csv"
    out_df = pd.DataFrame({
        "id": df15["id"].values,
        "satisfaction": p_cand19.astype(np.float64)
    })
    out_df.to_csv(out_file, index=False, float_format="%.9f")

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Saved Successfully] {out_file} ({file_size_mb:.2f} MB, {len(out_df):,} rows)")
    print("Candidate 19 is ready for live Kaggle submission.")


if __name__ == "__main__":
    main()
