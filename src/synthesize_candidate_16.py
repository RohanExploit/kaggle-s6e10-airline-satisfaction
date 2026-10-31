#!/usr/bin/env python3
"""
Synthesize Candidate 16: Dual-Summit Zero-Tie Consensus (Top 10 Summit Push).
Architectural Design & Invariants Audited and Signed off by ml-architect-auditor.
Targets: Live Kaggle Leaderboard Rank 10 Cutoff (0.96238, srikalyan).

Inputs:
1. Sub 15 (Our Verified LB 0.96226 peak, ref 57844876)
2. Dennis V10 (Verified LB 0.96227 summit, Rank 16-17)
3. Dennis Orthogonal Direction d (old_direction.npz, std 8.82e-5, score-bin orthogonal)
4. Cloud RepLeaf PLR Super-Stack (OOF AUC 0.962228, continuous hyperplane tie-breaker)
5. Gaussian Disagreement Gate: W(p) = exp(-(p - 0.5)^2 / (2 * 0.18^2))
6. Hierarchical Lexical Operator: argsort(argsort(P_blend + 1e-7 * P_repleaf)) -> 0 ties
"""
import os
import sys
from pathlib import Path

ROOT = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
DENNIS_DIR = ROOT / "notebooks_analysis" / "dennis_dataset" / "XWC_S6E10_V10_MaskedSignal_v11" / "XWC_S6E10_V10_MaskedSignal_v11" / "known_best"
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


def main():
    print("=" * 75)
    print("CANDIDATE 16: DUAL-SUMMIT ZERO-TIE CONSENSUS (TOP 10 SUMMIT PUSH)")
    print("=" * 75)

    # 1. Load Sub 15 (Our Peak 0.96226)
    sub15_path = SUB_DIR / "sub15_top5_hybrid_breakthrough.csv"
    assert sub15_path.exists(), f"Sub 15 not found at {sub15_path}"
    df15 = pd.read_csv(sub15_path)
    s15 = df15["satisfaction"].values
    N = len(s15)
    r15 = pct(s15)
    print(f"[Loaded] Sub 15 (Our Peak 0.96226): {N} rows, {len(np.unique(s15))} unique values")

    # 2. Load Dennis V10 (Peak 0.96227)
    v10_path = DENNIS_DIR / "v10.csv"
    assert v10_path.exists(), f"Dennis V10 not found at {v10_path}"
    dfv10 = pd.read_csv(v10_path)
    sv10 = dfv10["satisfaction"].values
    rv10 = pct(sv10)
    ties_v10 = N - len(np.unique(sv10))
    print(f"[Loaded] Dennis V10 (Peak 0.96227): {N} rows, {len(np.unique(sv10))} unique values ({ties_v10} tied instances)")

    # 3. Load Dennis Orthogonal Direction d
    dir_path = DENNIS_DIR / "old_direction.npz"
    assert dir_path.exists(), f"old_direction.npz not found at {dir_path}"
    dir_data = np.load(dir_path)
    d_raw = dir_data["direction"].astype(np.float64)
    print(f"[Loaded] Dennis Direction d: shape={d_raw.shape}, std={np.std(d_raw):.8f}, min/max=[{d_raw.min():.8f}, {d_raw.max():.8f}]")

    # 4. Load Cloud RepLeaf Super-Stack for Tie-Breaking
    repleaf_path = CLOUD_OUT / "test_cloud_repleaf_superstack.npy"
    if not repleaf_path.exists():
        repleaf_path = CLOUD_OUT / "submission_cloud_repleaf_superstack.csv"
        df_rl = pd.read_csv(repleaf_path)
        s_rl = df_rl["satisfaction"].values
    else:
        s_rl = np.load(repleaf_path)
    r_rl = pct(s_rl)
    print(f"[Loaded] Cloud RepLeaf Super-Stack: shape={s_rl.shape}, unique={len(np.unique(s_rl))}")

    # Verify ID alignment
    assert (df15["id"].values == dfv10["id"].values).all(), "ID mismatch between Sub 15 and Dennis V10!"
    if "id" in dir_data:
        assert (df15["id"].values == dir_data["id"]).all(), "ID mismatch in direction file!"

    # 5. Gaussian Disagreement Gate W(p) with sigma = 0.18
    sigma_gate = 0.18
    p_center = 0.50 * r15 + 0.50 * rv10
    W = np.exp(-((p_center - 0.5) ** 2) / (2.0 * (sigma_gate ** 2)))
    print(f"[Gating] Gaussian Disagreement Gate: mean={W.mean():.4f}, median={np.median(W):.4f}, min={W.min():.4f}, max={W.max():.4f}")

    # 6. Dual-Summit Convex Combination + Gated Directional Nudge
    scale_d = 0.85
    p_blend = 0.50 * r15 + 0.50 * rv10 + W * (scale_d * d_raw)

    # 7. Hierarchical Lexical Operator for 100% Zero-Tie Resolution
    eps = 1e-7
    print(f"[Tie-Resolution] Applying Hierarchical Lexical Operator (eps={eps:.1e})...")
    p_perturbed = p_blend + eps * r_rl
    strict_ranks = np.argsort(np.argsort(p_perturbed)).astype(np.float64)
    p_cand16 = (strict_ranks + 0.5) / float(N)

    # 8. Rigorous Mathematical Audit
    n_unique = len(np.unique(p_cand16))
    n_ties = N - n_unique
    delta_r15 = p_cand16 - r15
    delta_rv10 = p_cand16 - rv10
    std_15 = np.std(delta_r15)
    std_v10 = np.std(delta_rv10)
    rho_15 = spearmanr(p_cand16, r15).statistic
    rho_v10 = spearmanr(p_cand16, rv10).statistic

    print("\n" + "=" * 70)
    print("MATHEMATICAL INVARIANT AUDIT REPORT: CANDIDATE 16")
    print("=" * 70)
    print(f"Total Rows Evaluated:           {N:,}")
    print(f"Total Unique Values:            {n_unique:,}")
    print(f"Tied Pairs:                     {n_ties} (Exact Target: 0)")
    print(f"Rank displacement std vs Sub 15:{std_15:.6f}")
    print(f"Rank displacement std vs V10:   {std_v10:.6f}")
    print(f"Spearman correlation vs Sub 15: {rho_15:.7f}")
    print(f"Spearman correlation vs V10:    {rho_v10:.7f}")
    print(f"Max absolute delta vs Sub 15:   {np.max(np.abs(delta_r15)):.6f}")
    print(f"Min / Median / Max Prediction:  {p_cand16.min():.9f} / {np.median(p_cand16):.9f} / {p_cand16.max():.9f}")
    print("=" * 70)

    # Invariant Assertions
    assert n_ties == 0, f"FAILED: Found {n_ties} tied ranks!"
    assert rho_15 >= 0.99950, f"FAILED: Spearman correlation vs Sub 15 {rho_15:.7f} < 0.99950"
    assert rho_v10 >= 0.99950, f"FAILED: Spearman correlation vs V10 {rho_v10:.7f} < 0.99950"
    assert not np.isnan(p_cand16).any(), "FAILED: Contains NaNs!"
    assert not np.isinf(p_cand16).any(), "FAILED: Contains Infs!"
    assert p_cand16.min() > 0.0 and p_cand16.max() < 1.0, "FAILED: Range out of bounds!"

    print(">> ALL MATHEMATICAL INVARIANTS SATISFIED & SIGNED OFF.")

    # 9. Save Candidate 16 Submission File
    out_file = SUB_DIR / "sub16_dual_summit_zero_tie_consensus.csv"
    out_df = pd.DataFrame({
        "id": df15["id"].values,
        "satisfaction": p_cand16.astype(np.float64)
    })
    out_df.to_csv(out_file, index=False, float_format="%.9f")

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Saved Successfully] {out_file} ({file_size_mb:.2f} MB, {len(out_df):,} rows)")
    print("Candidate 16 is ready for immediate live submission.")


if __name__ == "__main__":
    main()
