#!/usr/bin/env python3
"""
Frontier Reasoning Model Protocol Execution (Kaggle S6E10 Top 10 Summit).
Implements the exact mathematical specification provided by the user:

1. Ascent Vector Extrapolation with Localized Variance Gate:
   P_extrap = P_15 + alpha * v_climb * exp(-(P_15 - 0.5)^2 / (2 * sigma^2))
   alpha = 12.0, sigma = 0.12

2. Orthogonal Teacher Projection:
   T_raw = 0.65 * rank(T_lgb) + 0.35 * rank(T_plr)
   v_c = v_climb / ||v_climb||_2, p_15 = P_15 / ||P_15||_2
   T_perp = T_raw - <T_raw, v_c> v_c - <T_raw, p_15> p_15
   T_ortho = T_perp / ||T_perp||_2
   Stored as immutable numpy array: oof/T_ortho.npy

3. 0.96233 Consensus Masked Differential Extraction:
   P_33 = 0.45 * rank(RealMLP) + 0.55 * rank(LGB_ogp + XGB_d8)
   Delta_33 = P_33 - P_15
   Delta_masked = Delta_33 * (|Delta_33| > q_0.95)
   Stored as immutable numpy array: oof/Delta_masked.npy

4. Hierarchical Lexical Operator for Guaranteed Zero Ties:
   P_final = (argsort(argsort(P_extrap + 1e-7 * P_superstack)) + 0.5) / N
"""
import os
import sys
from pathlib import Path

ROOT = Path(r"E:\projects\playground-series-s6e10")
SUB_DIR = ROOT / "submissions"
OOF_DIR = ROOT / "oof"
GP_DIR = ROOT / "datasets" / "goodpjw_selected"
CLOUD_OUT = ROOT / "kaggle_repleaf_output"
DATA_DIR = ROOT / "data"
ORIG_DIR = ROOT / "datasets" / "original_data_aligned"

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
    print("=" * 80)
    print("FRONTIER MATHEMATICAL PROTOCOL: COMPUTING ORTHOGONAL ASSETS & SUBMISSION")
    print("=" * 80)

    # 1. Load P_15, Sub 11, Sub 10
    df15 = pd.read_csv(SUB_DIR / "sub15_top5_hybrid_breakthrough.csv")
    s15 = df15["satisfaction"].values
    s11 = pd.read_csv(SUB_DIR / "sub11_top10_climb.csv")["satisfaction"].values
    s10 = pd.read_csv(SUB_DIR / "sub10_consensus223_dual_reverse_climb.csv")["satisfaction"].values

    N = len(s15)
    P_15 = pct(s15)
    r11 = pct(s11)
    r10 = pct(s10)

    # 2. Compute Ascent Vector v_climb
    v_climb = r11 - r10
    v_climb_path = OOF_DIR / "v_climb.npy"
    np.save(v_climb_path, v_climb)
    print(f"[1/4] Computed v_climb: std={np.std(v_climb):.6f}, range=[{np.min(v_climb):.6f}, {np.max(v_climb):.6f}]")
    print(f"      Saved immutable array to: {v_climb_path}")

    # 3. Load Teacher Models & RepLeaf PLR
    sys.path.append(str(ORIG_DIR))
    import original_model

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print("[Teacher Models] Running original_model.score(test_df)...")
    t_preds = original_model.score(test_df)
    t_lgb = t_preds["ogp_lgb"]
    t_xgb = t_preds["ogp_xgb_d8"]
    t_plr = np.load(CLOUD_OUT / "test_cloud_repleaf.npy")

    # 4. Compute Orthogonal Teacher Signal T_ortho
    r_tlgb = pct(t_lgb)
    r_tplr = pct(t_plr)
    T_raw = 0.65 * r_tlgb + 0.35 * r_tplr

    # Gram-Schmidt Orthogonalization against v_climb and P_15
    v_c = v_climb / np.linalg.norm(v_climb)
    p_15_unit = P_15 / np.linalg.norm(P_15)

    proj_vc = np.dot(T_raw, v_c) * v_c
    proj_p15 = np.dot(T_raw, p_15_unit) * p_15_unit
    T_perp = T_raw - proj_vc - proj_p15
    norm_T_perp = np.linalg.norm(T_perp)
    assert norm_T_perp > 1e-12, "T_perp is degenerate!"
    T_ortho = T_perp / norm_T_perp

    t_ortho_path = OOF_DIR / "T_ortho.npy"
    np.save(t_ortho_path, T_ortho)
    print(f"[2/4] Computed T_ortho: norm={np.linalg.norm(T_ortho):.4f}, std={np.std(T_ortho):.8f}")
    print(f"      <T_ortho, v_c> = {np.dot(T_ortho, v_c):.2e}, <T_ortho, p_15> = {np.dot(T_ortho, p_15_unit):.2e}")
    print(f"      Saved immutable array to: {t_ortho_path}")

    # 5. Compute 0.96233 Consensus Masked Differential Delta_masked
    realmlp = np.load(GP_DIR / "test_realmlp_v4fa_auxfull.npy")
    r_realmlp = pct(realmlp)
    r_lgb_xgb = pct(t_lgb + t_xgb)

    P_33 = 0.45 * r_realmlp + 0.55 * r_lgb_xgb
    Delta_33 = P_33 - P_15
    q_095 = np.quantile(np.abs(Delta_33), 0.95)
    Delta_masked = Delta_33 * (np.abs(Delta_33) > q_095)

    delta_masked_path = OOF_DIR / "Delta_masked.npy"
    np.save(delta_masked_path, Delta_masked)
    non_zero_pct = (Delta_masked != 0).mean() * 100.0
    print(f"[3/4] Computed Delta_masked: q_0.95={q_095:.6f}, active rows={non_zero_pct:.2f}%")
    print(f"      Saved immutable array to: {delta_masked_path}")

    # 6. Execute Sub 18 (Gated Ascent Extrapolation: alpha=12.0, sigma=0.12)
    alpha = 12.0
    sigma = 0.12
    W = np.exp(-((P_15 - 0.5) ** 2) / (2.0 * (sigma ** 2)))
    print(f"\n[4/4] Synthesizing Gated Ascent Step (alpha={alpha:.1f}, sigma={sigma:.2f}):")
    print(f"      Variance gate W: mean={W.mean():.4f}, median={np.median(W):.4f}, min={W.min():.4f}, max={W.max():.4f}")

    P_extrap = P_15 + alpha * v_climb * W

    # 7. Apply Hierarchical Lexical Operator with Continuous SuperStack
    superstack = np.load(CLOUD_OUT / "test_cloud_repleaf_superstack.npy")
    r_superstack = pct(superstack)
    eps = 1e-7

    print(f"      Applying Hierarchical Lexical Operator (eps={eps:.1e}) via SuperStack...")
    p_perturbed = P_extrap + eps * r_superstack
    strict_ranks = np.argsort(np.argsort(p_perturbed)).astype(np.float64)
    P_final = (strict_ranks + 0.5) / float(N)

    # 8. Rigorous Mathematical Audit
    n_unique = len(np.unique(P_final))
    n_ties = N - n_unique
    delta_r = P_final - P_15
    std_delta = np.std(delta_r)
    max_delta = np.max(np.abs(delta_r))
    rho_15 = spearmanr(P_final, P_15).statistic
    rho_11 = spearmanr(P_final, r11).statistic

    print("\n" + "=" * 75)
    print("MATHEMATICAL INVARIANT AUDIT REPORT: SUBMISSION CANDIDATE 20 (PROTOCOL STEP 1)")
    print("=" * 75)
    print(f"Total Rows Evaluated:           {N:,}")
    print(f"Total Unique Values:            {n_unique:,}")
    print(f"Tied Pairs:                     {n_ties} (Exact Target: 0)")
    print(f"Rank displacement std vs P_15:  {std_delta:.6f} (Theoretical Target: ~0.021 - 0.026)")
    print(f"Max absolute displacement:      {max_delta:.6f}")
    print(f"Spearman correlation vs P_15:   {rho_15:.7f}")
    print(f"Spearman correlation vs Sub 11: {rho_11:.7f}")
    print(f"Min / Median / Max Prediction:  {P_final.min():.9f} / {np.median(P_final):.9f} / {P_final.max():.9f}")
    print("=" * 75)

    # Invariant Assertions
    assert n_ties == 0, f"FAILED: Found {n_ties} tied ranks!"
    assert not np.isnan(P_final).any(), "FAILED: NaNs found!"
    assert not np.isinf(P_final).any(), "FAILED: Infs found!"
    assert P_final.min() > 0.0 and P_final.max() < 1.0, "FAILED: Range out of bounds!"

    print(">> ALL INVARIANTS SATISFIED & SIGNED OFF.")

    # 9. Save Submission File
    out_file = SUB_DIR / "sub20_gated_ascent_extrap_a12.csv"
    out_df = pd.DataFrame({
        "id": df15["id"].values,
        "satisfaction": P_final.astype(np.float64)
    })
    out_df.to_csv(out_file, index=False, float_format="%.9f")

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Saved Successfully] {out_file} ({file_size_mb:.2f} MB, {len(out_df):,} rows)")
    print("File is ready for live submission to test the alpha=12.0 multiplier behavior.")


if __name__ == "__main__":
    main()
