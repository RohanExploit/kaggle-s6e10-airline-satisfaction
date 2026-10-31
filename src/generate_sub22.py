"""
Final Submission Generator: Candidate 22 (Summit Zero-Tie Breakthrough)
Target: Top 10 Horizon (0.96238+)
Competition: Kaggle Playground Series Season 6 Episode 10
Team: ROHAN Exploit
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def main():
    print("=" * 70)
    print("GENERATING CANDIDATE 22: SUMMIT ZERO-TIE BREAKTHROUGH")
    print("=" * 70)

    # File paths
    base_dir = "E:/projects/playground-series-s6e10"
    p_sub15_path = os.path.join(base_dir, "submissions/sub15_top5_hybrid_breakthrough.csv")
    p_sub21_path = os.path.join(base_dir, "submissions/sub21_calibrated_summit_push.csv")
    p_jia_path = os.path.join(base_dir, "external/jiatian/submission.csv")
    p_superstack_path = os.path.join(base_dir, "datasets/goodpjw_selected/test_stack245.npy")
    sample_sub_path = os.path.join(base_dir, "data/sample_submission.csv")
    out_path = os.path.join(base_dir, "submissions/sub22_final_summit_zero_tie.csv")

    assert os.path.exists(p_sub15_path), f"Missing {p_sub15_path}"
    assert os.path.exists(p_sub21_path), f"Missing {p_sub21_path}"
    assert os.path.exists(p_jia_path), f"Missing {p_jia_path}"
    assert os.path.exists(p_superstack_path), f"Missing {p_superstack_path}"
    assert os.path.exists(sample_sub_path), f"Missing {sample_sub_path}"

    df_sub15 = pd.read_csv(p_sub15_path)
    df_sub21 = pd.read_csv(p_sub21_path)
    df_jia = pd.read_csv(p_jia_path)
    df_sample = pd.read_csv(sample_sub_path)
    p_superstack = np.load(p_superstack_path)

    p_sub15 = df_sub15["satisfaction"].values
    p_sub21 = df_sub21["satisfaction"].values
    p_jia = df_jia["satisfaction"].values

    N = len(p_sub15)
    assert N == 299844, f"Unexpected row count: {N}"
    assert len(p_sub21) == N and len(p_jia) == N and len(p_superstack) == N

    # Strict rank conversion
    r15 = (np.argsort(np.argsort(p_sub15)) + 0.5) / N
    r21 = (np.argsort(np.argsort(p_sub21)) + 0.5) / N
    r_jia = (np.argsort(np.argsort(p_jia)) + 0.5) / N

    # Log-odds representation
    eps = 1e-6
    z21 = np.log(np.clip(r21, eps, 1.0 - eps) / (1.0 - np.clip(r21, eps, 1.0 - eps)))
    z_jia = np.log(np.clip(r_jia, eps, 1.0 - eps) / (1.0 - np.clip(r_jia, eps, 1.0 - eps)))

    # Zero-mean unit-variance normalization
    z21_std = (z21 - z21.mean()) / z21.std()
    z_jia_std = (z_jia - z_jia.mean()) / z_jia.std()

    # Gram-Schmidt orthogonal projection of Jiatian 15-fold GPU tree against Candidate 21
    proj = np.dot(z_jia_std, z21_std) / np.dot(z21_std, z21_std)
    ortho_jia = z_jia_std - proj * z21_std
    ortho_unit = ortho_jia / np.linalg.norm(ortho_jia)

    # Scale injection: beta = 0.0050 provides safe orthogonal variance
    beta = 0.0050
    z22 = z21_std + beta * np.sqrt(N) * ortho_unit

    # Hierarchical Lexical Operator for zero-tie continuous discrimination
    epsilon = 1e-7
    p_lex_input = z22 + epsilon * p_superstack
    final_ranks = (np.argsort(np.argsort(p_lex_input)) + 0.5) / N

    # Build final DataFrame
    sub22 = pd.DataFrame({
        "id": df_sample["id"].values,
        "satisfaction": final_ranks.astype(np.float64)
    })

    # Rigorous invariant checks
    assert len(sub22) == 299844, "Row count error"
    assert sub22["id"].equals(df_sample["id"]), "ID alignment error"
    assert sub22["satisfaction"].isna().sum() == 0, "Contains NaNs"
    assert np.isfinite(sub22["satisfaction"]).all(), "Contains non-finite values"
    assert sub22["satisfaction"].is_unique, "Contains ties! Must be 100% unique"
    assert sub22["satisfaction"].between(0.0, 1.0).all(), "Values outside (0, 1)"

    # Audit statistics
    diff_15 = final_ranks - r15
    diff_21 = final_ranks - r21
    rho_15 = spearmanr(final_ranks, r15).statistic
    rho_21 = spearmanr(final_ranks, r21).statistic

    print("\n" + "=" * 70)
    print("CANDIDATE 22 MATHEMATICAL AUDIT REPORT:")
    print("=" * 70)
    print(f"Total rows                         : {len(sub22):,}")
    print(f"Unique predictions (0 ties)        : {sub22['satisfaction'].nunique():,}")
    print(f"Spearman rho vs Sub 15 Peak Base   : {rho_15:.7f}")
    print(f"Rank displacement std vs Sub 15    : {diff_15.std():.6f} (target: <= 0.0040)")
    print(f"Max rank displacement vs Sub 15    : {np.max(np.abs(diff_15)):.6f}")
    print(f"Spearman rho vs Sub 21 Push Base   : {rho_21:.7f}")
    print(f"Rank displacement std vs Sub 21    : {diff_21.std():.6f}")
    print(f"Max rank displacement vs Sub 21    : {np.max(np.abs(diff_21)):.6f}")
    print(f"Prediction min / mean / max        : {sub22['satisfaction'].min():.7f} / {sub22['satisfaction'].mean():.6f} / {sub22['satisfaction'].max():.7f}")

    sub22.to_csv(out_path, index=False)
    sha256 = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved Candidate 22 to: {out_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"SHA256: {sha256}")
    print("=" * 70)

if __name__ == "__main__":
    main()
