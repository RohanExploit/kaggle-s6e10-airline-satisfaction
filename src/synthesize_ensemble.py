#!/usr/bin/env python3
"""
Master Ensemble & Orthogonal Manifold Synthesis Pipeline for S6E10.
Combines:
1. Calibrated Orthogonal Covariance Push (`R1`) + Mirror Pair Climb (`M`) + Orthogonal Residual (`Qp`)
2. Our Local 5-Fold Trained Models (High-Bin GPU XGBoost, Route+Teacher LightGBM, GPU Wide Ridge GLM)
3. Honest CV Stackers & TabPFN-3.5 / RealMLP / CatBoost CTR diversity signals
"""
import os
import sys
import glob
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


def find_file(name: str):
    hits = sorted(DATASETS_DIR.rglob(name))
    return hits[0] if hits else None


def load_test_vec(name: str, test_ids: np.ndarray):
    p = find_file(name)
    if p is None:
        return None
    if p.suffix == ".npy":
        v = np.load(p).astype(np.float64).ravel()
        assert len(v) == len(test_ids), f"Length mismatch for {p}"
        return pct(v)
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    v = df.set_index("id")[col].loc[test_ids].values.astype(np.float64)
    return pct(v)


def build_submissions():
    test_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"].values
    N = len(test_ids)
    print(f"[Synth] Loaded {N:,} test IDs.")

    # Load reference and anchor files
    ref_t3 = load_test_vec("ref_t3_096223.csv", test_ids)
    ref_r1 = load_test_vec("ref_r1_push_096217.csv", test_ids)
    p_213 = load_test_vec("meta_part4_096213_1023.csv", test_ids)
    q_217 = load_test_vec("meta_part6_climb1_096217_1030.csv", test_ids)
    g_216 = load_test_vec("meta_part6_probeG_096216_1028.csv", test_ids)
    twin_187 = load_test_vec("twin_submission_1015.csv", test_ids)

    print(f"[Synth] Found ref_t3_096223: {ref_t3 is not None}")
    print(f"[Synth] Found ref_r1_push_096217: {ref_r1 is not None}")
    print(f"[Synth] Found meta_part4_096213: {p_213 is not None}")
    print(f"[Synth] Found meta_part6_climb1_096217: {q_217 is not None}")
    print(f"[Synth] Found meta_part6_probeG_096216: {g_216 is not None}")
    print(f"[Synth] Found twin_submission_1015: {twin_187 is not None}")

    # Load our own local trained model predictions if ready
    own_xgb_path = OOF_DIR / "test_own_xgb_hb.npy"
    own_lgb_path = OOF_DIR / "test_own_lgb.npy"
    own_glm_path = OOF_DIR / "test_own_glm.npy"

    has_own = own_xgb_path.exists() and own_lgb_path.exists() and own_glm_path.exists()
    if has_own:
        r_xgb = pct(np.load(own_xgb_path))
        r_lgb = pct(np.load(own_lgb_path))
        r_glm = pct(np.load(own_glm_path))
        z_own = 0.55 * to_logit(r_xgb) + 0.42 * to_logit(r_lgb) + 0.03 * to_logit(r_glm)
        r_own = pct(z_own)
        print(f"[Synth] Loaded our own 5-fold trained models (XGB + LGB + GLM)!")
    else:
        r_own = None
        print(f"[Synth] Own models not yet saved; will use foundation signals.")

    # Load TabPFN-3.5 and KumoLarge and XGB TE2F50 from tkms_push_singles
    r_tp35 = load_test_vec("m_tabpfn35.csv", test_ids)
    r_kumo = load_test_vec("m_kumolarge.csv", test_ids)
    r_te50 = load_test_vec("m_xgb_te2f50.csv", test_ids)

    # Candidate 1: Exact Verified 0.96223 Orthogonal Push Anchor (Baseline Verification)
    if ref_t3 is not None:
        sub1 = pd.DataFrame({"id": test_ids, "satisfaction": ref_t3})
        sub1.to_csv(SUB_DIR / "sub1_orthogonal_push_096223.csv", index=False)
        print(f"[Synth] Saved sub1_orthogonal_push_096223.csv")

    # Candidate 2: Enhanced Orthogonal Push + Our Local Trained 5-Fold GBDT/GLM Ensemble + TabPFN-3.5
    if ref_t3 is not None and p_213 is not None and ref_r1 is not None and q_217 is not None:
        # Reconstruct orthogonal directions from P (meta_part4_096213):
        # In tkms_096223: T3 = pct(R1 + 1.3 * (M - P) + Qp)
        # Therefore (1.3 * (M - P) + Qp) is embedded in T3 vs R1!
        # Let's analyze the step vector d_ortho = ref_t3 - ref_r1 and d_push = ref_r1 - p_213
        d_push = ref_r1 - p_213
        d_ortho = ref_t3 - ref_r1

        # Candidate 2a: Scaled Orthogonal Push + Our Own 5-Fold GBDT/GLM Manifold Injection in Logit Space
        z_t3 = to_logit(ref_t3)
        if r_own is not None:
            # Blend 98.5% T3 logit + 1.5% our own 5-Fold (XGB_HighBin + LGB_RouteTeacher + Wide_GLM)
            z_c2 = 0.985 * z_t3 + 0.015 * to_logit(r_own)
            c2_ranks = pct(z_c2)
            corr_t3 = spearmanr(c2_ranks, ref_t3).statistic
            std_diff = np.std(c2_ranks - ref_t3)
            print(f"[Synth] Sub2 (T3 + 1.5% Own 5-Fold GBDT/GLM): spearman={corr_t3:.7f}, rank_std={std_diff:.6f}")
            pd.DataFrame({"id": test_ids, "satisfaction": c2_ranks}).to_csv(
                SUB_DIR / "sub2_t3_plus_own_5fold_models.csv", index=False
            )

        # Candidate 3: Extended Orthogonal Extrapolation (1.08x push + 1.05x ortho + Own Models contrastive calibration)
        ext_raw = p_213 + 1.06 * d_push + 1.05 * d_ortho
        if r_own is not None and r_tp35 is not None:
            # Push slightly toward TabPFN-3.5 and our Teacher-enriched GBDT while contrasting raw base XGB
            xgb_base = load_test_vec("probe_comp_xgb_base_raw.csv", test_ids)
            if xgb_base is not None:
                ext_raw = ext_raw + 0.012 * (r_own - xgb_base)
        c3_ranks = pct(ext_raw)
        corr_c3 = spearmanr(c3_ranks, ref_t3).statistic
        std_c3 = np.std(c3_ranks - ref_t3)
        print(f"[Synth] Sub3 (Extended Orthogonal Push + Contrastive Own Teacher-GBDT): spearman={corr_c3:.7f}, rank_std={std_c3:.6f}")
        pd.DataFrame({"id": test_ids, "satisfaction": c3_ranks}).to_csv(
            SUB_DIR / "sub3_extended_ortho_contrastive_own.csv", index=False
        )

        # Candidate 4: Anti-Shakeup Private LB Hedge (50% Top Orthogonal Push T3 + 35% Honest 245/247 Twin Stack + 15% Own 5-Fold Models)
        if twin_187 is not None:
            if r_own is not None:
                c4_ranks = pct(0.55 * ref_t3 + 0.30 * twin_187 + 0.15 * r_own)
            else:
                c4_ranks = pct(0.60 * ref_t3 + 0.40 * twin_187)
            corr_c4 = spearmanr(c4_ranks, ref_t3).statistic
            std_c4 = np.std(c4_ranks - ref_t3)
            print(f"[Synth] Sub4 (Private LB Anti-Shakeup Hedge): spearman={corr_c4:.7f}, rank_std={std_c4:.6f}")
            pd.DataFrame({"id": test_ids, "satisfaction": c4_ranks}).to_csv(
                SUB_DIR / "sub4_private_lb_hedge_stack.csv", index=False
            )


if __name__ == "__main__":
    build_submissions()
