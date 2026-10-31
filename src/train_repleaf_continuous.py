#!/usr/bin/env python3
"""
Train 5-Fold Continuous-Leaf RepLeafClassifier with TorchPLREncoder on GPU.
Architectural Specification: E:\projects\playground-series-s6e10\docs\RESEARCH_SPECIFICATION_REPLEAF_PLR.md
Hardware Constraints: NVIDIA RTX 2050 Laptop GPU (4GB VRAM), strictly 0 bytes to C:\.
"""
import os
import sys
import gc
import time
from pathlib import Path

# Enforce zero writes to C:\
ROOT_DIR = Path(r"E:\projects\playground-series-s6e10")
TMP_DIR = ROOT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch

DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
OOF_DIR = ROOT_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Hardware] PyTorch CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"[Hardware] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")

# Import RepLeafClassifier
from repleafgbm import RepLeafClassifier


def pct(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def compute_oof_route_bayes_target(train_df, test_df, target_col="satisfaction", route_col="Flight Distance", m_pseudo=35.0, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_encoded = np.zeros(len(train_df), dtype=np.float32)
    test_encoded = np.zeros(len(test_df), dtype=np.float32)
    y = train_df[target_col].values

    for trn_idx, val_idx in skf.split(train_df, y):
        tr_part = train_df.iloc[trn_idx]
        val_part = train_df.iloc[val_idx]
        
        global_mu = tr_part[target_col].mean()
        stats = tr_part.groupby(route_col)[target_col].agg(["count", "sum"])
        rate_map = (stats["sum"] + m_pseudo * global_mu) / (stats["count"] + m_pseudo)
        train_encoded[val_idx] = val_part[route_col].map(rate_map).fillna(global_mu).values

    full_global_mu = train_df[target_col].mean()
    full_stats = train_df.groupby(route_col)[target_col].agg(["count", "sum"])
    full_rate_map = (full_stats["sum"] + m_pseudo * full_global_mu) / (full_stats["count"] + m_pseudo)
    test_encoded = test_df[route_col].map(full_rate_map).fillna(full_global_mu).values

    return train_encoded, test_encoded


def main():
    t0 = time.time()
    print("[Pipeline] Loading competition train and test datasets...")
    tr_raw = pd.read_csv(DATA_DIR / "train.csv")
    te_raw = pd.read_csv(DATA_DIR / "test.csv")
    N_TR = len(tr_raw)
    N_TE = len(te_raw)
    y = tr_raw["satisfaction"].values.astype(np.int32)
    te_ids = te_raw["id"].values

    survey_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Cleanliness"
    ]
    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]

    # 1. External Teacher Model Logits (Shutko Original Survey)
    orig_dir = DATASETS_DIR / "original_data_aligned"
    if str(orig_dir) not in sys.path:
        sys.path.append(str(orig_dir))
    try:
        from original_model import score as score_original
        print("[FE] Scoring with original_model.py teacher logits (ogp_xgb_d8, ogp_lgb)...")
        tr_og = score_original(tr_raw)
        te_og = score_original(te_raw)
        tr_raw["ogp_xgb_d8"] = tr_og["ogp_xgb_d8"].astype(np.float32)
        tr_raw["ogp_lgb"] = tr_og["ogp_lgb"].astype(np.float32)
        te_raw["ogp_xgb_d8"] = te_og["ogp_xgb_d8"].astype(np.float32)
        te_raw["ogp_lgb"] = te_og["ogp_lgb"].astype(np.float32)
    except Exception as e:
        print(f"[FE] Warning loading original model: {e}")

    # 2. Empirical Bayes Lookup Rates
    lookup_file = orig_dir / "lookup_long.csv"
    if lookup_file.exists():
        print("[FE] Mapping empirical lookup priors from lookup_long.csv...")
        L = pd.read_csv(lookup_file, dtype={"value": str}, keep_default_na=False)
        for df in (tr_raw, te_raw):
            for f, t in L[L.feature.isin(df.columns)].groupby("feature"):
                val_map = dict(zip(t.value, t.rate))
                df["orig_rate__" + f] = df[f].astype(str).fillna("nan").map(val_map).astype(np.float32)

    # 3. Leakage-Free OOF Route Target Encoding on Flight Distance
    print("[FE] Computing Leakage-Free Out-of-Fold Route Target Encoding (m=35.0)...")
    tr_oof_route, te_route = compute_oof_route_bayes_target(tr_raw, te_raw, m_pseudo=35.0, n_splits=5, seed=42)
    tr_raw["oof_route_target"] = tr_oof_route
    te_raw["oof_route_target"] = te_route

    # 4. Shutko Original Survey Route Bayesian Priors
    orig_aligned_path = orig_dir / "original_aligned.csv"
    if orig_aligned_path.exists():
        print("[FE] Computing Shutko Original Survey Route Priors (m=20.0)...")
        df_orig = pd.read_csv(orig_aligned_path)
        orig_mu = df_orig["satisfaction"].mean()
        orig_stats = df_orig.groupby("Flight Distance")["satisfaction"].agg(["count", "sum"])
        orig_map = (orig_stats["sum"] + 20.0 * orig_mu) / (orig_stats["count"] + 20.0)
        tr_raw["shutko_route_prior"] = tr_raw["Flight Distance"].map(orig_map).fillna(orig_mu).astype(np.float32)
        te_raw["shutko_route_prior"] = te_raw["Flight Distance"].map(orig_map).fillna(orig_mu).astype(np.float32)

    # 5. Route Profile Geometry
    print("[FE] Computing Route Profile Geometry...")
    FD = "Flight Distance"
    comb = pd.concat([tr_raw.drop(columns=["id", "satisfaction"], errors="ignore"),
                      te_raw.drop(columns=["id", "satisfaction"], errors="ignore")], ignore_index=True)
    g_route = comb.groupby(FD)
    agg_route = g_route[survey_cols + ["Age"]].mean().add_prefix("fd_mean_").astype(np.float32)
    agg_route["fd_count"] = g_route.size().astype(np.int32)
    tr_raw = tr_raw.join(agg_route, on=FD)
    te_raw = te_raw.join(agg_route, on=FD)

    # 6. Delay dynamics & structural zero-rating flags
    for df in (tr_raw, te_raw):
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0.0)
        df["Departure Delay in Minutes"] = df["Departure Delay in Minutes"].fillna(0.0)
        df["log_dep"] = np.log1p(np.maximum(df["Departure Delay in Minutes"], 0)).astype(np.float32)
        df["log_arr"] = np.log1p(np.maximum(df["Arrival Delay in Minutes"], 0)).astype(np.float32)
        df["delay_diff"] = (df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]).astype(np.float32)
        for c in survey_cols:
            df[f"{c}_is_zero"] = (df[c] == 0).astype(np.int32)
        df["wifi0_booking0"] = ((df["Inflight wifi service"] == 0) & (df["Ease of Online booking"] == 0)).astype(np.int32)
        for c in cat_cols:
            df[c] = df[c].astype("category").cat.codes

    drop_cols = ["id", "satisfaction"]
    feature_cols = [c for c in tr_raw.columns if c not in drop_cols]
    print(f"[FE] Feature engineering complete in {time.time()-t0:.1f}s. Total Features: {len(feature_cols)}")

    X_tr = tr_raw[feature_cols].copy()
    X_te = te_raw[feature_cols].copy()

    # Fill NaNs with median for continuous representation
    for c in feature_cols:
        med = X_tr[c].median()
        X_tr[c] = X_tr[c].fillna(med)
        X_te[c] = X_te[c].fillna(med)

    # -------------------------------------------------------------------------
    # 5-Fold RepLeafClassifier with Continuous Embedded Linear Leaf Models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING 5-FOLD RepLeafClassifier (CONTINUOUS LEAF HYPERPLANES)")
    print("=" * 70)

    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_repleaf = np.zeros(N_TR, dtype=np.float32)
    test_repleaf = np.zeros(N_TE, dtype=np.float32)
    fold_aucs = []

    # Configured to strictly bound peak VRAM <= 1.85 GB on RTX 2050 (4GB)
    repleaf_params = dict(
        n_estimators=60,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=-1,
        leaf_model="embedded_linear",
        leaf_gate_margin=0.01,
        leaf_gate="loo",
        encoder="plr",
        max_leaf_emb_dim=64,
        l2_leaf=1.0,
        leaf_fit_precision="float32",
        random_state=42,
        device=DEVICE,
        verbose=0
    )

    t_train_start = time.time()
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y)):
        f_t0 = time.time()
        X_trn_f = X_tr.iloc[trn_idx]
        y_trn_f = y[trn_idx]
        X_val_f = X_tr.iloc[val_idx]
        y_val_f = y[val_idx]

        clf = RepLeafClassifier(**{**repleaf_params, "random_state": 42 + fold})
        clf.fit(X_trn_f, y_trn_f)

        val_preds = clf.predict_proba(X_val_f)[:, 1]
        oof_repleaf[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val_f, val_preds)
        fold_aucs.append(fold_auc)

        te_preds = clf.predict_proba(X_te)[:, 1]
        test_repleaf += te_preds / N_SPLITS

        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  Fold {fold+1}/{N_SPLITS} AUC: {fold_auc:.6f} | Time: {time.time()-f_t0:.1f}s | Peak VRAM: {mem_mb:.0f} MB")
            torch.cuda.empty_cache()
        else:
            print(f"  Fold {fold+1}/{N_SPLITS} AUC: {fold_auc:.6f} | Time: {time.time()-f_t0:.1f}s")
        gc.collect()

    overall_auc = roc_auc_score(y, oof_repleaf)
    print("=" * 70)
    print(f"--> [RepLeaf PLR] 5-Fold Overall OOF ROC-AUC: {overall_auc:.6f} (Mean Fold: {np.mean(fold_aucs):.6f})")
    print(f"    Total Training Time: {time.time()-t_train_start:.1f}s")
    print("=" * 70)

    # Save OOF and Test Predictions
    oof_path = OOF_DIR / "oof_own_repleaf.npy"
    test_path = OOF_DIR / "test_own_repleaf.npy"
    np.save(oof_path, oof_repleaf)
    np.save(test_path, test_repleaf)
    print(f"[Saved] {oof_path}")
    print(f"[Saved] {test_path}")


if __name__ == "__main__":
    main()
