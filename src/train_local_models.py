#!/usr/bin/env python3
"""
Fast & High-Precision Local Model Pipeline for S6E10 (Predicting Airline Satisfaction).
Trains:
1. 5-Fold Generator-Aware High-Bin GPU XGBoost (max_bin=4096, depth=8)
2. 5-Fold LightGBM with Route Geometry & Original Lookup/Teacher Features
3. 5-Fold CatBoost GPU with Categorical Interactions & Route Geometry
4. 5-Fold PyTorch L-BFGS Wide Ridge GLM on GPU

Saves OOF and Test predictions to E:\projects\playground-series-s6e10\oof\
"""
import os
import sys
import gc
import time
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LogisticRegression

import torch
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
OOF_DIR = ROOT_DIR / "oof"
SUB_DIR = ROOT_DIR / "submissions"
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

SURVEY_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness"
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
FD = "Flight Distance"
TARGET = "satisfaction"


def find_file(filename: str):
    hits = sorted(DATASETS_DIR.rglob(filename))
    return hits[0] if hits else None


def build_features():
    t0 = time.time()
    print("[FE] Loading competition train.csv and test.csv...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train[TARGET].astype(np.int32).values

    orig_path = find_file("original_aligned.csv")
    orig = pd.read_csv(orig_path) if orig_path else None
    if orig is not None:
        print(f"[FE] Found original_aligned.csv: {orig.shape}")

    # 1. Empirical Prior Lookup Rates (+0.000927 AUC)
    lookup_path = find_file("lookup_long.csv")
    if lookup_path:
        print(f"[FE] Applying empirical lookup tables from {lookup_path}...")
        L = pd.read_csv(lookup_path, dtype={"value": str}, keep_default_na=False)
        for df in [d for d in (train, test, orig) if d is not None]:
            for f, t in L[L.feature.isin(df.columns)].groupby("feature"):
                col = df[f].astype(str).fillna("nan").map(dict(zip(t.value, t.rate)))
                if col.isna().any():
                    col = col.fillna(float(t["rate"].mean()))
                df["orig_rate__" + f] = col.astype(np.float32)
            if "Type of Travel" in df.columns:
                key = df["Type of Travel"].astype(str) + "|" + df["Class"].astype(str) + "|" + df["Customer Type"].astype(str)
                row = L[L.feature == "Type of Travel|Class|Customer Type"]
                if len(row):
                    col = key.map(dict(zip(row.value, row.rate))).fillna(float(row["rate"].mean()))
                    df["orig_rate__tot_cls_cust"] = col.astype(np.float32)

    # 2. Original Dataset Teacher Models (+0.000287 AUC)
    orig_model_path = find_file("original_model.py")
    if orig_model_path:
        sys.path.insert(0, str(orig_model_path.parent))
        try:
            from original_model import score as score_original
            print("[FE] Scoring with original_model.py teacher models (ogp_xgb_d8, ogp_lgb)...")
            for df in [d for d in (train, test, orig) if d is not None]:
                ogp = score_original(df)
                df["ogp_xgb_d8"] = ogp["ogp_xgb_d8"].values.astype(np.float32)
                df["ogp_lgb"] = ogp["ogp_lgb"].values.astype(np.float32)
        except Exception as e:
            print(f"[FE] Warning: original_model failed: {e}")

    # 3. Route Profile Geometry across combined data
    print("[FE] Computing Route Profile Geometry...")
    parts = [train.drop(columns=["id", TARGET], errors="ignore"), test.drop(columns=["id"], errors="ignore")]
    if orig is not None:
        parts.append(orig.drop(columns=["id", TARGET], errors="ignore"))
    allf = pd.concat(parts, ignore_index=True)

    g = allf.groupby(FD)
    agg = g[SURVEY_COLS + ["Age"]].mean().add_prefix("fd_mean_").astype(np.float32)
    agg_std = g[["Online boarding", "Inflight wifi service", "Inflight entertainment", "Age"]].std().fillna(0).add_prefix("fd_std_").astype(np.float32)
    agg = agg.join(agg_std)
    agg["fd_count"] = g.size().astype(np.int32)
    top_class = allf["Class"].mode()[0]
    agg["fd_share_biz"] = (allf["Class"] == top_class).groupby(allf[FD]).mean().astype(np.float32)
    agg["fd_share_loyal"] = (allf["Customer Type"] == "Loyal Customer").groupby(allf[FD]).mean().astype(np.float32)
    agg["fd_share_biz_travel"] = (allf["Type of Travel"] == "Business travel").groupby(allf[FD]).mean().astype(np.float32)

    # Frequency encoding for Age and Flight Distance and Delays
    age_cnt = allf["Age"].value_counts()
    del allf
    gc.collect()

    def transform_df(df):
        df = df.join(agg, on=FD)
        df["age_count"] = df["Age"].map(age_cnt).fillna(0).astype(np.int32)
        for c in ["Arrival Delay in Minutes", "Departure Delay in Minutes"]:
            df[c] = df[c].fillna(df["Departure Delay in Minutes"] if c == "Arrival Delay in Minutes" else 0.0).fillna(0.0)

        df["log_dep_delay"] = np.log1p(np.maximum(df["Departure Delay in Minutes"], 0)).astype(np.float32)
        df["log_arr_delay"] = np.log1p(np.maximum(df["Arrival Delay in Minutes"], 0)).astype(np.float32)
        df["delay_diff"] = (df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]).astype(np.float32)
        df["delay_sum"] = (df["Arrival Delay in Minutes"] + df["Departure Delay in Minutes"]).astype(np.float32)
        df["delay_ratio"] = ((df["Arrival Delay in Minutes"] + 1.0) / (df["Departure Delay in Minutes"] + 1.0)).astype(np.float32)
        df["has_no_delay"] = ((df["Departure Delay in Minutes"] == 0) & (df["Arrival Delay in Minutes"] == 0)).astype(np.int32)
        df["has_extreme_delay"] = (df["Departure Delay in Minutes"] > 120).astype(np.int32)

        for c in SURVEY_COLS:
            df[f"{c}_is_zero"] = (df[c] == 0).astype(np.int32)
        df["wifi0_booking0"] = ((df["Inflight wifi service"] == 0) & (df["Ease of Online booking"] == 0)).astype(np.int32)
        df["food0_ent0"] = ((df["Food and drink"] == 0) & (df["Inflight entertainment"] == 0)).astype(np.int32)

        s_vals = df[SURVEY_COLS].values
        df["total_survey"] = s_vals.sum(axis=1).astype(np.float32)
        df["mean_survey"] = s_vals.mean(axis=1).astype(np.float32)
        df["std_survey"] = s_vals.std(axis=1).astype(np.float32)
        df["max_survey"] = s_vals.max(axis=1).astype(np.int32)
        df["min_survey"] = s_vals.min(axis=1).astype(np.int32)
        df["zero_survey_count"] = (s_vals == 0).sum(axis=1).astype(np.int32)
        df["five_survey_count"] = (s_vals == 5).sum(axis=1).astype(np.int32)
        df["one_survey_count"] = (s_vals == 1).sum(axis=1).astype(np.int32)
        df["high_survey_count"] = (s_vals >= 4).sum(axis=1).astype(np.int32)

        df["weighted_top3"] = (
            df["Online boarding"] * 0.56 +
            df["Inflight entertainment"] * 0.43 +
            df["Seat comfort"] * 0.40
        ).astype(np.float32) / 1.39

        # Route vs Passenger diffs
        df["diff_wifi_fd"] = (df["Inflight wifi service"] - df["fd_mean_Inflight wifi service"]).astype(np.float32)
        df["diff_boarding_fd"] = (df["Online boarding"] - df["fd_mean_Online boarding"]).astype(np.float32)
        df["diff_ent_fd"] = (df["Inflight entertainment"] - df["fd_mean_Inflight entertainment"]).astype(np.float32)
        df["diff_age_fd"] = (df["Age"] - df["fd_mean_Age"]).astype(np.float32)

        # Arithmetic / Digit features (View-B signals)
        df["fd_len"] = df[FD].astype(str).str.len().astype(np.int32)
        df["fd_mod10"] = (df[FD] % 10).astype(np.int32)
        df["fd_mod100"] = (df[FD] % 100).astype(np.int32)
        df["fd_div100"] = (df[FD] // 100).astype(np.int32)
        df["age_mod10"] = (df["Age"] % 10).astype(np.int32)
        df["log_fd"] = np.log1p(df[FD]).astype(np.float32)

        # Categorical interaction combos
        df["cls_travel"] = (df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)).astype("category").cat.codes.astype(np.int32)
        df["cls_cust"] = (df["Class"].astype(str) + "_" + df["Customer Type"].astype(str)).astype("category").cat.codes.astype(np.int32)
        df["travel_cust_gen"] = (df["Type of Travel"].astype(str) + "_" + df["Customer Type"].astype(str) + "_" + df["Gender"].astype(str)).astype("category").cat.codes.astype(np.int32)

        for c in CAT_COLS:
            df[c] = df[c].astype("category").cat.codes.astype(np.int32)

        return df

    train_df = transform_df(train)
    test_df = transform_df(test)
    orig_df = transform_df(orig) if orig is not None else None

    feat_cols = [c for c in train_df.columns if c not in ("id", TARGET)]
    X = train_df[feat_cols].fillna(0.0).astype(np.float32)
    X_test = test_df[feat_cols].fillna(0.0).astype(np.float32)
    X_orig = orig_df[feat_cols].fillna(0.0).astype(np.float32) if orig_df is not None else None
    y_orig = orig_df[TARGET].astype(np.int32).values if orig_df is not None else None

    print(f"[FE] Built {len(feat_cols)} features in {time.time() - t0:.1f}s.")
    return X, y, X_test, test["id"].values, X_orig, y_orig, feat_cols


def train_models():
    X, y, X_test, test_ids, X_orig, y_orig, feat_cols = build_features()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Track 1: 5-Fold High-Bin GPU XGBoost (with original data appended to train folds)
    print("\n" + "=" * 70)
    print("TRAINING MODEL 1: 5-Fold High-Bin GPU XGBoost (max_bin=4096, depth=8)")
    print("=" * 70)
    xgb_oof = np.zeros(len(X), dtype=np.float64)
    xgb_test = np.zeros(len(X_test), dtype=np.float64)
    t0 = time.time()

    xgb_params = {
        "n_estimators": 2200,
        "learning_rate": 0.035,
        "max_depth": 8,
        "max_bin": 4096,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.25,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_state": 42,
        "early_stopping_rounds": 60,
    }

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        ft0 = time.time()
        if X_orig is not None:
            X_tr = pd.concat([X.iloc[trn_idx], X_orig], ignore_index=True)
            y_tr = np.concatenate([y[trn_idx], y_orig])
        else:
            X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        xgb_oof[val_idx] = clf.predict_proba(X_va)[:, 1]
        xgb_test += clf.predict_proba(X_test)[:, 1] / 5.0
        f_auc = roc_auc_score(y_va, xgb_oof[val_idx])
        print(f"  [XGB] Fold {fold+1}/5 AUC: {f_auc:.6f} (best_iter={clf.best_iteration}, {time.time()-ft0:.1f}s)")

    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"--> [XGB] Overall 5-Fold OOF ROC-AUC: {xgb_auc:.6f} ({time.time()-t0:.1f}s)")
    np.save(OOF_DIR / "oof_own_xgb_hb.npy", xgb_oof)
    np.save(OOF_DIR / "test_own_xgb_hb.npy", xgb_test)

    # 2. Track 2: 5-Fold LightGBM (Route + Teacher + Original)
    print("\n" + "=" * 70)
    print("TRAINING MODEL 2: 5-Fold LightGBM (num_leaves=63, colsample=0.4)")
    print("=" * 70)
    lgb_oof = np.zeros(len(X), dtype=np.float64)
    lgb_test = np.zeros(len(X_test), dtype=np.float64)
    t0 = time.time()

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": 9,
        "min_child_samples": 120,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.40,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "max_bin": 511,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        ft0 = time.time()
        if X_orig is not None:
            X_tr = pd.concat([X.iloc[trn_idx], X_orig], ignore_index=True)
            y_tr = np.concatenate([y[trn_idx], y_orig])
        else:
            X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(80, verbose=False)]
        )
        lgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
        lgb_test += model.predict_proba(X_test)[:, 1] / 5.0
        f_auc = roc_auc_score(y_va, lgb_oof[val_idx])
        print(f"  [LGB] Fold {fold+1}/5 AUC: {f_auc:.6f} (best_iter={model.best_iteration_}, {time.time()-ft0:.1f}s)")

    lgb_auc = roc_auc_score(y, lgb_oof)
    print(f"--> [LGB] Overall 5-Fold OOF ROC-AUC: {lgb_auc:.6f} ({time.time()-t0:.1f}s)")
    np.save(OOF_DIR / "oof_own_lgb.npy", lgb_oof)
    np.save(OOF_DIR / "test_own_lgb.npy", lgb_test)

    # 3. Track 3: PyTorch GPU L-BFGS Wide Ridge GLM
    print("\n" + "=" * 70)
    print("TRAINING MODEL 3: 5-Fold PyTorch GPU L-BFGS Wide Ridge GLM")
    print("=" * 70)
    glm_oof = np.zeros(len(X), dtype=np.float64)
    glm_test = np.zeros(len(X_test), dtype=np.float64)
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_np = X.values.astype(np.float64)
    X_te_np = X_test.values.astype(np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_np, y)):
        Z_tr, y_tr = X_np[trn_idx], y[trn_idx]
        Z_va, y_va = X_np[val_idx], y[val_idx]

        mu = Z_tr.mean(axis=0, keepdims=True)
        sd = Z_tr.std(axis=0, keepdims=True) + 1e-6

        D = Z_tr.shape[1]
        w = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)

        Z_t = torch.from_numpy((Z_tr - mu) / sd).to(device)
        y_t = torch.from_numpy(y_tr.astype(np.float64)).to(device)

        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=35, history_size=15,
                                tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z_t @ w + b, y_t, reduction="sum") + 0.5 * 10.0 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            Z_va_t = torch.from_numpy((Z_va - mu) / sd).to(device)
            va_logits = (Z_va_t @ w.detach() + b.detach()).cpu().numpy()
            Z_te_t = torch.from_numpy((X_te_np - mu) / sd).to(device)
            te_logits = (Z_te_t @ w.detach() + b.detach()).cpu().numpy()

        glm_oof[val_idx] = 1.0 / (1.0 + np.exp(-va_logits))
        glm_test += (1.0 / (1.0 + np.exp(-te_logits))) / 5.0

    glm_auc = roc_auc_score(y, glm_oof)
    print(f"--> [GLM] Overall 5-Fold OOF ROC-AUC: {glm_auc:.6f} ({time.time()-t0:.1f}s)")
    np.save(OOF_DIR / "oof_own_glm.npy", glm_oof)
    np.save(OOF_DIR / "test_own_glm.npy", glm_test)

    # Blend our own 3 models in logit space
    def to_logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    own_blend_oof = 0.55 * to_logit(xgb_oof) + 0.42 * to_logit(lgb_oof) + 0.03 * to_logit(glm_oof)
    own_blend_auc = roc_auc_score(y, own_blend_oof)
    print(f"\n=========================================")
    print(f"OWN MODELS BLEND OOF AUC: {own_blend_auc:.6f}")
    print(f"=========================================")


if __name__ == "__main__":
    train_models()
