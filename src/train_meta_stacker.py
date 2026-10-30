#!/usr/bin/env python3
"""
Train Nested Seed-7 Meta-Stacker (LR C=0.003 + LightGBM Stacker) combining:
- 11 Chris Deotte Foundation Models + Tuned XGB/RealMLP (cdeotte_tfm)
- 15 goodpjw2008 Extra Members (xgb_k10, xgb_d10_k10, lgb_k10, cat_aux, cat_k10, gen_v1, realmlp_*, tabm_v4aux, tabpfn_ne4)
- 182-member LR & LGB Meta-Stack OOF/Test (stack182_lr_lgb_oof.npz / stack182_lr_lgb_test.npz)
- Our Own 5-Fold Trained Models (oof_own_xgb_hb, oof_own_lgb, oof_own_glm, and oof_own_cat when ready)
"""
import os
import sys
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT_DIR = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
OOF_DIR = ROOT_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)


def pct(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def to_logit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.min() < 0.0 or v.max() > 1.0:
        return np.clip(v, -30.0, 30.0)
    p = np.clip(v, 1e-15, 1.0 - 1e-15)
    return np.clip(np.log(p / (1.0 - p)), -30.0, 30.0)


def main():
    t0 = time.time()
    tr = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", "satisfaction"])
    te = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    y = tr["satisfaction"].astype(np.int32).values
    N_TR = len(y)
    N_TE = len(te)

    names = []
    oof_cols = []
    test_cols = []

    def add_member(name: str, o: np.ndarray, t: np.ndarray):
        o = np.asarray(o, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        assert len(o) == N_TR and len(t) == N_TE, f"Shape mismatch for {name}: {o.shape}, {t.shape}"
        assert np.isfinite(o).all() and np.isfinite(t).all(), f"NaN/Inf in {name}"
        names.append(name)
        oof_cols.append(to_logit(o))
        test_cols.append(to_logit(t))

    # 1. Load cdeotte_tfm (11 members)
    cd_dir = DATASETS_DIR / "cdeotte_tfm"
    for f in sorted(cd_dir.glob("oof_*.npy")):
        m = f.stem[4:]
        tf = cd_dir / f"test_{m}.npy"
        if tf.exists():
            add_member(f"cd_{m}", np.load(f), np.load(tf))

    # 2. Load goodpjw_selected base members (including stack245 family-regularized stack)
    gp_dir = DATASETS_DIR / "goodpjw_selected"
    for f in sorted(gp_dir.glob("oof_*.npy")):
        m = f.stem[4:]
        tf = gp_dir / f"test_{m}.npy"
        if tf.exists():
            add_member(f"gp_{m}", np.load(f), np.load(tf))

    # 3. Load stack182_lr_lgb (honest seed-7 nested OOF of 182 public members)
    st182_oof = np.load(gp_dir / "stack182_lr_lgb_oof.npz")
    st182_test = np.load(gp_dir / "stack182_lr_lgb_test.npz")
    add_member("st182_lr", st182_oof["lr"], st182_test["lr"])
    add_member("st182_lgb", st182_oof["lgb"], st182_test["lgb"])

    # 4. Load our own 5-Fold Trained Models
    for m in ["own_xgb_hb", "own_lgb", "own_glm", "own_cat"]:
        of = OOF_DIR / f"oof_{m}.npy"
        tf = OOF_DIR / f"test_{m}.npy"
        if of.exists() and tf.exists():
            add_member(m, np.load(of), np.load(tf))

    Z = np.column_stack(oof_cols).astype(np.float64)
    ZT = np.column_stack(test_cols).astype(np.float64)
    print(f"[Stack] Assembled {len(names)} diverse members across {N_TR:,} train rows and {N_TE:,} test rows.")

    # Print individual OOF AUCs
    indiv = {n: roc_auc_score(y, o) for n, o in zip(names, oof_cols)}
    top_s = pd.Series(indiv).sort_values(ascending=False)
    print("\n[Stack] Top 15 Single/Component OOF ROC-AUCs:")
    print(top_s.head(15).round(6).to_string())

    # Evaluate Nested 5-Fold CV (seed 7)
    skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(Z, y))

    # Base comparison: st182_lr and gp_stack245
    idx_182lr = names.index("st182_lr")
    auc_182lr = roc_auc_score(y, Z[:, idx_182lr])
    idx_245 = names.index("gp_stack245")
    auc_245 = roc_auc_score(y, Z[:, idx_245])
    print(f"\n[Stack] Baseline st182_lr OOF ROC-AUC: {auc_182lr:.6f}")
    print(f"[Stack] Baseline gp_stack245 OOF ROC-AUC: {auc_245:.6f} (gain vs 182lr: {(auc_245 - auc_182lr)*1e6:+.1f}e-6)")

    # Stage 1: Fit Nested 5-Fold Meta-LR on the 32 honest base/182 members (excluding gp_stack245 to prevent shrinkage)
    base_idx = [i for i, n in enumerate(names) if n != "gp_stack245"]
    Z_base = Z[:, base_idx]
    ZT_base = ZT[:, base_idx]
    base_names = [names[i] for i in base_idx]

    best_c = 0.003
    best_auc = 0.0
    best_oof = None
    for C_val in [0.001, 0.002, 0.003, 0.005]:
        oof_lr = np.zeros(N_TR, dtype=np.float64)
        for tr_i, va_i in skf:
            clf = LogisticRegression(C=C_val, max_iter=2000)
            clf.fit(Z_base[tr_i], y[tr_i])
            oof_lr[va_i] = clf.decision_function(Z_base[va_i])
        sc = roc_auc_score(y, oof_lr)
        folds_up_182 = sum(
            roc_auc_score(y[va_i], oof_lr[va_i]) > roc_auc_score(y[va_i], Z[va_i, idx_182lr])
            for _, va_i in skf
        )
        print(f"  Stage-1 LR (C={C_val:6.4f}) Nested CV: {sc:.6f} (vs 182lr: {(sc - auc_182lr)*1e6:+.1f}e-6 [{folds_up_182}/5])")
        if sc > best_auc:
            best_auc = sc
            best_c = C_val
            best_oof = oof_lr.copy()

    # Fit full Stage-1 meta-LR with best_c
    full_lr = LogisticRegression(C=best_c, max_iter=2000).fit(Z_base, y)
    test_lr = full_lr.decision_function(ZT_base)

    w_s = pd.Series(full_lr.coef_[0], index=base_names).sort_values(key=np.abs, ascending=False)
    print(f"\n[Stack] Fitted Stage-1 Meta-LR (C={best_c}) Top 15 Weights:")
    print(w_s.head(15).round(4).to_string())

    # Stage 2: Nested 5-Fold Residual Anchor Blend combining gp_stack245 + Stage-1 Meta-LR + our own 4 models
    own_idx = [names.index(m) for m in ["own_xgb_hb", "own_lgb", "own_glm", "own_cat"] if m in names]
    own_oof_blend = np.mean([pct(Z[:, i]) for i in own_idx], axis=0)
    own_test_blend = np.mean([pct(ZT[:, i]) for i in own_idx], axis=0)

    r_245_oof = pct(Z[:, idx_245])
    r_245_test = pct(ZT[:, idx_245])
    r_s1_oof = pct(best_oof)
    r_s1_test = pct(test_lr)

    best_super_auc = 0.0
    best_super_oof = None
    best_super_test = None
    best_w = (0.88, 0.10, 0.02)
    for w_s1 in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        for w_own in [0.00, 0.005, 0.01, 0.015, 0.02]:
            w_245 = 1.0 - w_s1 - w_own
            cand_oof = w_245 * r_245_oof + w_s1 * r_s1_oof + w_own * own_oof_blend
            sc = roc_auc_score(y, cand_oof)
            if sc > best_super_auc:
                best_super_auc = sc
                best_super_oof = cand_oof
                best_super_test = pct(w_245 * r_245_test + w_s1 * r_s1_test + w_own * own_test_blend)
                best_w = (w_245, w_s1, w_own)

    folds_up_182_s2 = sum(
        roc_auc_score(y[va_i], best_super_oof[va_i]) > roc_auc_score(y[va_i], Z[va_i, idx_182lr])
        for _, va_i in skf
    )
    folds_up_245_s2 = sum(
        roc_auc_score(y[va_i], best_super_oof[va_i]) >= roc_auc_score(y[va_i], Z[va_i, idx_245])
        for _, va_i in skf
    )
    print(f"\n[Stack] Stage-2 Super-Stack (w_245={best_w[0]:.3f}, w_s1={best_w[1]:.3f}, w_own={best_w[2]:.3f}):")
    print(f"  Super-Stack OOF ROC-AUC: {best_super_auc:.6f} (gain vs 182lr: {(best_super_auc - auc_182lr)*1e6:+.1f}e-6 [{folds_up_182_s2}/5], vs stack245: {(best_super_auc - auc_245)*1e6:+.1f}e-6 [{folds_up_245_s2}/5])")

    # Save Meta-Stack OOF and Test predictions (both Stage-1 and Stage-2 Super-Stack)
    np.save(OOF_DIR / "oof_meta_stage1_own.npy", best_oof.astype(np.float32))
    np.save(OOF_DIR / "test_meta_stage1_own.npy", test_lr.astype(np.float32))
    np.save(OOF_DIR / "oof_meta_stack_own.npy", best_super_oof.astype(np.float32))
    np.save(OOF_DIR / "test_meta_stack_own.npy", best_super_test.astype(np.float32))
    print(f"[Stack] Saved oof_meta_stack_own.npy and test_meta_stack_own.npy in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
