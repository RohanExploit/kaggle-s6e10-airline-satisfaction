#!/usr/bin/env python3
"""
Train Model 4: 5-Fold LightGBM DART / ExtraTrees Classifier (colsample=0.25, num_leaves=127, extra_trees=True)
on the 113 engineered features + Teacher Logits + Original Data Aligned.
"""
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
OOF_DIR = ROOT_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

ORIG_DIR = DATASETS_DIR / "original_data_aligned"
if str(ORIG_DIR) not in sys.path:
    sys.path.insert(0, str(ORIG_DIR))

from train_local_models import build_features


def main():
    X_tr, y_tr, X_te, test_ids, X_orig, y_orig, feat_cols = build_features()

    print("\n" + "=" * 70)
    print("TRAINING MODEL 4: 5-Fold Deep ExtraTrees LightGBM (num_leaves=127, extra_trees=True, colsample=0.28)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    et_oof = np.zeros(len(X_tr), dtype=np.float32)
    et_test = np.zeros(len(X_te), dtype=np.float32)

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        ft0 = time.time()
        X_f_tr = pd.concat([X_tr.iloc[trn_idx], X_orig], ignore_index=True)
        y_f_tr = np.concatenate([y_tr[trn_idx], y_orig])
        X_f_va = X_tr.iloc[val_idx]
        y_f_va = y_tr[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=3500,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=10,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.28,
            extra_trees=True,
            reg_alpha=3.0,
            reg_lambda=10.0,
            random_state=1042 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_f_tr,
            y_f_tr,
            eval_set=[(X_f_va, y_f_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        val_p = model.predict_proba(X_f_va)[:, 1].astype(np.float32)
        et_oof[val_idx] = val_p
        f_auc = roc_auc_score(y_f_va, val_p)
        et_test += model.predict_proba(X_te)[:, 1].astype(np.float32) / 5.0
        print(f"  [LGBM-ExtraTrees] Fold {fold+1}/5 AUC: {f_auc:.6f} (best_iter={model.best_iteration_}, {time.time()-ft0:.1f}s)")

    overall_auc = roc_auc_score(y_tr, et_oof)
    print(f"--> [LGBM-ExtraTrees] Overall 5-Fold OOF ROC-AUC: {overall_auc:.6f} ({time.time()-t0:.1f}s)")
    np.save(OOF_DIR / "oof_own_cat.npy", et_oof)
    np.save(OOF_DIR / "test_own_cat.npy", et_test)


if __name__ == "__main__":
    main()
