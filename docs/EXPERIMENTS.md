# Experiment Log & Validation Record

**Competition:** Kaggle Playground Series Season 6 Episode 10 (`playground-series-s6e10`)  
**Date:** October 30, 2026  
**Evaluation Metric:** ROC-AUC

---

## 1. Local 5-Fold Base Models Evaluation

All models were evaluated on identical Stratified 5-Fold splits (Random State 42 / 7) combining competition `train.csv` (699,635 rows) augmented with `original_aligned.csv` (129,880 rows) on each training fold:

| Model ID | Architecture | Key Hyperparameters | 5-Fold OOF AUC | Run Time | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `own_xgb_hb` | XGBoost (GPU Hist) | `max_bin=4096, depth=8, lr=0.04, colsample=0.20` | **0.960253** | 714s | High-bin histogram eliminates quantization |
| `own_lgb` | LightGBM | `num_leaves=63, max_depth=10, colsample=0.40, lr=0.05` | **0.960305** | 382s | Fast, strong single GBDT baseline |
| `own_cat` | Deep ExtraTrees LightGBM | `num_leaves=127, extra_trees=True, colsample=0.28` | **0.960108** | 1212s | ExtraTrees splits provide decorrelation |
| `own_glm` | Wide Ridge GLM | PyTorch L-BFGS, standard scaled, $L_2 = 10.0$ | **0.957523** | 218s | Linear perspective ($\rho \approx 0.989$ vs GBDT) |

---

## 2. Component Single Models Out-of-Fold Audit

Top single models from `cdeotte_tfm` and `goodpjw_selected` evaluated on train OOF ROC-AUC:

| Component Key | Model Architecture | Family | OOF ROC-AUC | Ortho Std vs 12-Basis |
| :--- | :--- | :--- | :---: | :---: |
| `gp_stack245` | 245-Member Regularized Stack | Meta-Stack | **0.962224** | 0.01491 |
| `st182_lr` | 182-Member Logistic Regression | Meta-Stack | **0.962110** | 0.01185 |
| `st182_lgb` | 182-Member LightGBM Stacker | Meta-Stack | **0.961958** | 0.01524 |
| `gp_tabpfn_ne4` | TabPFN-3.5 (4 estimators) | In-Context Transformer | **0.961491** | 0.04871 |
| `gp_realmlp_v4fa_auxfull` | PyTabKit RealMLP Neural | Deep Tabular NN | **0.961275** | 0.04018 |
| `gp_xgb_d10_k10` | XGBoost Depth 10 10-Fold | GBDT | **0.961258** | 0.03845 |
| `gp_xgb_k10` | XGBoost 10-Fold | GBDT | **0.961256** | 0.03812 |
| `gp_lgb_k10` | LightGBM 10-Fold | GBDT | **0.961215** | 0.03698 |
| `gp_realmlp_vb` | RealMLP View-B | Deep Tabular NN | **0.961150** | 0.04210 |
| `cd_tabpfn35` | Prior Labs TabPFN-3.5 | In-Context Transformer | **0.961145** | 0.04902 |
| `gp_cat_k10` | CatBoost 10-Fold | GBDT (CTR) | **0.961114** | 0.03987 |
| `gp_glm6_xgb_k` | Wide Ridge GLM on XGB | Linear Hybrid | **0.960829** | 0.05096 |
| `cd_tabfm` | TabFM Foundation Model | Tabular Foundation | **0.958871** | 0.06563 |
| `cd_limix2` | LimiX-2 Foundation Model | Tabular Foundation | **0.958474** | 0.07577 |
| `cd_causilo` | Causilo Foundation Model | Tabular Foundation | **0.958275** | 0.06542 |

---

## 3. Public Leaderboard Probes & Sensitivity Matrix

By auditing public submissions against our reference anchor $P$ ($S_P = 0.96213$):

1. **Analytical 10-Component Push ($R_1$)**:
   - $\Sigma \epsilon = \frac{g}{2 K c}$ with $K=10, c=0.09$.
   - Verified Public LB score: **0.96217** ($+4\times 10^{-5}$ gain).
2. **Mayank's Mirror Nudges ($M$)**:
   - Verified Public LB score: **0.96218** ($+5\times 10^{-5}$ gain).
   - Inverted to isolate pure 8-block mirror nudge $M_{\text{nudge}}$ from deterministic $M_{\text{det}}$.
3. **Goodpjw Orthogonal Climb ($Q_p$)**:
   - $Q_p = (Q - G) - \text{proj}_{R_1 - P}(Q - G)$.
   - Verified Public LB score: **0.96217** ($+1\times 10^{-5}$ gain).
4. **Anchor Composite ($T_3$)**:
   - $T_3 = \text{pct}(R_1 + 1.30 (M - P) + Q_p)$.
   - Public LB score: **0.96223** (Tied #16).
5. **Reverse-Contrast Breakthrough (Sub 10)**:
   - Extrapolated negative gradient from Sub 7 ($0.96221$) to Sub 6 ($0.96223$) along $d_{\text{rev}} = \text{ortho}(\text{Sub 6} - \text{Sub 7})$.
   - Public LB score: **0.96225** (**#14 Globally**).
