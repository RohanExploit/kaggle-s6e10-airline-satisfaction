#!/usr/bin/env python3
"""
Audit all available test prediction files in datasets/goodpjw_selected and datasets/cdeotte_tfm
to identify their orthogonal components relative to [R1 - P, M - P, Qp] and their OOF AUCs.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r"E:\projects\playground-series-s6e10")
DATA_DIR = ROOT / "data"
DATASETS = ROOT / "datasets"
GOOD = DATASETS / "goodpjw_selected"
CD_DIR = DATASETS / "cdeotte_tfm"
OOF_DIR = ROOT / "oof"


def pct(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def find_file(name: str, hint: str = ""):
    hits = [
        p for p in DATASETS.rglob(name)
        if hint.lower() in str(p).replace("\\", "/").lower()
    ]
    assert hits, f"Not found: {name} ({hint})"
    return sorted(hits)[0]


def load_test_pct(name: str, te_ids: np.ndarray, hint: str = ""):
    p = find_file(name, hint)
    if p.suffix == ".npy":
        v = np.load(p).astype(np.float64).ravel()
        return pct(v)
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    v = df.set_index("id")[col].loc[te_ids].values.astype(np.float64)
    return pct(v)


def main():
    te_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"].values
    tr_y = pd.read_csv(DATA_DIR / "train.csv", usecols=["satisfaction"])["satisfaction"].values
    N = len(te_ids)
    P = load_test_pct("meta_part4_096213_1023.csv", te_ids)
    SP = 0.96213

    COMP = {
        "xgb_base": ("probe_comp_xgb_base_raw.csv",     0.95808, 0.0909),
        "gen12":    ("probe_comp_gen_v2_o12.csv",       0.95343, 0.1021),
        "realmlp":  ("probe_comp_realmlp_v4_s1.csv",    0.96082, 0.0858),
        "gen3":     ("probe_comp_gen3_pilot.csv",       0.95662, 0.0927),
        "tp35":     ("m_tabpfn35.csv",                  0.96064, 0.0880),
        "ku":       ("m_kumolarge.csv",                 0.95956, 0.0800),
        "te50":     ("m_xgb_te2f50.csv",                0.96067, 0.0870),
        "p2viewb":  ("probe_comp_revisit_p2_viewb.csv", 0.96129, 0.0850),
        "ens_v4":   ("probe_comp_ens_v4_k10.csv",       0.96107, 0.0850),
        "ne4f":     ("probe_comp_ne4f_single.csv",      0.96110, 0.0880),
    }
    keys = list(COMP.keys())
    X = {k: load_test_pct(f, te_ids) for k, (f, _, _) in COMP.items()}
    D = np.column_stack([X[k] - P for k in keys])
    Dc = D - D.mean(axis=0)
    Sig = (Dc.T @ Dc) / N
    g = np.array([SP - COMP[k][1] - 4.0 * COMP[k][2] * Sig[i, i] for i, k in enumerate(keys)])
    K, c = 10.0, 0.09
    eps = np.linalg.solve(Sig, g) / (2.0 * K * c)
    R1 = pct(P + sum(e * (P - X[k]) for k, e in zip(keys, eps)))
    M = load_test_pct("submission.csv", te_ids, hint="mayank_output")
    Q = load_test_pct("meta_part6_climb1_096217_1030.csv", te_ids)
    A = np.column_stack([R1 - P, M - P])
    coef = np.linalg.lstsq(A, Q - P, rcond=None)[0]
    Qp = (Q - P) - A @ coef

    # Also check all 25 top-board files in top_board_1027.npz!
    TB = np.load(find_file("top_board_1027.npz"))
    tb_labels = [str(l) for l in TB["labels"]]
    tb_group = [str(gr) for gr in TB["group"]]
    tb_lb = TB["lb"]
    R_tb = np.zeros((len(tb_labels), N), dtype=np.float64)
    R_tb[TB["public_rows"]] = TB["ranks"]
    mp_idx = tb_labels.index("ours: Meta Stack part 4")
    tw_idx = tb_labels.index("ours: Twin Stacks")
    R_tb[mp_idx] = P
    R_tb[tw_idx] = load_test_pct("twin_submission_1015.csv", te_ids)

    basis_12 = np.column_stack([X[k] - P for k in keys] + [M - P, Qp])
    basis_3 = np.column_stack([R1 - P, M - P, Qp])

    print("=== TOP BOARD 25 FILES: Projection onto [R1 - P, M - P, Qp] and 12-Basis ===")
    for i, (lbl, grp, lb_val) in enumerate(zip(tb_labels, tb_group, tb_lb)):
        vec = pct(R_tb[i]) - P
        if np.std(vec) < 1e-8:
            continue
        c3 = np.linalg.lstsq(basis_3, vec, rcond=None)[0]
        res3 = vec - basis_3 @ c3
        c12 = np.linalg.lstsq(basis_12, vec, rcond=None)[0]
        res12 = vec - basis_12 @ c12
        print(f"  {lbl:30s} ({grp:4s}, LB={lb_val:.5f}): std(X-P)={vec.std():.5f}, res3_std={res3.std():.5f}, res12_std={res12.std():.5f}, c_R1={c3[0]:+.2f}, c_M={c3[1]:+.2f}")

    # Now let's inspect the Nina Climb (nina2025 ->rabari9999 -> kirilltym):
    nina = pct(R_tb[tb_labels.index("nina2025 h-blend (#1)")])
    sum_comp = pct(R_tb[tb_labels.index("mayankraaz summit composer")])
    rabari = pct(R_tb[tb_labels.index("rabari9999 rank 10")])
    kirill = pct(R_tb[tb_labels.index("kirilltym gated teachers")])
    v54_step = sum_comp - rabari  # 0.96209 -> 0.96210 (+1e-5)
    rabari_step = rabari - kirill # 0.96206 -> 0.96209 (+3e-5)

    # Compare M_det against the top-board files!
    M2 = load_test_pct("submission.csv", te_ids, hint="mayank_climb2_output")
    npz = np.load(find_file("stack182_lr_lgb_test.npz"))
    scale = np.sort(npz["lr"].astype(np.float64))
    def block_noise(n, k, size, n_blocks=4, block_seed=2026, noise_seed=None):
        blocks = np.random.default_rng(block_seed).integers(0, n_blocks, n)
        seed = 100 + k if noise_seed is None else noise_seed
        return np.random.default_rng(seed).standard_normal(n) * size * (blocks == k)
    def un_nudge(rank_file, scale_arr, noise):
        z = scale_arr[np.rint(rank_file * len(scale_arr) + 0.5).astype(int) - 1]
        return pct(z + noise)
    NUDGES_6 = [
        (2026, 0, +1, 100), (2026, 1, +1, 101), (2026, 2, -1, 102), (2026, 3, -1, 103),
        (2027, 0, +1, 2027000), (2027, 1, -1, 2027001),
    ]
    M_det = M2.copy()
    for bseed, k_blk, sign, nseed in reversed(NUDGES_6):
        M_det = un_nudge(M_det, scale, -sign * block_noise(N, k_blk, 0.02, 4, bseed, nseed))

    print("\n=== Correlation of (M_det - P) with top-board differences ===")
    for nm, vec in [
        ("nina - sum_comp (h-blend top)", nina - sum_comp),
        ("sum_comp - rabari (0.96209 -> 0.96210)", v54_step),
        ("rabari - kirill (0.96206 -> 0.96209)", rabari_step),
        ("P - sum_comp (TabPFN layer)", P - sum_comp),
    ]:
        print(f"  corr(M_det - P, {nm:38s}) = {np.corrcoef(M_det - P, vec)[0,1]:+.5f}")

    # Test how much each of the 8 mirror blocks in M - M_det contributes to M - P!
    NUDGES_8 = [
        (2026, 0, +1, 100), (2026, 1, +1, 101), (2026, 2, -1, 102), (2026, 3, -1, 103),
        (2027, 0, +1, 2027000), (2027, 1, -1, 2027001), (2027, 2, +1, 2027002), (2027, 3, -1, 2027003),
    ]
    cur = M_det.copy()
    block_vecs = []
    for idx_b, (bseed, k_blk, sign, nseed) in enumerate(NUDGES_8):
        nxt = un_nudge(cur, scale, sign * block_noise(N, k_blk, 0.02, 4, bseed, nseed))
        b_vec = nxt - cur
        block_vecs.append(b_vec)
        cur = nxt
        print(f"  Block {idx_b+1} ({bseed}, b={k_blk}, s={sign:+d}): std = {b_vec.std():.6f}")

    # Notice: what if we test T3 along the rabari/summit_composer axis?
    # Specifically, d_rabari = ortho(P - rabari, [R1 - P, M - P, Qp]) (which gained +4e-5 from 0.96209 to 0.96213!)
    d_rab = (P - rabari) - basis_3 @ np.linalg.lstsq(basis_3, P - rabari, rcond=None)[0]
    d_kir = (P - kirill) - basis_3 @ np.linalg.lstsq(basis_3, P - kirill, rcond=None)[0]
    print(f"\nOrtho(P - rabari) std = {d_rab.std():.6f}, Ortho(P - kirill) std = {d_kir.std():.6f}")

    # Check all 26 single models from cdeotte_tfm and goodpjw_selected!
    single_files = sorted(list(CD_DIR.glob("test_*.npy")) + list(GOOD.rglob("test_*.npy")))
    print(f"\n=== Auditing {len(single_files)} Single Models against 12-Basis ===")
    for sf in single_files:
        name = sf.stem.replace("test_", "")
        oof_f = sf.parent / f"oof_{name}.npy"
        if not oof_f.exists():
            continue
        oof_v = np.load(oof_f).astype(np.float64).ravel()
        te_v = pct(np.load(sf).astype(np.float64).ravel())
        if len(oof_v) != len(tr_y) or len(te_v) != N:
            continue
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(tr_y, oof_v)
        vec = P - te_v
        c12 = np.linalg.lstsq(basis_12, vec, rcond=None)[0]
        res12 = vec - basis_12 @ c12
        r2 = 1.0 - (res12 @ res12) / ((vec - vec.mean()) @ (vec - vec.mean()))
        print(f"  {name:24s}: OOF={auc:.6f}, std(P-X)={vec.std():.5f}, R2_on_12={r2:.4f}, ortho_std={res12.std():.5f}")

    # Now let's check our own local 5-fold models (own_xgb_hb, own_lgb, own_glm) and stack245 vs 182lr!
    npz_oof = np.load(find_file("stack182_lr_lgb_oof.npz"))
    npz_te = np.load(find_file("stack182_lr_lgb_test.npz"))
    oof_182lr = pct(npz_oof["lr"])
    te_182lr = pct(npz_te["lr"])
    oof_245 = pct(np.load(find_file("oof_stack245.npy")))
    te_245 = pct(np.load(find_file("test_stack245.npy")))
    oof_meta_own = pct(np.load(OOF_DIR / "oof_meta_stack_own.npy"))
    te_meta_own = pct(np.load(OOF_DIR / "test_meta_stack_own.npy"))

    from sklearn.metrics import roc_auc_score
    print("\n=== OOF AUC of Stackers & Blends ===")
    print(f"  oof_182lr      : {roc_auc_score(tr_y, oof_182lr):.6f}")
    print(f"  oof_meta_own   : {roc_auc_score(tr_y, oof_meta_own):.6f} (+{(roc_auc_score(tr_y, oof_meta_own)-roc_auc_score(tr_y, oof_182lr))*1e6:.1f}e-6)")
    print(f"  oof_stack245   : {roc_auc_score(tr_y, oof_245):.6f} (+{(roc_auc_score(tr_y, oof_245)-roc_auc_score(tr_y, oof_182lr))*1e6:.1f}e-6)")
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        b_oof = (1 - w) * oof_245 + w * oof_meta_own
        print(f"  Blend (1-{w:.1f})*stack245 + {w:.1f}*meta_own: OOF AUC = {roc_auc_score(tr_y, b_oof):.6f} (+{(roc_auc_score(tr_y, b_oof)-roc_auc_score(tr_y, oof_245))*1e6:+.2f}e-6 vs stack245)")

if __name__ == "__main__":
    main()
