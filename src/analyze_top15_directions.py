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
MAYANK1 = DATASETS / "mayank_output" / "submission.csv"
MAYANK2 = DATASETS / "mayank_climb2_output" / "submission.csv"

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
    Sig = (Dc.T @ Dc) / len(P)
    g = np.array([SP - COMP[k][1] - 4.0 * COMP[k][2] * Sig[i, i] for i, k in enumerate(keys)])
    K, c = 10.0, 0.09
    eps = np.linalg.solve(Sig, g) / (2.0 * K * c)

    R1 = pct(P + sum(e * (P - X[k]) for k, e in zip(keys, eps)))
    M = load_test_pct("submission.csv", te_ids, hint="mayank_output")
    M2 = load_test_pct("submission.csv", te_ids, hint="mayank_climb2_output")
    G = load_test_pct("meta_part6_probeG_096216_1028.csv", te_ids)
    C = load_test_pct("meta_part6_climb1_096217_1030.csv", te_ids)
    Qp = (C - G) - (((C - G) @ (R1 - P)) / ((R1 - P) @ (R1 - P))) * (R1 - P)

    # Check M vs M2
    d_m_m2 = M - M2
    print(f"M (0.96218) vs M2 (0.96217): std(M - M2) = {d_m_m2.std():.6f}, corr(M, M2) = {np.corrcoef(M, M2)[0,1]:.8f}")
    print(f"corr(M - M2, R1 - P) = {np.corrcoef(d_m_m2, R1 - P)[0,1]:.5f}")
    print(f"corr(M - M2, M2 - P) = {np.corrcoef(d_m_m2, M2 - P)[0,1]:.5f}")
    print(f"corr(M - M2, Qp)     = {np.corrcoef(d_m_m2, Qp)[0,1]:.5f}")

    npz = np.load(find_file("stack182_lr_lgb_test.npz"))
    print("npz.files:", npz.files, "lr min/max:", npz["lr"].min(), npz["lr"].max())
    st182 = npz["lr"].astype(np.float64)
    if st182.min() < 0 or st182.max() > 1:
        scale = np.sort(st182)
    else:
        p_clip = np.clip(st182, 1e-6, 1 - 1e-6)
        scale = np.sort(np.log(p_clip / (1 - p_clip)))

    def block_noise(n, k, size, n_blocks=4, block_seed=2026, noise_seed=None):
        blocks = np.random.default_rng(block_seed).integers(0, n_blocks, n)
        seed = 100 + k if noise_seed is None else noise_seed
        return np.random.default_rng(seed).standard_normal(n) * size * (blocks == k)

    def un_nudge(rank_file, scale, noise):
        z = scale[np.rint(rank_file * len(scale) + 0.5).astype(int) - 1]
        return pct(z + noise)

    test_m = M2.copy()
    for bseed, k, sign, nseed in [(2027, 2, +1, 2027002), (2027, 3, -1, 2027003)]:
        test_m = un_nudge(test_m, scale, sign * block_noise(N, k, 0.02, 4, bseed, nseed))
    print(f"\nApplying nudges 7 & 8 to M2 using stack182_lr scale -> max|diff from M| = {np.max(np.abs(test_m - M)):.6e}, std(diff) = {(test_m - M).std():.6e}")

    NUDGES_6 = [
        (2026, 0, +1, 100),
        (2026, 1, +1, 101),
        (2026, 2, -1, 102),
        (2026, 3, -1, 103),
        (2027, 0, +1, 2027000),
        (2027, 1, -1, 2027001),
    ]
    m_det = M2.copy()
    for bseed, k, sign, nseed in reversed(NUDGES_6):
        m_det = un_nudge(m_det, scale, -sign * block_noise(N, k, 0.02, 4, bseed, nseed))
    m_re = m_det.copy()
    for bseed, k, sign, nseed in NUDGES_6:
        m_re = un_nudge(m_re, scale, sign * block_noise(N, k, 0.02, 4, bseed, nseed))
    print(f"Recovered M_det (un-nudged deterministic Summit+Nina+TabPFN layer): re-nudge std diff = {(m_re - M2).std():.6e}")
    print(f"std(M_det - P) = {(m_det - P).std():.6f} (pure deterministic signal!)")
    print(f"std(M - M_det) = {(M - m_det).std():.6f} (pure 8-block mirror nudge!)")
    print(f"std(M - M2)    = {(M - M2).std():.6f} (blocks 2&3 of seed 2027 that lifted 0.96217 -> 0.96218!)")

    cd_xgb = pct(np.load(CD_DIR / "test_xgboost.npy"))
    cd_rmlp = pct(np.load(CD_DIR / "test_realmlp.npy"))
    cd_tp35 = pct(np.load(CD_DIR / "test_tabpfn35.npy"))

    for label, vec in [("P - cd_xgb", P - cd_xgb), ("P - cd_rmlp", P - cd_rmlp), ("P - cd_tp35", P - cd_tp35)]:
        c_vec, *_ = np.linalg.lstsq(Dc, vec - vec.mean(), rcond=None)
        recon = Dc @ c_vec
        res = (vec - vec.mean()) - recon
        r2 = 1.0 - (res @ res) / ((vec - vec.mean()) @ (vec - vec.mean()))
        print(f"\nProjection of {label} onto 10-model COMP D: R^2 = {r2:.4f}, raw std = {vec.std():.5f}, residual std = {res.std():.5f}")
        top_idx = np.argsort(np.abs(c_vec))[::-1][:3]
        print("  Top COMP matches:", [(keys[i], round(c_vec[i], 3)) for i in top_idx])

if __name__ == "__main__":
    main()
