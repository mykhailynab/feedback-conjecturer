#!/usr/bin/env python3
import json
import argparse
import numpy as np
from scipy import sparse

def percentile_summary(x: np.ndarray, name: str, percentiles=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x)
    qs = np.percentile(x, percentiles)
    lines = [f"{name} percentiles:"]
    for p, q in zip(percentiles, qs):
        lines.append(f"  p{p:>3}: {q}")
    return "\n".join(lines)

def topk_idx_vals(x: np.ndarray, k: int):
    k = min(int(k), x.size)
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=x.dtype)
    # argpartition for speed
    idx = np.argpartition(x, -k)[-k:]
    # sort descending
    idx = idx[np.argsort(-x[idx])]
    return idx, x[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="sae_sparse.npz", help="Path to CSR .npz from sparse.save_npz")
    ap.add_argument("--topk", type=int, default=50, help="Top-K features to display for various metrics")
    ap.add_argument("--feature_limit", type=int, default=None, help="Optional: only analyze first N features (debug)")
    ap.add_argument("--value_percentiles", type=str, default="0,1,5,10,25,50,75,90,95,99,100",
                    help="Percentiles for activation value stats (comma-separated)")
    ap.add_argument("--enhance-meta-in-json", type=str, default=None, help="Path to JSON where to enhance per-feature meta in")
    ap.add_argument("--enhance-meta-in-json-out", type=str, default=None, help="Output JSON path")
    ap.add_argument("--enhance-meta-in-json-topk", type=int, default=2000, help="Top K features to consider enhancing")
    args = ap.parse_args()

    enhance_meta_json = None
    if args.enhance_meta_in_json is not None:
        assert args.enhance_meta_in_json_out is not None, "Must specify --enhance-meta-in-json-out"
        enhance_meta_json = json.loads(open(args.enhance_meta_in_json, 'r').read())

    csr: sparse.csr_matrix = sparse.load_npz(args.npz).tocsr()

    n_rows, n_feat = csr.shape
    if args.feature_limit is not None:
        n = int(args.feature_limit)
        csr = csr[:, :n]
        n_rows, n_feat = csr.shape

    nnz_total = csr.nnz
    density = nnz_total / (n_rows * n_feat) if (n_rows > 0 and n_feat > 0) else 0.0

    # Per-feature nnz count (usage frequency) and sum of activations
    # csr.getnnz(axis=0) is efficient (counts nonzeros per column)
    nnz_per_feat = np.asarray(csr.getnnz(axis=0)).ravel().astype(np.int64)  # shape [n_feat]
    sums_per_feat = np.asarray(csr.sum(axis=0)).ravel().astype(np.float64)  # shape [n_feat]

    dead_mask = nnz_per_feat == 0
    n_dead = int(dead_mask.sum())
    pct_dead = 100.0 * n_dead / n_feat if n_feat else 0.0

    # Mean activation INCLUDING zeros
    mean_per_feat = sums_per_feat / float(n_rows) if n_rows else np.zeros_like(sums_per_feat)

    # Mean activation conditional on being active
    mean_when_active = np.zeros_like(sums_per_feat, dtype=np.float64)
    active_mask = nnz_per_feat > 0
    mean_when_active[active_mask] = sums_per_feat[active_mask] / nnz_per_feat[active_mask]

    # Usage rate = nnz / n_rows (prob feature active on a random token)
    usage_rate = nnz_per_feat / float(n_rows) if n_rows else np.zeros_like(nnz_per_feat, dtype=np.float64)

    if enhance_meta_json is not None:
        top_idx, top_vals = topk_idx_vals(
            nnz_per_feat.astype(np.float64), args.enhance_meta_in_json_topk
        )
        n_set = 0
        for i, (fid, cnt) in enumerate(zip(top_idx, top_vals), start=1):
            rate = usage_rate[fid]
            mwa = mean_when_active[fid]
            mpf = mean_per_feat[fid]
            if str(fid) not in enhance_meta_json['features']:
                continue
            enhance_meta_json['features'][str(fid)]['mention_rate'] = float(rate)
            enhance_meta_json['features'][str(fid)]['nnz_count'] = int(cnt)
            enhance_meta_json['features'][str(fid)]['mean_when_active'] = float(mwa)
            enhance_meta_json['features'][str(fid)]['mean_all'] = float(mpf)
            n_set += 1
        if len(enhance_meta_json['features']) > n_set:
            print(f'[warn] Not all features set: set {n_set} < {len(enhance_meta_json["features"])} features')
        print(f"Writing '{args.enhance_meta_in_json_out}'...")
        open(args.enhance_meta_in_json_out, 'w').write(json.dumps(enhance_meta_json, ensure_ascii=False, indent=2))

    # ---- Print headline stats ----
    print("\n=== Matrix summary ===")
    print(f"File: {args.npz}")
    print(f"CSR shape: rows(tokens)={n_rows:,}, features={n_feat:,}")
    print(f"Total nnz: {nnz_total:,}")
    print(f"Density: {density:.6e} (avg nnz per row = {nnz_total / n_rows:.3f} if n_rows>0 else n/a)")

    print("\n=== Dead features ===")
    print(f"Dead features (nnz=0): {n_dead:,} / {n_feat:,}  ({pct_dead:.2f}%)")

    # ---- Top features by usage count ----
    print("\n=== Top features by usage frequency (nnz count) ===")
    analyse_activations = set()
    top_idx, top_vals = topk_idx_vals(nnz_per_feat.astype(np.float64), args.topk)
    for i, (fid, cnt) in enumerate(zip(top_idx, top_vals), start=1):
        rate = usage_rate[fid]
        analyse_activations.add(fid)
        print(f"{i:>3}. feature {fid:>7}  nnz={int(cnt):>10}  rate={rate:.6f}")

    # ---- Top features by mean activation (zeros included) ----
    print("\n=== Top features by mean activation (zeros INCLUDED) ===")
    top_idx, top_vals = topk_idx_vals(mean_per_feat, args.topk)
    for i, (fid, m) in enumerate(zip(top_idx, top_vals), start=1):
        cnt = nnz_per_feat[fid]
        mwa = mean_when_active[fid]
        analyse_activations.add(fid)
        print(f"{i:>3}. feature {fid:>7}  mean={m:.6e}  nnz={cnt:>10}  mean_when_active={mwa:.6e}")

    print("Activations to analyse:")
    open("top_features.txt", 'w').write("\n".join(map(str, analyse_activations)))

    # ---- Percentile stats ----
    print("\n=== Percentile stats: usage frequency ===")
    print(percentile_summary(nnz_per_feat.astype(np.float64), "# times activated (per feature)"))
    print()
    print(percentile_summary(usage_rate.astype(np.float64), "usage rate per feature"))

    print("\n=== Percentile stats: activation strength ===")
    print(percentile_summary(mean_per_feat.astype(np.float64), "mean activation per feature (zeros included)"))
    print()
    # conditional means can be informative but ignore dead features
    if active_mask.any():
        print(percentile_summary(mean_when_active[active_mask].astype(np.float64), "mean activation per feature (only non-zero)"))
    else:
        print("mean_when_active: no active features (all dead)")

    # ---- Global value distribution (nonzeros only) ----
    val_ps = tuple(int(x) for x in args.value_percentiles.split(",") if x.strip() != "")
    if csr.nnz > 0:
        data = csr.data.astype(np.float64)
        print("\n=== Nonzero activation value distribution (global) ===")
        print(f"nonzero values: count={data.size:,}  min={data.min():.6}  max={data.max():.6}  mean={data.mean():.6}")
        qs = np.percentile(data, val_ps)
        for p, q in zip(val_ps, qs):
            print(f"  p{p:>3}: {q:.6}")

    # ---- Concentration: how much mass in top-K features ----
    total_mass = sums_per_feat.sum()
    if total_mass > 0:
        print()
        for num_top_feats in (10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000):
            num_top_feats = min(num_top_feats, n_feat)
            top_idx, top_sums = topk_idx_vals(sums_per_feat, num_top_feats)
            share = top_sums.sum() / total_mass
            print(f"Mass concentration: top-{num_top_feats} ({num_top_feats/n_feat*100:.2f}%) features account for {share*100:.2f}% of total activation mass")
        print()
    else:
        print("\nMass concentration: total activation mass is 0 (all entries are zero).")

    # ---- Optional: report “alive but extremely rare” ----
    if n_rows > 0:
        rare_threshold = 1e-5  # active on <= 1e-5 of tokens
        rare = (usage_rate > 0) & (usage_rate <= rare_threshold)
        n_rare = int(rare.sum())
        if n_rare > 0:
            print(f"\nRare features (0 < usage_rate <= {rare_threshold}): {n_rare:,}")
            # show a few of the rarest (by nnz then by mean_when_active)
            rare_ids = np.where(rare)[0]
            rare_ids = rare_ids[np.argsort(nnz_per_feat[rare_ids])][:min(20, rare_ids.size)]
            print("Rarest (by nnz):")
            for fid in rare_ids:
                print(f"  feature {fid:>7}  nnz={nnz_per_feat[fid]:>6}  rate={usage_rate[fid]:.6e}  mean_when_active={mean_when_active[fid]:.6e}")

if __name__ == "__main__":
    main()
