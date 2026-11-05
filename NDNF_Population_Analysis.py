# To be run after CNMF-E pipeline
# This script measures how strongly calcium activity is correlated between every pair of neurons in a miniscope recording.

# Imports
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

DEFAULT_NPZ = ("/Users/noodle/Desktop/Master Neuro UPC /Master 2/Python/Rule reversal/"
               "10_15_52/miniscopeDeviceName/caiman_output_fullvideo/traces_and_rois.npz")

# Functions
def parse_args():
    p = argparse.ArgumentParser(description="Pairwise Pearson correlations between ROIs")
    p.add_argument("--npz", default=DEFAULT_NPZ, help="traces_and_rois.npz written by the pipeline")
    p.add_argument("--out", default=None, help="output folder (default: <npz folder>/correlation_analysis)")
    p.add_argument("--variant", default="dFF",
                   help="trace to use: dFF (the method chosen in the pipeline), dFF_demixed, dFF_annulus, "
                        "dFF_regression, F_raw, F_demixed (default: dFF)")
    p.add_argument("--tmin", type=float, default=None, help="start of the analysed window (s)")
    p.add_argument("--tmax", type=float, default=None, help="end of the analysed window (s)")
    p.add_argument("--bin_s", type=float, default=0.0,
                   help="bin width in seconds (0 = no binning, use every frame)")
    p.add_argument("--n_shuffles", type=int, default=500, help="circular-shift shuffles for the null")
    p.add_argument("--min_shift_s", type=float, default=10.0, help="minimum circular shift (s)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    p.add_argument("--n_clusters", type=int, default=0, help="cut the dendrogram into this many clusters (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def load_traces(npz_path, variant, tmin, tmax, bin_s):
    d = np.load(npz_path)
    if variant not in d.files:
        raise KeyError(f"'{variant}' not in {npz_path}. Available: {', '.join(d.files)}")
    X = np.asarray(d[variant], dtype=float)
    t = np.asarray(d["time_s"], dtype=float)
    fr = float(d["fr"])
    centroids = np.asarray(d["centroids"], dtype=float)
    masks = np.asarray(d["masks"], dtype=bool)

    sel = np.ones(len(t), bool)
    if tmin is not None:
        sel &= t >= tmin
    if tmax is not None:
        sel &= t <= tmax
    X, t = X[:, sel], t[sel]
    if X.shape[1] < 10:
        raise ValueError("Fewer than 10 frames in the selected window.")

    fr_eff = fr
    f = max(1, int(round(bin_s * fr))) if bin_s and bin_s > 0 else 1
    if f > 1:
        nb = X.shape[1] // f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN bins
            X = np.nanmean(X[:, :nb * f].reshape(X.shape[0], nb, f), axis=2)
        t = t[:nb * f].reshape(nb, f).mean(axis=1)
        fr_eff = fr / f

    ok = ~np.isnan(X).any(axis=0)
    print(f"  {X.shape[0]} ROIs, {X.shape[1]} samples in window, {int((~ok).sum())} dropped "
          f"(flagged motion / NaN) -> {int(ok.sum())} used, effective rate {fr_eff:.2f} Hz")
    if ok.sum() < 10:
        raise ValueError("Fewer than 10 valid samples left.")
    return X[:, ok], t[ok], fr_eff, centroids, masks

def zscore_rows(X):
    Z = X - X.mean(axis=1, keepdims=True)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

def pearson_matrix(X):
    C = zscore_rows(X) @ zscore_rows(X).T
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1, 1)

def circular_shift_null(X, n_shuffles, min_shift, rng):
    K, T = X.shape
    Z = zscore_rows(X)
    iu = np.triu_indices(K, 1)
    null = np.empty((n_shuffles, iu[0].size), np.float32)
    for s in range(n_shuffles):
        shifts = rng.integers(min_shift, T - min_shift + 1, size=K)
        Zs = np.stack([np.roll(Z[k], shifts[k]) for k in range(K)])
        null[s] = (Zs @ Zs.T)[iu]
    return null

def empirical_p(r, null_abs_sorted):
    n = null_abs_sorted.size
    n_ge = n - np.searchsorted(null_abs_sorted, np.abs(r), side="left")
    return (1.0 + n_ge) / (n + 1.0)

def bh_fdr(p):
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out

def save_matrix_csv(path, M, labels):
    np.savetxt(path, M, delimiter=",", header=",".join(labels), comments="", fmt="%.6g")

def plot_matrix(M, order, title, path, vmax, labels=None):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    Mo = M[np.ix_(order, order)]
    im = ax.imshow(Mo, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    K = len(order)
    if K <= 40:
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_histogram(r_pairs, null_pairs, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-1, 1, 101)
    ax.hist(null_pairs, bins=bins, density=True, color="0.75", label="circular-shift null")
    ax.hist(r_pairs, bins=bins, density=True, histtype="step", color="tab:red", lw=1.5, label="observed")
    lim = max(0.3, np.percentile(np.abs(np.concatenate([r_pairs, null_pairs])), 99.5))
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("pairwise Pearson r")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of pairwise correlations (mean r = {r_pairs.mean():.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_r_vs_distance(dist, r, sig, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(dist[~sig], r[~sig], s=4, color="0.6", alpha=0.5, label="n.s.")
    ax.scatter(dist[sig], r[sig], s=5, color="tab:red", alpha=0.6, label="significant (FDR)")
    edges = np.linspace(0, dist.max(), 11)
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dist >= a) & (dist < b + (1e-9 if b == edges[-1] else 0))
        if m.sum() >= 5:
            centers.append((a + b) / 2)
            means.append(r[m].mean())
    ax.plot(centers, means, "k-o", ms=4, lw=1.5, label="binned mean")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("distance between ROI centroids (px)")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_cluster_map(masks, labels, path):
    """ROI masks coloured by cluster."""
    H, W = masks.shape[1:]
    img = np.full((H, W), np.nan)
    for m, c in zip(masks, labels):
        img[m] = c
    fig, ax = plt.subplots(figsize=(7, 7 * H / W))
    ax.set_facecolor("black")
    n = int(labels.max())
    ax.imshow(np.ma.masked_invalid(img), cmap="tab10", vmin=0.5, vmax=max(n, 1) + 0.5 if n > 10 else 10.5)
    ax.set_title(f"ROIs coloured by correlation cluster ({n} clusters)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# Main
def main():
    args = parse_args()
    npz_path = Path(args.npz).expanduser()
    if not npz_path.is_file():
        raise FileNotFoundError(f"{npz_path} not found: run ndnf_cnmfe_pipeline-2.py first (or pass --npz).")
    out = Path(args.out).expanduser() if args.out else npz_path.parent / "correlation_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("1 Loading traces")
    X, t, fr_eff, centroids, masks = load_traces(npz_path, args.variant, args.tmin, args.tmax, args.bin_s)
    K, T = X.shape
    if K < 3:
        raise ValueError("At least 3 ROIs are needed.")
    labels = [f"ROI_{k}" for k in range(K)]
    iu = np.triu_indices(K, 1)

    print("2 Pearson correlations")
    R = pearson_matrix(X)
    r_pairs = R[iu]
    print(f"  mean r = {r_pairs.mean():.3f}, median = {np.median(r_pairs):.3f}, "
          f"range = [{r_pairs.min():.3f}, {r_pairs.max():.3f}]")

    print(f"3 Circular-shift null ({args.n_shuffles} shuffles) and FDR")
    min_shift = int(min(max(args.min_shift_s * fr_eff, 1), T // 4))
    null = circular_shift_null(X, args.n_shuffles, min_shift, rng)
    null_flat = null.ravel()
    p_pairs = empirical_p(r_pairs, np.sort(np.abs(null_flat)))
    q_pairs = bh_fdr(p_pairs)
    sig_pairs = q_pairs < args.alpha
    print(f"  significant pairs (FDR < {args.alpha}): {int(sig_pairs.sum())} / {len(r_pairs)} "
          f"({int((sig_pairs & (r_pairs > 0)).sum())} positive, {int((sig_pairs & (r_pairs < 0)).sum())} negative)")

    Q = np.full((K, K), np.nan)
    Pm = np.full((K, K), np.nan)
    Rsig = np.zeros((K, K))
    Q[iu], Q[(iu[1], iu[0])] = q_pairs, q_pairs
    Pm[iu], Pm[(iu[1], iu[0])] = p_pairs, p_pairs
    Rsig[iu], Rsig[(iu[1], iu[0])] = np.where(sig_pairs, r_pairs, 0), np.where(sig_pairs, r_pairs, 0)
    np.fill_diagonal(Rsig, 1.0)

    # ordering / clustering
    D = squareform(1.0 - R, checks=False)
    Z = linkage(np.clip(D, 0, None), method="average")
    order = leaves_list(Z)
    clusters = fcluster(Z, t=args.n_clusters, criterion="maxclust") if args.n_clusters > 0 else None

    # distance between ROI centroids
    dist_pairs = pdist(centroids)            

    #Save
    save_matrix_csv(out / "pearson_r.csv", R, labels)
    save_matrix_csv(out / "pearson_p_empirical.csv", Pm, labels)
    save_matrix_csv(out / "pearson_q_fdr.csv", Q, labels)
    save_matrix_csv(out / "pearson_r_significant_only.csv", Rsig, labels)
    np.savetxt(out / "cluster_order.csv", order, fmt="%d", header="roi_in_clustered_order", comments="")

    with open(out / "pairs.csv", "w") as fh:
        fh.write("roi_i,roi_j,r,p_empirical,q_fdr,significant,centroid_distance_px\n")
        for i, j, r, p, q, s, dd in zip(iu[0], iu[1], r_pairs, p_pairs, q_pairs, sig_pairs, dist_pairs):
            fh.write(f"{i},{j},{r:.5f},{p:.3g},{q:.3g},{int(s)},{dd:.2f}\n")

    Roff = R.copy()
    np.fill_diagonal(Roff, np.nan)
    Sig = np.zeros((K, K), bool)
    Sig[iu], Sig[(iu[1], iu[0])] = sig_pairs, sig_pairs
    with open(out / "roi_summary.csv", "w") as fh:
        fh.write("roi,mean_r_with_others,n_sig_positive,n_sig_negative"
                 + (",cluster" if clusters is not None else "") + "\n")
        for k in range(K):
            row = (f"{k},{np.nanmean(Roff[k]):.4f},{int((Sig[k] & (Roff[k] > 0)).sum())},"
                   f"{int((Sig[k] & (Roff[k] < 0)).sum())}")
            fh.write(row + (f",{int(clusters[k])}" if clusters is not None else "") + "\n")

    np.savez_compressed(out / "correlation_results.npz", R=R, P=Pm, Q=Q, R_significant=Rsig, order=order,
                        linkage=Z, null_r_pairs=null.astype(np.float16), pairs_i=iu[0], pairs_j=iu[1],
                        variant=args.variant, n_samples=T, fr_effective=fr_eff,
                        clusters=clusters if clusters is not None else np.array([]))

    vmax = float(max(0.1, np.percentile(np.abs(r_pairs), 99)))
    ident = np.arange(K)
    plot_matrix(R, ident, f"Pearson r, ROI order ({args.variant})", out / "pearson_matrix.png", vmax)
    plot_matrix(R, order, "Pearson r, hierarchically clustered", out / "pearson_matrix_clustered.png", vmax)
    plot_matrix(Rsig, order, f"Pearson r, only pairs with FDR < {args.alpha} (clustered)",
                out / "pearson_matrix_significant.png", vmax)
    plot_histogram(r_pairs, null_flat[::max(1, null_flat.size // 500000)], out / "pearson_histogram.png")
    plot_r_vs_distance(dist_pairs, r_pairs, sig_pairs, out / "pearson_vs_distance.png")
    if clusters is not None:
        plot_cluster_map(masks, clusters, out / "cluster_map.png")
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    main()
