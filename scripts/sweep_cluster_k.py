"""Sweep K-Means cluster count (k) to find optimal k via silhouette score.

Extracts kinematic features from all motions, then runs K-Means for each k
and reports silhouette score, cluster size stats.

Usage:
    python scripts/sweep_cluster_k.py --amass_dir amass_g1/processed/g1
    python scripts/sweep_cluster_k.py --amass_dir amass_g1/processed/g1 --k_range 8 40 4
    python scripts/sweep_cluster_k.py --embeddings amass_g1/clusters_kinematic.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Sweep k for kinematic K-Means clustering.")
    parser.add_argument("--amass_dir", type=str, help="Directory of .npz motion files")
    parser.add_argument("--embeddings", type=str, help="Pre-computed .npz with embeddings array")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset names to skip")
    parser.add_argument(
        "--k_range", type=int, nargs=3, default=[4, 40, 4],
        help="K range: start end step (default: 4 40 4)",
    )
    parser.add_argument("--sample_size", type=int, default=5000, help="Silhouette sample size (0=all)")
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if args.embeddings:
        print(f"Loading pre-computed embeddings from {args.embeddings}...", flush=True)
        data = np.load(args.embeddings, allow_pickle=True)
        features = data["embeddings"]
    elif args.amass_dir:
        from cluster_amass_kinematic import extract_all

        print(f"Extracting kinematic features from {args.amass_dir}...", flush=True)
        features, _ = extract_all(args.amass_dir, args.exclude)
    else:
        parser.error("Provide --amass_dir or --embeddings")

    print(f"Features: {features.shape}\n", flush=True)

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    k_start, k_end, k_step = args.k_range
    ks = list(range(k_start, k_end + 1, k_step))
    sil_kwargs = {"random_state": 42}
    if args.sample_size > 0 and len(features_norm) > args.sample_size:
        sil_kwargs["sample_size"] = args.sample_size

    print(f"{'k':>4s}  {'silhouette':>10s}  {'avg_size':>8s}  {'min':>6s}  {'max':>6s}")
    print("-" * 42)

    results = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_norm)
        sil = silhouette_score(features_norm, labels, **sil_kwargs)
        sizes = [int(np.sum(labels == c)) for c in range(k)]
        results.append((k, sil, sizes))
        print(f"{k:>4d}  {sil:>10.4f}  {np.mean(sizes):>8.0f}  {min(sizes):>6d}  {max(sizes):>6d}", flush=True)

    best_k, best_sil, _ = max(results, key=lambda x: x[1])
    print(f"\nBest: k={best_k} (silhouette={best_sil:.4f})")


if __name__ == "__main__":
    main()
