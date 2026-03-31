"""Build cluster mapping from clustering results.

Reads the cluster .npz output from cluster_amass_tmr.py and creates:
1. cluster_mapping.json — motion file -> cluster_id mapping
2. cluster_summary.json — cluster_id -> list of motion files
3. cluster_representatives.json — cluster_id -> representative motion (nearest to centroid)

Usage:
    python scripts/build_cluster_mapping.py \
        --clusters amass_g1/clusters_filtered.npz \
        --output_dir amass_g1/cluster_mapping
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize


def main():
    parser = argparse.ArgumentParser(description="Build cluster mapping files")
    parser.add_argument("--clusters", type=str, required=True, help="Path to clusters .npz")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for mapping files")
    parser.add_argument("--strip_prefix", type=str, default="", help="Prefix to strip from file paths")
    args = parser.parse_args()

    data = np.load(args.clusters, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    files = list(data["files"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strip prefix from paths for portability
    if args.strip_prefix:
        files_clean = [f.replace(args.strip_prefix, "") for f in files]
    else:
        files_clean = files

    # 1. Motion -> cluster mapping
    motion_to_cluster = {}
    for f, l in zip(files_clean, labels):
        motion_to_cluster[f] = int(l)

    with open(output_dir / "cluster_mapping.json", "w") as f:
        json.dump(motion_to_cluster, f, indent=2)
    print(f"cluster_mapping.json: {len(motion_to_cluster)} motions")

    # 2. Cluster -> motion list
    cluster_to_motions = defaultdict(list)
    for f, l in zip(files_clean, labels):
        cluster_to_motions[str(int(l))].append(f)

    # Sort by cluster id
    cluster_to_motions = dict(sorted(cluster_to_motions.items(), key=lambda x: int(x[0])))

    with open(output_dir / "cluster_summary.json", "w") as f:
        json.dump(cluster_to_motions, f, indent=2)

    print(f"cluster_summary.json: {len(cluster_to_motions)} clusters")
    for cid in sorted(cluster_to_motions, key=int):
        print(f"  Cluster {cid}: {len(cluster_to_motions[cid])} motions")

    # 3. Representative motion per cluster (nearest to centroid)
    emb_n = normalize(embeddings, norm="l2")
    cluster_ids = sorted(set(labels))

    representatives = {}
    for cid in cluster_ids:
        mask = labels == cid
        cluster_embs = emb_n[mask]
        cluster_files = [f for f, m in zip(files_clean, mask) if m]

        centroid = cluster_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8

        sims = cluster_embs @ centroid
        best_idx = sims.argmax()
        representatives[str(int(cid))] = {
            "file": cluster_files[best_idx],
            "similarity": float(sims[best_idx]),
            "cluster_size": int(mask.sum()),
        }

    with open(output_dir / "cluster_representatives.json", "w") as f:
        json.dump(representatives, f, indent=2)

    print(f"\ncluster_representatives.json:")
    for cid in sorted(representatives, key=int):
        r = representatives[cid]
        print(f"  Cluster {cid} ({r['cluster_size']} motions): {r['file']}")


if __name__ == "__main__":
    main()
