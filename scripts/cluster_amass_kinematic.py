"""Cluster AMASS motions using kinematic features (no learned encoder needed).

Extracts 14-dim kinematic features from G1 motion NPZ files, normalizes,
and clusters with K-Means. Output format is identical to cluster_amass_tmr.py
so the downstream pipeline (build_cluster_mapping.py, merge_cluster_motions.py,
batch_train_clusters.sh) works unchanged.

Requirements:
    pip install scikit-learn

Usage:
    # Full pipeline (extract + cluster)
    python scripts/cluster_amass_kinematic.py \
        --amass_dir amass_g1/filtered/g1 \
        --output amass_g1/clusters_kinematic.npz \
        --k 20

    # Re-cluster from saved embeddings (skip extraction)
    python scripts/cluster_amass_kinematic.py \
        --embeddings amass_g1/clusters_kinematic.npz \
        --output amass_g1/clusters_kinematic_k16.npz \
        --k 16

    # Exclude datasets
    python scripts/cluster_amass_kinematic.py \
        --amass_dir amass_g1/filtered/g1 \
        --exclude WEIZMANN \
        --output clusters_kinematic.npz
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# Kinematic feature extraction
# ============================================================

ROOT_IDX = 0
FOOT_BODY_INDICES = [5, 6, 11, 12]  # G1 left/right ankle and toe

FEATURE_NAMES = [
    "root_speed_xy_mean",
    "root_speed_xy_std",
    "root_vel_z_mean",
    "root_ang_vel_mean",
    "root_height_mean",
    "root_height_std",
    "joint_range",
    "joint_vel_mean",
    "joint_vel_std",
    "contact_ratio",
    "duration",
    "body_spread",
    "root_speed_xy_max",
    "upper_ang_vel_mean",
]


def extract_features(npz_path):
    """Extract 14-dim kinematic feature vector from a motion NPZ.

    Returns None if the file is invalid or too short.
    """
    try:
        d = np.load(npz_path)
    except Exception:
        return None

    fps = float(d["fps"].item() if d["fps"].ndim > 0 else d["fps"])

    if "body_pos_w" in d:
        body_pos = d["body_pos_w"]
        body_lin_vel = d["body_lin_vel_w"]
        body_ang_vel = d["body_ang_vel_w"]
    elif "body_positions" in d:
        # Raw AMASS format — no velocities, skip
        return None
    else:
        return None

    joint_pos = d["joint_pos"]
    joint_vel = d["joint_vel"]
    T = body_pos.shape[0]

    if T < 10:
        return None

    root_vel_xy = body_lin_vel[:, ROOT_IDX, :2]
    root_speed_xy = np.linalg.norm(root_vel_xy, axis=-1)
    root_vel_z = body_lin_vel[:, ROOT_IDX, 2]
    root_ang_vel_mag = np.linalg.norm(body_ang_vel[:, ROOT_IDX], axis=-1)
    root_height = body_pos[:, ROOT_IDX, 2]

    joint_range = np.mean(np.ptp(joint_pos, axis=0))
    joint_vel_mag = np.linalg.norm(joint_vel, axis=-1)

    foot_heights = body_pos[:, FOOT_BODY_INDICES, 2]
    contact_frames = np.any(foot_heights < 0.05, axis=1)
    contact_ratio = np.mean(contact_frames)

    duration = T / fps

    body_offsets = body_pos - body_pos[:, ROOT_IDX : ROOT_IDX + 1, :]
    body_spread = np.mean(np.linalg.norm(body_offsets, axis=-1))

    num_bodies = body_ang_vel.shape[1]
    upper_idx = list(range(13, min(30, num_bodies)))
    upper_ang_vel_mag = np.mean(np.linalg.norm(body_ang_vel[:, upper_idx], axis=-1)) if upper_idx else 0.0

    return np.array(
        [
            np.mean(root_speed_xy),
            np.std(root_speed_xy),
            np.mean(np.abs(root_vel_z)),
            np.mean(root_ang_vel_mag),
            np.mean(root_height),
            np.std(root_height),
            joint_range,
            np.mean(joint_vel_mag),
            np.std(joint_vel_mag),
            contact_ratio,
            duration,
            body_spread,
            np.max(root_speed_xy),
            upper_ang_vel_mag,
        ],
        dtype=np.float32,
    )


# ============================================================
# Feature extraction from directory
# ============================================================


def extract_all(amass_dir, exclude):
    """Extract kinematic features from all NPZ files in amass_dir."""
    root = Path(amass_dir)
    npz_files = sorted(root.rglob("*.npz"))

    exclude_set = set(exclude)
    if exclude_set:
        before = len(npz_files)
        npz_files = [f for f in npz_files if not any(ex in f.parts for ex in exclude_set)]
        print(f"  Excluded {exclude_set}: {before} -> {len(npz_files)} files", flush=True)
    else:
        print(f"  {len(npz_files)} files", flush=True)

    embeddings = []
    valid_files = []

    for i, f in enumerate(npz_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(npz_files)} ({len(valid_files)} done)", flush=True)
        feat = extract_features(str(f))
        if feat is not None:
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(feat)
            valid_files.append(str(f))

    embeddings = np.stack(embeddings)
    print(f"  Extracted: {embeddings.shape}", flush=True)
    return embeddings, valid_files


# ============================================================
# Clustering
# ============================================================


def cluster(embeddings, k=20):
    """Cluster embeddings with K-Means on standardized features."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    emb_n = scaler.fit_transform(embeddings)

    print(f"  K-Means clustering (k={k})...", flush=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_n)

    sil = silhouette_score(emb_n, labels)
    print(f"  Silhouette score: {sil:.3f}", flush=True)

    return labels


# ============================================================
# Auto-label clusters
# ============================================================


def auto_label(mean_features):
    """Generate a descriptive label from cluster mean features."""
    spd = mean_features[0]
    ang = mean_features[3]
    hgt = mean_features[4]
    jrng = mean_features[6]
    jvel = mean_features[7]

    parts = []
    if spd > 2.0:
        parts.append("sprint")
    elif spd > 1.0:
        parts.append("run")
    elif spd > 0.5:
        parts.append("walk")
    elif spd > 0.2:
        parts.append("slow")
    else:
        parts.append("static")

    if hgt < 0.5:
        parts.append("low")
    elif hgt > 0.9:
        parts.append("tall")

    if ang > 2.0:
        parts.append("spin")
    elif ang > 1.0:
        parts.append("turn")

    if jrng > 0.8:
        parts.append("wide_rom")
    if jvel > 1.5:
        parts.append("fast_joints")

    return "_".join(parts) if parts else "misc"


# ============================================================
# Printing
# ============================================================


def print_results(labels, valid_files, embeddings):
    """Print cluster summary with feature profiles."""
    n_clusters = len(set(labels) - {-1})
    print(f"\n{'='*80}")
    print(f"Total: {len(labels)}, Clusters: {n_clusters}")
    print(f"{'='*80}")

    clusters = defaultdict(list)
    for idx, (f, l) in enumerate(zip(valid_files, labels)):
        clusters[l].append((f, idx))

    header = f"{'Cl':>3s} {'N':>5s} {'Label':<25s} {'Speed':>6s} {'AngVel':>7s} {'Height':>7s} {'Contact':>8s} {'JntRange':>9s}"
    print(header)
    print("-" * 80)

    for cid in sorted(clusters):
        if cid == -1:
            print(f"\n  Noise: {len(clusters[cid])} motions")
            continue
        indices = [idx for _, idx in clusters[cid]]
        mean_feat = embeddings[indices].mean(axis=0)
        label = auto_label(mean_feat)
        print(
            f"{cid:>3d} {len(clusters[cid]):>5d} {label:<25s} "
            f"{mean_feat[0]:>6.2f} {mean_feat[3]:>7.2f} "
            f"{mean_feat[4]:>7.2f} {mean_feat[9]:>8.2f} "
            f"{mean_feat[6]:>9.2f}"
        )

    print(f"{'='*80}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Cluster AMASS motions using kinematic features.")
    parser.add_argument("--amass_dir", type=str, help="Directory of .npz motion files")
    parser.add_argument("--embeddings", type=str, help="Pre-computed .npz to re-cluster")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset names to skip")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--k", type=int, default=20, help="Number of clusters (default: 20)")
    args = parser.parse_args()

    if args.embeddings:
        print(f"[1/2] Loading pre-computed embeddings from {args.embeddings}...", flush=True)
        data = np.load(args.embeddings, allow_pickle=True)
        embeddings = data["embeddings"]
        valid_files = list(data["files"])

        if args.exclude:
            exclude_set = set(args.exclude)
            mask = [not any(ex in f for ex in exclude_set) for f in valid_files]
            embeddings = embeddings[mask]
            valid_files = [f for f, m in zip(valid_files, mask) if m]
            print(f"  After excluding {exclude_set}: {len(valid_files)} files", flush=True)
    elif args.amass_dir:
        print(f"[1/2] Extracting kinematic features from {args.amass_dir}...", flush=True)
        embeddings, valid_files = extract_all(args.amass_dir, args.exclude)
    else:
        parser.error("Provide --amass_dir or --embeddings")

    print(f"[2/2] Clustering {len(embeddings)} motions...", flush=True)
    labels = cluster(embeddings, k=args.k)

    print_results(labels, valid_files, embeddings)

    np.savez(
        args.output,
        embeddings=embeddings,
        labels=labels,
        files=np.array(valid_files, dtype=object),
    )
    print(f"\nSaved: {args.output}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  labels: {labels.shape} (k={len(set(labels) - {-1})})")
    print(f"  files: {len(valid_files)}")


if __name__ == "__main__":
    main()
