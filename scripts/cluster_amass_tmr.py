"""Cluster AMASS motions using TMR (Text-Motion Retrieval) semantic embeddings.

Converts G1 robot skeleton motions to HumanML3D 263-dim features,
encodes them with a pretrained TMR motion encoder, then clusters
the resulting 256-dim embeddings with UMAP + HDBSCAN.

Requirements:
    pip install umap-learn hdbscan
    TMR repo cloned at /home/jovyan/TMR with pretrained models downloaded.

Usage:
    # Full pipeline (encode + cluster)
    python scripts/cluster_amass_tmr.py --amass_dir amass_g1/g1 --output clusters.npz

    # Exclude datasets
    python scripts/cluster_amass_tmr.py --amass_dir amass_g1/g1 --exclude WEIZMANN --output clusters.npz

    # Re-cluster from saved embeddings (skip encoding)
    python scripts/cluster_amass_tmr.py --embeddings clusters.npz --exclude WEIZMANN --output clusters_no_weizmann.npz

    # Density-balanced clustering (subsample dense regions, assign all points)
    python scripts/cluster_amass_tmr.py --embeddings clusters.npz --density_subsample 5000 --output clusters_balanced.npz
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ============================================================
# G1 -> SMPL-22 joint mapping
# ============================================================
# G1 (30 bodies): pelvis(0), l_hip_pitch(1), l_hip_roll(2), l_hip_yaw(3),
#   l_knee(4), l_ankle_pitch(5), l_ankle_roll(6), r_hip_pitch(7), r_hip_roll(8),
#   r_hip_yaw(9), r_knee(10), r_ankle_pitch(11), r_ankle_roll(12), waist_yaw(13),
#   waist_roll(14), torso(15), l_shoulder_pitch(16), l_shoulder_roll(17),
#   l_shoulder_yaw(18), l_elbow(19), l_wrist_roll(20), l_wrist_pitch(21),
#   l_wrist_yaw(22), r_shoulder_pitch(23), r_shoulder_roll(24), r_shoulder_yaw(25),
#   r_elbow(26), r_wrist_roll(27), r_wrist_pitch(28), r_wrist_yaw(29)
#
# SMPL-22: pelvis(0), l_hip(1), r_hip(2), spine1(3), l_knee(4), r_knee(5),
#   spine2(6), l_ankle(7), r_ankle(8), spine3(9), l_foot(10), r_foot(11),
#   neck(12), l_collar(13), r_collar(14), head(15), l_shoulder(16),
#   r_shoulder(17), l_elbow(18), r_elbow(19), l_wrist(20), r_wrist(21)

G1_TO_SMPL22 = [
    0,   # pelvis
    3,   # l_hip -> l_hip_yaw
    9,   # r_hip -> r_hip_yaw
    13,  # spine1 -> waist_yaw
    4,   # l_knee
    10,  # r_knee
    14,  # spine2 -> waist_roll
    6,   # l_ankle -> l_ankle_roll
    12,  # r_ankle -> r_ankle_roll
    15,  # spine3 -> torso
    6,   # l_foot -> l_ankle_roll (G1 has no foot)
    12,  # r_foot -> r_ankle_roll (G1 has no foot)
    15,  # neck -> torso (G1 has no neck)
    16,  # l_collar -> l_shoulder_pitch
    23,  # r_collar -> r_shoulder_pitch
    15,  # head -> torso (G1 has no head)
    18,  # l_shoulder -> l_shoulder_yaw
    25,  # r_shoulder -> r_shoulder_yaw
    19,  # l_elbow
    26,  # r_elbow
    22,  # l_wrist -> l_wrist_yaw
    29,  # r_wrist -> r_wrist_yaw
]

MAX_FRAMES = 200  # cap at 10s @ 20fps


def compute_humanml3d_features(positions):
    """Compute 263-dim HumanML3D features from (T, 22, 3) joint positions (Z-up).

    Features: root angular velocity (1), root linear velocity XZ (2), root height (1),
    rotation-invariant joint positions (63), continuous 6D rotations (126),
    local joint velocities (66), foot contact (4).
    """
    T = positions.shape[0]
    positions = positions[:, :, [0, 2, 1]].copy()  # Z-up -> Y-up

    # Root facing direction from hips
    l_hip, r_hip = positions[:, 1], positions[:, 2]
    across = r_hip - l_hip
    across /= np.linalg.norm(across, axis=-1, keepdims=True) + 1e-8
    forward = np.cross(np.array([[0, 1, 0]]), across)
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8

    root_angle = np.arctan2(forward[:, 0], forward[:, 2])
    c, s = np.cos(root_angle), np.sin(root_angle)
    R_inv = np.zeros((T, 3, 3))
    R_inv[:, 0, 0] = c
    R_inv[:, 0, 2] = -s
    R_inv[:, 1, 1] = 1
    R_inv[:, 2, 0] = s
    R_inv[:, 2, 2] = c

    # Root angular velocity
    r_vel = np.diff(root_angle)
    r_vel = (r_vel + np.pi) % (2 * np.pi) - np.pi

    # Rotation-invariant joint positions
    root_pos = positions[:, 0:1, :]
    local_pos = positions - root_pos
    ric = np.einsum("tij,tnj->tni", R_inv, local_pos)

    # Root linear velocity (local frame, XZ)
    root_vel_g = np.diff(root_pos[:, 0], axis=0)
    l_vel = np.einsum("tij,tj->ti", R_inv[1:], root_vel_g)[:, [0, 2]]

    root_y = positions[:, 0, 1:2]

    # Joint rotations: identity approximation (no SMPL rotations available)
    cont6d = np.zeros((T, 22, 6))
    cont6d[:, :, 0] = 1.0
    cont6d[:, :, 3] = 1.0

    # Joint velocities (local frame)
    gvel = np.diff(positions, axis=0)
    lvel = np.einsum("tij,tnj->tni", R_inv[1:], gvel)

    # Foot contact (velocity thresholding)
    foot_vel = np.sum(gvel[:, [7, 10, 8, 11]] ** 2, axis=-1)
    fc = (foot_vel < 0.002).astype(np.float32)

    To = T - 1
    return np.concatenate(
        [
            r_vel[:, None],                        # 1
            l_vel,                                  # 2
            root_y[:-1],                            # 1
            ric[:-1, 1:].reshape(To, -1),           # 63
            cont6d[:-1, 1:].reshape(To, -1),        # 126
            lvel.reshape(To, -1),                    # 66
            fc,                                      # 4
        ],
        axis=-1,
    )


def convert_g1_npz(npz_path):
    """Load a G1 npz file and return HumanML3D features, or None if too short.

    Supports both raw format (body_positions) and processed/filtered format (body_pos_w).
    """
    data = np.load(npz_path)
    if "body_positions" in data:
        body_pos = data["body_positions"]
    elif "body_pos_w" in data:
        body_pos = data["body_pos_w"]
    else:
        return None
    if body_pos.shape[0] < 10:
        return None
    pos_22 = body_pos[:, G1_TO_SMPL22]
    step = max(1, round(float(data["fps"][0]) / 20.0))
    pos_22 = pos_22[::step]
    if pos_22.shape[0] > MAX_FRAMES:
        pos_22 = pos_22[:MAX_FRAMES]
    if pos_22.shape[0] < 5:
        return None
    return compute_humanml3d_features(pos_22)


def encode_all(amass_dir, exclude, tmr_dir):
    """Convert G1 motions to HumanML3D features and encode with TMR."""
    import torch

    sys.path.insert(0, tmr_dir)
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from src.data.collate import collate_x_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/3] Loading TMR model...", flush=True)
    cfg = read_config(f"{tmr_dir}/models/tmr_humanml3d_guoh3dfeats")
    model = load_model_from_cfg(cfg, "last", eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)
    print("  OK!", flush=True)

    print("[2/3] Converting & encoding...", flush=True)
    root = Path(amass_dir)
    npz_files = sorted(root.rglob("*.npz"))

    # Filter excluded datasets
    exclude_set = set(exclude)
    if exclude_set:
        before = len(npz_files)
        npz_files = [
            f for f in npz_files if not any(ex in f.parts for ex in exclude_set)
        ]
        print(f"  Excluded {exclude_set}: {before} -> {len(npz_files)} files", flush=True)
    else:
        print(f"  {len(npz_files)} files", flush=True)

    embeddings = []
    valid_files = []

    with torch.inference_mode():
        for i, f in enumerate(npz_files):
            if i % 1000 == 0:
                print(f"  {i}/{len(npz_files)} ({len(valid_files)} done)", flush=True)
            try:
                feat = convert_g1_npz(str(f))
                if feat is None or feat.shape[0] < 4:
                    continue
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

                m = torch.from_numpy(feat).float()
                m = normalizer(m)
                xd = collate_x_dict([{"x": m, "length": len(feat)}])
                xd["x"] = xd["x"].to(device)
                if "mask" in xd and isinstance(xd["mask"], torch.Tensor):
                    xd["mask"] = xd["mask"].to(device)
                if "length" in xd and isinstance(xd["length"], torch.Tensor):
                    xd["length"] = xd["length"].to(device)

                lat = model.encode(xd, sample_mean=True)
                embeddings.append(lat[0].cpu().numpy())
                valid_files.append(str(f))
            except Exception:
                continue

    embeddings = np.stack(embeddings)
    print(f"  Encoded: {embeddings.shape}", flush=True)
    return embeddings, valid_files


def farthest_point_sampling(embeddings, n_samples):
    """Select n_samples points via Farthest Point Sampling.

    Iteratively picks the point farthest from all previously selected points.
    Dense regions get fewer samples, sparse regions keep more — no dataset
    labels needed.

    Args:
        embeddings: (N, D) normalized embedding matrix.
        n_samples: number of points to select.

    Returns:
        indices: (n_samples,) array of selected indices.
    """
    N = embeddings.shape[0]
    n_samples = min(n_samples, N)
    selected = [np.random.RandomState(42).randint(N)]
    min_dists = np.full(N, np.inf)

    for i in range(1, n_samples):
        if i % 500 == 0:
            print(f"    FPS {i}/{n_samples}", flush=True)
        # Update min distance to selected set
        dists = 1.0 - embeddings @ embeddings[selected[-1]]  # cosine distance
        min_dists = np.minimum(min_dists, dists)
        # Pick the farthest point
        selected.append(np.argmax(min_dists))

    return np.array(selected)


def cluster(embeddings, valid_files, min_cluster_size=80, density_subsample=0):
    """Cluster embeddings with UMAP + HDBSCAN.

    If density_subsample > 0, uses Farthest Point Sampling to select a
    density-balanced subset for clustering, then assigns all remaining
    points to the nearest cluster centroid.
    """
    from sklearn.preprocessing import normalize
    from umap import UMAP
    from hdbscan import HDBSCAN

    emb_n = normalize(embeddings, norm="l2")

    if density_subsample > 0 and density_subsample < len(emb_n):
        print(f"  Farthest Point Sampling: {len(emb_n)} -> {density_subsample}...", flush=True)
        subset_idx = farthest_point_sampling(emb_n, density_subsample)
        emb_subset = emb_n[subset_idx]

        print(f"  UMAP on {len(emb_subset)} subset points...", flush=True)
        umap_model = UMAP(
            n_components=15, random_state=42, n_neighbors=50, min_dist=0.0, metric="cosine"
        )
        umap_subset = umap_model.fit_transform(emb_subset)

        print(f"  HDBSCAN clustering...", flush=True)
        subset_labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=5, cluster_selection_method="leaf"
        ).fit_predict(umap_subset)

        # Compute cluster centroids in embedding space (excluding noise)
        cluster_ids = sorted(set(subset_labels) - {-1})
        centroids = np.stack([
            emb_n[subset_idx[subset_labels == cid]].mean(axis=0)
            for cid in cluster_ids
        ])
        centroids = normalize(centroids, norm="l2")

        # Assign ALL points to nearest cluster centroid
        print(f"  Assigning all {len(emb_n)} points to {len(cluster_ids)} clusters...", flush=True)
        similarities = emb_n @ centroids.T  # (N, n_clusters)
        labels = np.array([cluster_ids[j] for j in similarities.argmax(axis=1)])
    else:
        print("  UMAP + HDBSCAN clustering...", flush=True)
        umap_emb = UMAP(
            n_components=15, random_state=42, n_neighbors=50, min_dist=0.0, metric="cosine"
        ).fit_transform(emb_n)
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=5, cluster_selection_method="leaf"
        ).fit_predict(umap_emb)

    return labels


def print_results(labels, valid_files, amass_dir=""):
    """Print cluster summary with sample files."""
    import random

    random.seed(42)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"\n{'='*60}")
    print(f"Total: {len(labels)}, Clusters: {n_clusters}, Noise: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")
    print(f"{'='*60}")

    clusters = defaultdict(list)
    for f, l in zip(valid_files, labels):
        clusters[l].append(f)

    for cid in sorted(clusters):
        if cid == -1:
            print(f"\n  Noise: {len(clusters[cid])} motions")
            continue
        samples = clusters[cid]
        picks = random.sample(samples, min(5, len(samples)))
        short = [p.replace(amass_dir + "/", "") if amass_dir else p for p in picks]
        print(f"\n  Cluster {cid} ({len(samples)} motions):")
        for p in short:
            print(f"    {p}")


def main():
    parser = argparse.ArgumentParser(description="Cluster AMASS motions with TMR embeddings")
    parser.add_argument("--amass_dir", type=str, default=None, help="Path to amass_g1/g1 directory")
    parser.add_argument("--embeddings", type=str, default=None, help="Load pre-computed embeddings (.npz)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset names to exclude (e.g., WEIZMANN)")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--tmr_dir", type=str, default="/home/jovyan/TMR", help="Path to TMR repo")
    parser.add_argument("--min_cluster_size", type=int, default=80, help="HDBSCAN min_cluster_size")
    parser.add_argument("--density_subsample", type=int, default=0, help="FPS subsample size for density-balanced clustering (0=disabled)")
    args = parser.parse_args()

    if args.embeddings:
        # Re-cluster from saved embeddings
        print(f"Loading embeddings from {args.embeddings}...", flush=True)
        data = np.load(args.embeddings, allow_pickle=True)
        embeddings = data["embeddings"]
        valid_files = list(data["files"])

        # Apply exclusions
        if args.exclude:
            exclude_set = set(args.exclude)
            mask = [
                not any(ex in f for ex in exclude_set) for f in valid_files
            ]
            embeddings = embeddings[mask]
            valid_files = [f for f, m in zip(valid_files, mask) if m]
            print(f"  After excluding {exclude_set}: {len(valid_files)} motions", flush=True)
    elif args.amass_dir:
        embeddings, valid_files = encode_all(args.amass_dir, args.exclude, args.tmr_dir)
    else:
        parser.error("Provide either --amass_dir or --embeddings")

    labels = cluster(embeddings, valid_files, args.min_cluster_size, args.density_subsample)

    amass_dir = args.amass_dir or ""
    print_results(labels, valid_files, amass_dir)

    np.savez(args.output, embeddings=embeddings, labels=labels, files=np.array(valid_files))
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
