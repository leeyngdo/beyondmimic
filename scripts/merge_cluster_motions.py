"""Merge all motions in a cluster into a single NPZ file for multi-motion training.

Concatenates all motion clips with boundary markers so MotionLoader can
randomly sample from different motions during training.

Usage:
    python scripts/merge_cluster_motions.py \
        --cluster_summary amass_g1/cluster_mapping/cluster_summary.json \
        --processed_dir amass_g1/processed/g1 \
        --output_dir amass_g1/cluster_motions

    This creates one merged NPZ per cluster:
        amass_g1/cluster_motions/cluster_0.npz
        amass_g1/cluster_motions/cluster_1.npz
        ...
"""

import argparse
import json
from pathlib import Path

import numpy as np


def merge_motions(motion_files, processed_dir):
    """Merge multiple motion NPZ files into one with boundary markers.

    Returns:
        dict with keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
        body_lin_vel_w, body_ang_vel_w, motion_boundaries (start indices of each motion),
        motion_lengths (frame count of each motion)
    """
    all_data = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    boundaries = []
    lengths = []
    fps = None
    offset = 0
    loaded = 0

    for rel_path in motion_files:
        proc_path = processed_dir / rel_path

        # Handle _clip files: try original
        if not proc_path.exists():
            stem = proc_path.stem
            for suffix in ["_clip0", "_clip1", "_clip2", "_clip3"]:
                if suffix in stem:
                    original = proc_path.parent / (stem.replace(suffix, "") + ".npz")
                    if original.exists():
                        proc_path = original
                        break

        if not proc_path.exists():
            continue

        try:
            data = np.load(proc_path)
            if fps is None:
                fps = data["fps"]

            n_frames = data["joint_pos"].shape[0]
            if n_frames < 10:
                continue

            for key in all_data:
                all_data[key].append(data[key])

            boundaries.append(offset)
            lengths.append(n_frames)
            offset += n_frames
            loaded += 1
        except Exception:
            continue

    if loaded == 0:
        return None

    merged = {
        "fps": fps,
        "motion_boundaries": np.array(boundaries, dtype=np.int64),
        "motion_lengths": np.array(lengths, dtype=np.int64),
        "num_motions": np.array([loaded]),
    }
    for key in all_data:
        merged[key] = np.concatenate(all_data[key], axis=0)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge cluster motions into single NPZ files")
    parser.add_argument("--cluster_summary", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_motions", type=int, default=0, help="Max motions per cluster (0=all)")
    args = parser.parse_args()

    with open(args.cluster_summary) as f:
        summary = json.load(f)

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cid in sorted(summary, key=int):
        cluster = summary[cid]
        name = cluster.get("name", f"cluster_{cid}")
        files = cluster["files"]

        if args.max_motions > 0:
            # Sample evenly
            import random
            random.seed(42)
            files = random.sample(files, min(args.max_motions, len(files)))

        print(f"Cluster {cid} ({name}): merging {len(files)} motions...", end=" ", flush=True)

        merged = merge_motions(files, processed_dir)
        if merged is None:
            print("SKIPPED (no valid motions)")
            continue

        output_path = output_dir / f"cluster_{cid}.npz"
        np.savez(output_path, **merged)

        total_frames = merged["joint_pos"].shape[0]
        n_motions = int(merged["num_motions"][0])
        print(f"OK ({n_motions} motions, {total_frames} frames)")

    print("\nDone!")


if __name__ == "__main__":
    main()
