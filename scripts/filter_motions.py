# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The KRAFTON Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license) and Legged Lab Project (BSD-3-Clause license),
# with modifications by KRAFTON Lab Project (BSD-3-Clause license).

# Copyright (c) 2025, The KRAFTON Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

"""Offline motion filtering script to remove infeasible frames from motion datasets.

This script filters motion data based on physical feasibility criteria:
- Root height: Reject frames where robot root is too low (falling/fallen)
- Roll/Pitch: Reject frames with excessive body tilt

When a frame is rejected, surrounding frames within a configurable buffer are also removed
to eliminate transitions to/from infeasible states.

Environment Variables:
    DATASET_DIR: Base directory containing dataset folders. If not set, defaults to
                 the parent directory of this script (i.e., dataset/).

Usage:
    # Using default dataset directory (dataset/)
    python filter_motions.py --dataset LAFAN1_Retargeting_Dataset --robot g1 --config ../configs/lafan1/filter_cfg.yaml

    # Using custom dataset directory via environment variable
    DATASET_DIR=/path/to/datasets python filter_motions.py --dataset LAFAN1_Retargeting_Dataset --robot g1

    # Override specific parameters
    python filter_motions.py --dataset LAFAN1_Retargeting_Dataset --robot g1 --min_root_height 0.4 --max_abs_roll 0.79
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import yaml


def quat_to_euler(quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quaternion (wxyz) to Euler angles (roll, pitch, yaw).

    Args:
        quat_wxyz: Quaternion array of shape (..., 4) in wxyz format.

    Returns:
        Tuple of (roll, pitch, yaw) arrays, each of shape (...,) in radians.
    """
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp to avoid numerical issues at singularities
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def find_valid_segments(valid_mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous segments of valid frames.

    Args:
        valid_mask: Boolean array where True indicates valid frames.

    Returns:
        List of (start_idx, end_idx) tuples for each valid segment.
        end_idx is exclusive (Python slice convention).
    """
    segments = []
    in_segment = False
    start_idx = 0

    for i, is_valid in enumerate(valid_mask):
        if is_valid and not in_segment:
            # Start of a new segment
            start_idx = i
            in_segment = True
        elif not is_valid and in_segment:
            # End of current segment
            segments.append((start_idx, i))
            in_segment = False

    # Handle segment that extends to the end
    if in_segment:
        segments.append((start_idx, len(valid_mask)))

    return segments


def filter_motion_file(
    npz_path: str,
    output_dir: str,
    min_root_height: float,
    max_abs_roll: float,
    max_abs_pitch: float,
    buffer_duration: float,
    min_clip_duration: float,
) -> dict:
    """Filter a single motion NPZ file and save valid clips.

    Args:
        npz_path: Path to input NPZ file.
        output_dir: Directory to save filtered clips.
        min_root_height: Minimum root z-position (meters).
        max_abs_roll: Maximum absolute roll angle (radians).
        max_abs_pitch: Maximum absolute pitch angle (radians).
        buffer_duration: Duration to remove around invalid frames (seconds).
        min_clip_duration: Minimum clip duration to keep (seconds).

    Returns:
        Dict with statistics about the filtering:
        - original_frames: Number of frames in original file
        - invalid_frames: Number of frames failing thresholds
        - buffered_frames: Number of additional frames removed by buffer
        - output_clips: Number of clips saved
        - output_frames: Total frames in output clips
    """
    stats = {
        "original_frames": 0,
        "invalid_frames": 0,
        "buffered_frames": 0,
        "output_clips": 0,
        "output_frames": 0,
    }

    # Load motion data
    with np.load(npz_path) as data:
        body_pos_w = data["body_pos_w"]  # [num_frames, num_bodies, 3]
        body_quat_w = data["body_quat_w"]  # [num_frames, num_bodies, 4] in wxyz
        body_lin_vel_w = data["body_lin_vel_w"]  # [num_frames, num_bodies, 3]
        body_ang_vel_w = data["body_ang_vel_w"]  # [num_frames, num_bodies, 3]
        joint_pos = data["joint_pos"]  # [num_frames, num_joints]
        joint_vel = data["joint_vel"]  # [num_frames, num_joints]

        # Get FPS
        if "fps" in data:
            fps_val = data["fps"]
            fps = int(fps_val[0]) if isinstance(fps_val, np.ndarray) else int(fps_val)
        else:
            fps = 50  # Default to 50 Hz

    num_frames = body_pos_w.shape[0]
    stats["original_frames"] = num_frames

    # Extract root state (body index 0)
    root_height = body_pos_w[:, 0, 2]  # z-position
    root_quat = body_quat_w[:, 0, :]  # quaternion wxyz

    # Convert quaternion to Euler angles
    roll, pitch, _ = quat_to_euler(root_quat)

    # Find frames that fail thresholds
    height_invalid = root_height < min_root_height
    roll_invalid = np.abs(roll) > max_abs_roll
    pitch_invalid = np.abs(pitch) > max_abs_pitch

    # Combine invalid masks
    invalid_mask = height_invalid | roll_invalid | pitch_invalid
    stats["invalid_frames"] = int(np.sum(invalid_mask))

    # Expand invalid regions by buffer duration
    buffer_frames = int(buffer_duration * fps)
    if buffer_frames > 0:
        # Create expanded invalid mask
        expanded_invalid = invalid_mask.copy()
        invalid_indices = np.where(invalid_mask)[0]

        for idx in invalid_indices:
            start = max(0, idx - buffer_frames)
            end = min(num_frames, idx + buffer_frames + 1)
            expanded_invalid[start:end] = True

        stats["buffered_frames"] = int(np.sum(expanded_invalid)) - stats["invalid_frames"]
        valid_mask = ~expanded_invalid
    else:
        valid_mask = ~invalid_mask

    # Find contiguous valid segments
    segments = find_valid_segments(valid_mask)

    # Filter segments by minimum duration
    min_clip_frames = int(min_clip_duration * fps)
    valid_segments = [(start, end) for start, end in segments if (end - start) >= min_clip_frames]

    if not valid_segments:
        return stats

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each valid segment as a separate clip
    base_name = Path(npz_path).stem

    for clip_idx, (start, end) in enumerate(valid_segments):
        clip_data = {
            "fps": np.array([fps]),
            "body_pos_w": body_pos_w[start:end],
            "body_quat_w": body_quat_w[start:end],
            "body_lin_vel_w": body_lin_vel_w[start:end],
            "body_ang_vel_w": body_ang_vel_w[start:end],
            "joint_pos": joint_pos[start:end],
            "joint_vel": joint_vel[start:end],
        }

        # Name clips: original_clip0.npz, original_clip1.npz, etc.
        # If only one clip and it's the full motion, keep original name
        if len(valid_segments) == 1 and start == 0 and end == num_frames:
            clip_name = f"{base_name}.npz"
        else:
            clip_name = f"{base_name}_clip{clip_idx}.npz"

        output_path = os.path.join(output_dir, clip_name)
        np.savez(output_path, **clip_data)

        stats["output_clips"] += 1
        stats["output_frames"] += end - start

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter motion data to remove infeasible frames (low height, extreme roll/pitch)."
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name (e.g., LAFAN1_Retargeting_Dataset, amass_g1)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        help="Robot type (e.g., g1). Default: g1",
    )

    # Config file (optional - overrides defaults)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file with filter parameters",
    )

    # Direct threshold overrides (override config file if specified)
    parser.add_argument(
        "--min_root_height",
        type=float,
        default=None,
        help="Minimum root z-position in meters (reject if below)",
    )
    parser.add_argument(
        "--max_abs_roll",
        type=float,
        default=None,
        help="Maximum absolute roll angle in radians",
    )
    parser.add_argument(
        "--max_abs_pitch",
        type=float,
        default=None,
        help="Maximum absolute pitch angle in radians",
    )
    parser.add_argument(
        "--buffer_duration",
        type=float,
        default=None,
        help="Buffer duration in seconds to remove around invalid frames",
    )
    parser.add_argument(
        "--min_clip_duration",
        type=float,
        default=None,
        help="Minimum clip duration in seconds to keep",
    )

    # Dry run mode
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Analyze files without saving output (for previewing filter effects)",
    )

    args = parser.parse_args()

    # Default filter parameters
    params = {
        "min_root_height": 0.4,
        "max_abs_roll": 0.79,
        "max_abs_pitch": 0.79,
        "buffer_duration": 1.0,
        "min_clip_duration": 1.0,
    }

    # Load from config file if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.resolve()
            config_path = script_dir / args.config
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {args.config}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
            for key in params:
                if key in config:
                    params[key] = config[key]
        print(f"Loaded config from: {config_path}")

    # Override with command-line arguments if specified
    if args.min_root_height is not None:
        params["min_root_height"] = args.min_root_height
    if args.max_abs_roll is not None:
        params["max_abs_roll"] = args.max_abs_roll
    if args.max_abs_pitch is not None:
        params["max_abs_pitch"] = args.max_abs_pitch
    if args.buffer_duration is not None:
        params["buffer_duration"] = args.buffer_duration
    if args.min_clip_duration is not None:
        params["min_clip_duration"] = args.min_clip_duration

    # Determine paths
    # Use DATASET_DIR env variable if set, otherwise default to script's parent directory
    dataset_base_dir = os.environ.get("DATASET_DIR")
    if dataset_base_dir:
        dataset_base_dir = Path(dataset_base_dir).resolve()
        print(f"Using DATASET_DIR from environment: {dataset_base_dir}")
    else:
        script_dir = Path(__file__).parent.resolve()
        dataset_base_dir = script_dir.parent
        print(f"Using default dataset directory: {dataset_base_dir}")

    dataset_dir = dataset_base_dir / args.dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    input_dir = dataset_dir / "processed" / args.robot
    output_dir = dataset_dir / "filtered" / args.robot

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all NPZ files (recursively to preserve subfolder structure)
    npz_files = sorted(glob.glob(str(input_dir / "**" / "*.npz"), recursive=True))

    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return

    # Print configuration
    print("=" * 60)
    print("Motion Filtering Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Robot: {args.robot}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(npz_files)}")
    print()
    print("Filter Parameters:")
    print(f"  min_root_height: {params['min_root_height']} m")
    print(f"  max_abs_roll: {params['max_abs_roll']} rad ({np.degrees(params['max_abs_roll']):.1f} deg)")
    print(f"  max_abs_pitch: {params['max_abs_pitch']} rad ({np.degrees(params['max_abs_pitch']):.1f} deg)")
    print(f"  buffer_duration: {params['buffer_duration']} s")
    print(f"  min_clip_duration: {params['min_clip_duration']} s")
    print()
    if args.dry_run:
        print("DRY RUN MODE - No files will be saved")
        print()
    print("=" * 60)

    # Process each file
    total_stats = {
        "original_frames": 0,
        "invalid_frames": 0,
        "buffered_frames": 0,
        "output_clips": 0,
        "output_frames": 0,
        "files_processed": 0,
        "files_with_output": 0,
        "files_fully_filtered": 0,
    }

    for idx, npz_path in enumerate(npz_files, 1):
        # Compute relative path to preserve subfolder structure
        rel_path = Path(npz_path).relative_to(input_dir)
        file_output_dir = output_dir / rel_path.parent

        _ = Path(npz_path).name
        print(f"[{idx}/{len(npz_files)}] {rel_path}...", end=" ")

        try:
            if args.dry_run:
                # Still analyze but don't save
                stats = filter_motion_file(
                    npz_path=npz_path,
                    output_dir="/tmp/filter_dry_run",  # Temporary, won't actually save in dry run
                    min_root_height=params["min_root_height"],
                    max_abs_roll=params["max_abs_roll"],
                    max_abs_pitch=params["max_abs_pitch"],
                    buffer_duration=params["buffer_duration"],
                    min_clip_duration=params["min_clip_duration"],
                )
                # In dry run, don't actually count output since we're not saving
            else:
                stats = filter_motion_file(
                    npz_path=npz_path,
                    output_dir=str(file_output_dir),
                    min_root_height=params["min_root_height"],
                    max_abs_roll=params["max_abs_roll"],
                    max_abs_pitch=params["max_abs_pitch"],
                    buffer_duration=params["buffer_duration"],
                    min_clip_duration=params["min_clip_duration"],
                )

            # Update totals
            for key in ["original_frames", "invalid_frames", "buffered_frames", "output_clips", "output_frames"]:
                total_stats[key] += stats[key]
            total_stats["files_processed"] += 1

            if stats["output_clips"] > 0:
                total_stats["files_with_output"] += 1
                retention = stats["output_frames"] / stats["original_frames"] * 100
                print(
                    f"OK ({stats['output_clips']} clips, {stats['output_frames']}/{stats['original_frames']} frames,"
                    f" {retention:.1f}% retained)"
                )
            else:
                total_stats["files_fully_filtered"] += 1
                print(f"FILTERED OUT ({stats['invalid_frames']} invalid frames)")

        except Exception as e:
            print(f"ERROR: {e}")

    # Compute statistics
    original_frames = total_stats["original_frames"]
    output_frames = total_stats["output_frames"]
    rejected_frames = original_frames - output_frames

    if original_frames > 0:
        retention_pct = output_frames / original_frames * 100
        rejection_pct = rejected_frames / original_frames * 100
    else:
        retention_pct = 0.0
        rejection_pct = 0.0

    # Compute duration in hours (assuming 50 Hz = 0.02s per frame)
    fps = 50  # Default FPS
    original_hours = (original_frames / fps) / 3600
    output_hours = (output_frames / fps) / 3600
    rejected_hours = (rejected_frames / fps) / 3600

    # Print summary
    print()
    print("=" * 60)
    print("Filtering Summary")
    print("=" * 60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Files with output: {total_stats['files_with_output']}")
    print(f"Files fully filtered out: {total_stats['files_fully_filtered']}")
    print()
    print(f"Original frames: {original_frames:,} ({original_hours:.2f} hours)")
    print(f"Invalid frames: {total_stats['invalid_frames']:,}")
    print(f"Buffered frames: {total_stats['buffered_frames']:,}")
    print(f"Output frames: {output_frames:,} ({output_hours:.2f} hours) - {retention_pct:.1f}% preserved")
    print(f"Rejected frames: {rejected_frames:,} ({rejected_hours:.2f} hours) - {rejection_pct:.1f}% rejected")

    print()
    if not args.dry_run:
        print(f"Filtered motions saved to: {output_dir}")

        # Generate summary text file
        summary_path = output_dir / "filter_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Motion Filtering Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Dataset: {args.dataset}\n")
            f.write(f"  Robot: {args.robot}\n")
            f.write(f"  min_root_height: {params['min_root_height']} m\n")
            f.write(f"  max_abs_roll: {params['max_abs_roll']} rad ({np.degrees(params['max_abs_roll']):.1f} deg)\n")
            f.write(f"  max_abs_pitch: {params['max_abs_pitch']} rad ({np.degrees(params['max_abs_pitch']):.1f} deg)\n")
            f.write(f"  buffer_duration: {params['buffer_duration']} s\n")
            f.write(f"  min_clip_duration: {params['min_clip_duration']} s\n\n")

            f.write("File Statistics:\n")
            f.write(f"  Files processed: {total_stats['files_processed']}\n")
            f.write(f"  Files with output: {total_stats['files_with_output']}\n")
            f.write(f"  Files fully filtered out: {total_stats['files_fully_filtered']}\n\n")

            f.write("Frame Statistics:\n")
            f.write(f"  Original frames: {original_frames:,}\n")
            f.write(f"  Output frames: {output_frames:,}\n")
            f.write(f"  Rejected frames: {rejected_frames:,}\n")
            f.write(f"    - Invalid (threshold): {total_stats['invalid_frames']:,}\n")
            f.write(f"    - Buffered: {total_stats['buffered_frames']:,}\n\n")

            f.write("Retention:\n")
            f.write(f"  Preserved: {retention_pct:.2f}%\n")
            f.write(f"  Rejected: {rejection_pct:.2f}%\n\n")

            f.write("Duration (at 50 Hz):\n")
            f.write(f"  Original: {original_hours:.2f} hours ({original_hours * 60:.1f} minutes)\n")
            f.write(f"  Output: {output_hours:.2f} hours ({output_hours * 60:.1f} minutes)\n")
            f.write(f"  Rejected: {rejected_hours:.2f} hours ({rejected_hours * 60:.1f} minutes)\n")
            f.write("=" * 60 + "\n")

        print(f"Summary saved to: {summary_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
