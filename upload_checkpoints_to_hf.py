"""
Upload model_29999.pt checkpoints to a HuggingFace repository.

Organizes files as: beyondmimic/g1/LAFAN1/{motion_name}/model_29999.pt
e.g., beyondmimic/g1/LAFAN1/dance2_subject4/model_29999.pt

Usage:
    python upload_checkpoints_to_hf.py <hf_repo_id>
    # e.g., python upload_checkpoints_to_hf.py myuser/g1-whole-body-tracking
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


LOGS_DIR = Path(__file__).parent / "logs" / "rsl_rl" / "g1_flat"
CHECKPOINT_NAME = "model_29999.pt"
# Pattern: {date}_{time}_lafan1_{motion}_{subject}
FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_lafan1_(.+)$")


def collect_checkpoints() -> dict[str, Path]:
    """Return {motion_name: path_to_model_29999.pt} for all available checkpoints."""
    checkpoints = {}
    for folder in sorted(LOGS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        match = FOLDER_PATTERN.match(folder.name)
        if not match:
            continue
        model_path = folder / CHECKPOINT_NAME
        if model_path.exists():
            motion_name = match.group(1)  # e.g., "dance2_subject4"
            checkpoints[motion_name] = model_path
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace")
    parser.add_argument("repo_id", help="HuggingFace repo id (e.g., myuser/g1-checkpoints)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    checkpoints = collect_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints to upload:\n")
    for name in sorted(checkpoints):
        print(f"  beyondmimic/g1/LAFAN1/{name}/{CHECKPOINT_NAME}")

    print(f"\nUploading to: https://huggingface.co/{args.repo_id}\n")

    for name, local_path in sorted(checkpoints.items()):
        path_in_repo = f"beyondmimic/g1/LAFAN1/{name}/{CHECKPOINT_NAME}"
        print(f"  Uploading {path_in_repo} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
        )

    print(f"\nDone! {len(checkpoints)} files uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
