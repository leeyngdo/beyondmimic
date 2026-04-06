# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)
[[Checkpoints]](https://huggingface.co/KRAFTON/physical_ai_motion_tracking_experts)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository with submodules:

```bash
git clone --recurse-submodules git@github.com:HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
```

- Pull the robot description files from GCS

```bash
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

- Install TMR dependencies (for AMASS motion clustering)

```bash
pip install umap-learn hdbscan huggingface_hub
cd third_party/TMR && bash prepare/download_pretrain_models.sh && cd ../..
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

We leverage the WandB registry to store and load reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses):

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - [AMASS](https://amass.is.tue.mpg.de/) retargeted to G1 (see [AMASS Clustering](#amass-motion-clustering) below)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP)
    - Balance motions are from [HuB](https://hub-robot.github.io/)

- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics:

```bash
# LAFAN1 (CSV format)
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless

# AMASS G1 (NPZ format)
python scripts/npz_to_npz.py --input_file {motion_name}.npz --output_name {motion_name} --robot g1 --headless
```

- Batch preprocessing:

```bash
# Preprocess all LAFAN1 motions
bash batch_preprocess.sh

# Preprocess AMASS G1: npz_to_npz -> filter -> cluster
bash batch_preprocess_amass.sh
```

### Policy Training

- Train a single-motion expert:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

- Train all LAFAN1 per-motion experts (multi-GPU):

```bash
bash batch_train.sh
```

- Train AMASS cluster experts (multi-GPU, see [AMASS Clustering](#amass-motion-clustering)):

```bash
bash batch_train_clusters.sh amass_g1/cluster_mapping
```

### Policy Evaluation

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

### Pre-trained Checkpoints

Pre-trained checkpoints are available on [HuggingFace](https://huggingface.co/KRAFTON/physical_ai_motion_tracking_experts):

| Type | Robot | Dataset | Experts | Iterations |
|------|-------|---------|---------|------------|
| Per-motion | Unitree G1 | LAFAN1 | 40 | 30,000 |
| Cluster | Unitree G1 | AMASS (filtered) | 16 | 100,000 |

## AMASS Motion Clustering

We provide a pipeline to cluster the large-scale [AMASS](https://amass.is.tue.mpg.de/) dataset into semantically meaningful groups using [TMR](https://github.com/Mathux/TMR) (Text-Motion Retrieval) embeddings. This enables training one multi-motion expert per cluster instead of thousands of individual experts.

### Pipeline

```
Raw AMASS G1 (.npz)
    |  npz_to_npz.py (Forward Kinematics via Isaac Sim)
    v
Processed motions (body_pos_w, joint_pos, ...)
    |  filter_motions.py (reject infeasible frames: low height, extreme tilt)
    v
Filtered motions (18,424 clips from 17,596 files, 96.6% preserved)
    |  cluster_amass_tmr.py (TMR encoding + FPS + UMAP + HDBSCAN)
    v
16 semantic clusters
    |  merge_cluster_motions.py (concatenate all motions per cluster)
    v
Merged cluster NPZ files -> WandB registry -> batch_train_clusters.sh
```

### Clustering Method

1. **TMR Encoding**: Each motion is converted to HumanML3D 263-dim features (G1 skeleton mapped to SMPL-22 joints), then encoded into 256-dim embeddings using a pretrained TMR motion encoder.

2. **Density-Balanced Subsampling**: Farthest Point Sampling (FPS) selects 5,000 representative points from the embedding space. Dense regions (e.g., thousands of similar walking clips) are automatically thinned, while sparse regions (rare motions) are preserved.

3. **Clustering**: UMAP dimensionality reduction + HDBSCAN on the subsampled points produces cluster labels. All remaining points are assigned to the nearest cluster centroid.

### Resulting Clusters

| Cluster | Name | Motions | Description |
|---------|------|---------|-------------|
| 0 | jumping_acrobatic | 383 | Jumping, scampering, acrobatic movements |
| 1 | sitting_wiping | 645 | Sitting, wiping, upper-body tasks |
| 2 | fast_curved_walking | 379 | Fast S-shape and curved walking |
| 3 | running_sprinting | 358 | Running, sprinting, fast locomotion |
| 4 | push_recovery | 425 | Balance recovery from pushes |
| 5 | mixed_speed_walking | 790 | Variable-speed walking |
| 6 | lifting_knocking | 343 | Lifting objects, knocking |
| 7 | circular_walking | 300 | Circular and elliptical walking |
| 8 | jogging | 639 | Jogging and light running |
| 9 | diverse_actions | 1111 | Kicking, throwing, motorcycle, misc |
| 10 | normal_walking | 776 | Normal-speed straight walking |
| 11 | locomotion_general | 2322 | General locomotion and transitions |
| 12 | object_manipulation | 2613 | Grasping, pouring, object interaction |
| 13 | standing_gestures | 5759 | Standing poses, gestures, subtle movements |
| 14 | upper_body_sports | 1128 | Handball, sports throwing |
| 15 | throwing_dynamic | 453 | Dynamic throwing and fast upper-body |

### Clustering Scripts

```bash
# Step 1: Encode and cluster filtered motions
cd third_party/TMR  # TMR needs to be the working directory for model loading
python ../../scripts/cluster_amass_tmr.py \
    --amass_dir ../../amass_g1/filtered/g1 \
    --density_subsample 5000 \
    --output ../../amass_g1/clusters_filtered.npz

# Step 2: Build mapping files
python scripts/build_cluster_mapping.py \
    --clusters amass_g1/clusters_filtered.npz \
    --output_dir amass_g1/cluster_mapping

# Step 3: Merge motions per cluster
python scripts/merge_cluster_motions.py \
    --cluster_summary amass_g1/cluster_mapping/cluster_summary.json \
    --processed_dir amass_g1/processed/g1 \
    --output_dir amass_g1/cluster_motions

# Step 4: Upload to WandB and train
bash batch_train_clusters.sh amass_g1/cluster_mapping
```

## Code Structure

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`** - MDP atomic functions (commands, rewards, observations, terminations, events)
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`** - Environment hyperparameters
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`** - PPO hyperparameters
- **`source/whole_body_tracking/whole_body_tracking/robots`** - Robot-specific settings
- **`scripts/`** - Preprocessing, training, evaluation, and clustering scripts
- **`third_party/TMR`** - TMR submodule for semantic motion embeddings

### Key Scripts

| Script | Description |
|--------|-------------|
| `scripts/csv_to_npz.py` | LAFAN1 CSV to enriched NPZ (FK via Isaac Sim) |
| `scripts/npz_to_npz.py` | AMASS G1 NPZ to enriched NPZ (FK via Isaac Sim) |
| `scripts/filter_motions.py` | Remove physically infeasible frames |
| `scripts/cluster_amass_tmr.py` | TMR-based motion clustering |
| `scripts/build_cluster_mapping.py` | Generate cluster mapping JSONs |
| `scripts/merge_cluster_motions.py` | Merge cluster motions into single NPZ |
| `scripts/rsl_rl/train.py` | Policy training |
| `scripts/rsl_rl/play.py` | Policy evaluation |
| `batch_train.sh` | Batch train LAFAN1 per-motion experts |
| `batch_train_clusters.sh` | Batch train AMASS cluster experts |
| `batch_preprocess_amass.sh` | End-to-end AMASS preprocessing pipeline |
| `upload_checkpoints_to_hf.py` | Upload checkpoints to HuggingFace |
