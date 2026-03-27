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

"""This script replays a motion from an AMASS G1 .npz file (IsaacLab AMP format) and outputs it to an enriched npz file.

The input format is the IsaacLab AMP format with keys: fps, root_pos, root_rot, dof_pos
The output format includes FK-computed body kinematics: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

.. code-block:: bash

    # Usage (just specify the filename - script will find it in amass_g1 dataset)
    python npz_to_npz.py --input_file motion_001.npz --output_name motion_001 --robot g1

    # Or specify a full path if the file is elsewhere
    python npz_to_npz.py --input_file /path/to/custom/file.npz --output_name custom_motion --robot g1

    # The script will automatically look recursively in the amass_g1 directory
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


def resolve_input_file(input_file: str) -> str:
    """Resolves the input file path.

    If the input_file is just a filename (no path separators), automatically
    looks for it in the amass_g1 dataset directory relative to this script.
    Otherwise, uses the provided path as-is.

    Args:
        input_file: The input file path or filename.

    Returns:
        The resolved absolute path to the input file.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # If it's already an absolute path or contains path separators, check if it exists
    if os.path.isabs(input_file) or os.sep in input_file or "/" in input_file:
        if os.path.exists(input_file):
            return os.path.abspath(input_file)
        else:
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Otherwise, it's just a filename - look in the dataset directory
    # Script is at: scripts/npz_to_npz.py (or dataset/scripts/amass_g1/npz_to_npz.py)
    # Dataset is at: amass_g1/ (relative to repo root)
    script_dir = Path(__file__).parent.resolve()
    # Try repo-root-level amass_g1 first, then fall back to dataset layout
    dataset_dir = script_dir.parent / "amass_g1"
    if not dataset_dir.exists():
        dataset_dir = script_dir.parent.parent / "amass_g1"

    # Search recursively for the file
    import glob

    matches = glob.glob(str(dataset_dir / "**" / input_file), recursive=True)
    if matches:
        return matches[0]  # Return first match

    # Fall back to root dataset directory
    dataset_file = dataset_dir / input_file
    if dataset_file.exists():
        return str(dataset_file.resolve())
    else:
        error_msg = f"Input file '{input_file}' not found in dataset directory: {dataset_dir}"
        error_msg += "\nPlease ensure the file exists or provide a full path."
        raise FileNotFoundError(error_msg)


# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from AMASS G1 .npz file and output to enriched npz file.")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help=(
        "The input motion .npz file (IsaacLab AMP format). Can be just a filename (e.g., 'motion_001.npz') "
        "which will be automatically found in amass_g1 dataset, or a full path."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the output motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion. Default: 50.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="The robot type to use (currently only 'g1' is supported). Default: 'g1'. ",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Resolve input file path (automatically find in dataset directory if just filename)
args_cli.input_file = resolve_input_file(args_cli.input_file)
print(f"[INFO]: Using robot: {args_cli.robot}")
print(f"[INFO]: Using input file: {args_cli.input_file}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##


def get_robot_config(robot_name: str) -> ArticulationCfg:
    """Gets the robot configuration based on robot name.

    Args:
        robot_name: The name of the robot (e.g., 'g1').

    Returns:
        The robot articulation configuration.

    Raises:
        ValueError: If the robot name is not supported.
    """
    robot_name = robot_name.lower()
    if robot_name == "g1":
        from krafton_lab.assets.unitree import G1_BEYONDMIMIC_CFG

        return G1_BEYONDMIMIC_CFG
    else:
        raise ValueError(
            f"Unsupported robot: {robot_name}. Currently only 'g1' is supported for AMASS G1 dataset. "
            "If you need another robot, please add it to get_robot_config()."
        )


def get_robot_joint_names(robot_name: str) -> list[str]:
    """Gets the joint names for a robot.

    Args:
        robot_name: The name of the robot (e.g., 'g1').

    Returns:
        List of joint names in the order expected by the motion data.
    """
    robot_name = robot_name.lower()
    if robot_name == "g1":
        return [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    else:
        raise ValueError(f"Unsupported robot: {robot_name}. Currently only 'g1' is supported.")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation - will be set dynamically based on robot
    robot: ArticulationCfg = MISSING


class MotionLoader:
    """Loads and manages motion data from AMASS G1 .npz files with optional fps interpolation."""

    def __init__(
        self,
        motion_file: str,
        output_fps: int,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the .npz file (AMASS G1 format)."""
        data = np.load(self.motion_file)

        # Debug: print available keys
        print(f"[DEBUG]: NPZ file keys: {list(data.keys())}")

        # AMASS G1 format has these keys:
        # - fps, dof_names, body_names
        # - dof_positions, dof_velocities
        # - body_positions, body_rotations, body_linear_velocities, body_angular_velocities

        # Extract root state from body data (index 0 is typically root/pelvis)
        body_positions = data["body_positions"]  # (N, num_bodies, 3)
        body_rotations = data["body_rotations"]  # (N, num_bodies, 4)

        # Root position and rotation (first body)
        self.motion_base_poss_input = torch.from_numpy(body_positions[:, 0, :]).to(torch.float32).to(self.device)
        motion_base_rots = torch.from_numpy(body_rotations[:, 0, :]).to(torch.float32).to(self.device)

        # Debug: print first quaternion to check format
        print(f"[DEBUG]: First root quaternion: {motion_base_rots[0].tolist()}")

        # AMASS G1 data appears to already be in wxyz format (Isaac Sim native)
        # Do NOT convert - use as-is
        self.motion_base_rots_input = motion_base_rots

        # Joint positions
        self.motion_dof_poss_input = torch.from_numpy(data["dof_positions"]).to(torch.float32).to(self.device)

        # Get fps - handle both scalar and array formats
        fps_data = data["fps"]
        if isinstance(fps_data, np.ndarray):
            self.input_fps = int(fps_data.item()) if fps_data.ndim == 0 else int(fps_data[0])
        else:
            self.input_fps = int(fps_data)
        self.input_dt = 1.0 / self.input_fps

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

        print(f"[INFO]: Motion loaded from {self.motion_file}")
        print(
            f"[INFO]: Duration: {self.duration:.2f} sec, Input frames: {self.input_frames}, Input FPS: {self.input_fps}"
        )
        print(f"[INFO]: DOF count: {self.motion_dof_poss_input.shape[1]}")
        print(f"[INFO]: Body count: {body_positions.shape[1]}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.num_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[INFO]: Motion interpolated: {self.input_frames} frames @ {self.input_fps} fps -> "
            f"{self.num_frames} frames @ {self.output_fps} fps"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.num_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def visualize_body_positions(sim: sim_utils.SimulationContext, log: dict):
    """Visualize accumulated body_pos_w data as spheres.

    Args:
        sim: Simulation context
        log: Dictionary containing logged data with 'body_pos_w' key
    """
    body_pos_w = log["body_pos_w"]  # Shape: [num_frames, num_bodies, 3]
    num_frames, num_bodies, _ = body_pos_w.shape

    # Create visualization markers
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/body_positions",
        markers={
            "body_sphere": sim_utils.SphereCfg(
                radius=0.05,  # Adjust size as needed
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red spheres
            ),
        },
    )
    body_markers = VisualizationMarkers(marker_cfg)
    marker_indices = torch.zeros(num_bodies, dtype=torch.long, device=sim.device)

    # Convert to torch tensors
    body_positions_torch = torch.from_numpy(body_pos_w).to(sim.device)
    # Create identity quaternions for all bodies (wxyz format)
    body_orientations = torch.zeros(num_bodies, 4, device=sim.device)
    body_orientations[:, 0] = 1.0  # w = 1, x = y = z = 0

    print(f"[INFO]: Visualizing {num_frames} frames with {num_bodies} bodies each...")

    # Visualize each frame
    frame_idx = 0
    while simulation_app.is_running() and frame_idx < num_frames:
        # Get positions for current frame
        current_positions = body_positions_torch[frame_idx, :, :]  # [num_bodies, 3]

        # Visualize spheres at body positions
        body_markers.visualize(current_positions, body_orientations, marker_indices=marker_indices)

        # Step simulation (for rendering)
        sim.render()
        sim.step()

        # Update camera to follow first body
        if num_bodies > 0:
            first_body_pos = current_positions[0, :].cpu().numpy()
            sim.set_camera_view(first_body_pos + np.array([2.0, 2.0, 0.5]), first_body_pos)

        frame_idx += 1

        # Optional: Add small delay or frame skipping for slower playback
        # Can be controlled via sim.step() frequency


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- real-time visualization markers ------------------------------------
    num_bodies = robot.data.body_pos_w.shape[1]
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/body_positions_realtime",
        markers={
            "body_sphere": sim_utils.SphereCfg(
                radius=0.05,  # Adjust size as needed
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    body_markers_realtime = VisualizationMarkers(marker_cfg)
    marker_indices_realtime = torch.zeros(num_bodies, dtype=torch.long, device=sim.device)
    # Create identity quaternions for all bodies (wxyz format)
    body_orientations_realtime = torch.zeros(num_bodies, 4, device=sim.device)
    body_orientations_realtime[:, 0] = 1.0  # w = 1, x = y = z = 0
    # --------------------------------------------------------------------------

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    frame_count = 0
    # --------------------------------------------------------------------------

    print(f"[INFO]: Processing {motion.num_frames} frames...")

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # Visualize body positions in real-time
        if not file_saved:
            body_positions_realtime = robot.data.body_pos_w[0, :]  # Shape: [num_bodies, 3]
            body_markers_realtime.visualize(
                body_positions_realtime, body_orientations_realtime, marker_indices=marker_indices_realtime
            )

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
            frame_count += 1
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"[INFO]: Processed {frame_count}/{motion.num_frames} frames...")

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            # Save NPZ file to dataset/amass_g1/processed/<robot>/<output_name>.npz
            script_dir = Path(__file__).parent.resolve()
            dataset_dir = script_dir.parent.parent / "amass_g1"
            processed_dir = dataset_dir / "processed" / args_cli.robot
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Ensure output_name has .npz extension
            output_filename = args_cli.output_name
            if not output_filename.endswith(".npz"):
                output_filename += ".npz"

            output_path = processed_dir / output_filename
            np.savez(str(output_path), **log)
            print(f"[INFO]: Saved motion data to: {output_path}")

            # Upload to wandb registry
            import wandb

            COLLECTION = "amass_g1_" + args_cli.output_name
            run = wandb.init(project="npz_to_npz", name=COLLECTION)
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motion"
            logged_artifact = run.log_artifact(artifact_or_path=str(output_path), name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

            # Visualize accumulated body positions (skip in headless mode)
            if not args_cli.headless:
                print("[INFO]: Data collection complete. Starting visualization...")
                visualize_body_positions(sim, log)
                print("[INFO]: Visualization complete. Closing simulation...")
            else:
                print("[INFO]: Data collection complete. Skipping visualization (headless mode).")

            import sys
            sys.exit(0)


def main():
    """Main function."""
    # Get robot configuration and joint names
    robot_cfg = get_robot_config(args_cli.robot)
    joint_names = get_robot_joint_names(args_cli.robot)

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)

    # Use output_fps for simulation timestep
    sim_cfg.dt = 1.0 / args_cli.output_fps
    print(f"[INFO]: Output FPS: {args_cli.output_fps}, Simulation dt: {sim_cfg.dt}")

    sim = SimulationContext(sim_cfg)
    # Design scene with robot-specific configuration
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=joint_names,
    )


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app
        if simulation_app.is_running():
            simulation_app.close()
