# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to render videos from an HDF5 dataset using camera observations.

This script loads camera observations directly from a dataset collected by collect_data.py
and renders them as video files.

Usage:
    python render_dataset.py --dataset_file rollout_dataset/cubestack.hdf5
"""

import argparse
import h5py
import os
import random

import mediapy as media
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Render videos from HDF5 dataset using camera observations.")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset file.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save videos. Defaults to dataset directory.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="List of episode indices to render. If empty, randomly samples 10 episodes.",
)
parser.add_argument("--framerate", type=int, default=30, help="Video framerate.")
parser.add_argument("--video_width", type=int, default=1280, help="Output video width.")
parser.add_argument("--video_height", type=int, default=720, help="Output video height.")

# parse the arguments
args_cli = parser.parse_args()


class VideoRecorder:
    """Helper class to write video frames to MP4 files using mediapy."""

    def __init__(self, output_path: str, width: int, height: int, framerate: int = 30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video buffer."""
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        # Resize if needed
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            # Use mediapy for resizing
            frame = media.resize_image(frame, (self.height, self.width))
        self.frames.append(frame)

    def save(self):
        """Save all frames to an MP4 file using mediapy."""
        if not self.frames:
            print(f"Warning: No frames to save for {self.output_path}")
            return

        # Stack frames and write using mediapy
        video_array = np.stack(self.frames, axis=0)
        media.write_video(self.output_path, video_array, fps=self.framerate)
        print(f"Saved: {self.output_path} ({len(self.frames)} frames)")


def get_episode_names(hdf5_file: str) -> list[str]:
    """Get list of episode names in the HDF5 file."""
    with h5py.File(hdf5_file, "r") as f:
        if "data" not in f:
            return []
        return list(f["data"].keys())  # type: ignore[union-attr]


def load_episode_camera_observations(hdf5_file: str, episode_name: str, camera_names: list[str]) -> dict[str, np.ndarray]:
    """Load camera observations for an episode from the HDF5 file.
    
    Args:
        hdf5_file: Path to the HDF5 dataset file.
        episode_name: Name of the episode.
        camera_names: List of camera names to load (e.g., ['external_camera', 'wrist_camera']).
    
    Returns:
        Dictionary mapping camera_name -> array of shape (num_steps, height, width, channels).
    """
    camera_obs: dict[str, np.ndarray] = {}
    
    with h5py.File(hdf5_file, "r") as f:
        obs_path = f"data/{episode_name}/obs"
        if obs_path not in f:
            return camera_obs
        
        obs_group = f[obs_path]
        obs_keys = list(obs_group.keys())  # type: ignore[union-attr]
        if "vision" not in obs_keys:
            return camera_obs
        
        vision_group = obs_group["vision"]  # type: ignore[index]
        vision_keys = list(vision_group.keys())  # type: ignore[union-attr]
        for camera_name in camera_names:
            if camera_name in vision_keys:
                data = np.array(vision_group[camera_name])  # type: ignore[index]
                camera_obs[camera_name] = data
    
    return camera_obs


def get_available_cameras(hdf5_file: str, episode_name: str) -> list[str]:
    """Get list of available camera observations in an episode.
    
    Args:
        hdf5_file: Path to the HDF5 dataset file.
        episode_name: Name of the episode.
    
    Returns:
        List of available camera names.
    """
    with h5py.File(hdf5_file, "r") as f:
        obs_path = f"data/{episode_name}/obs"
        if obs_path not in f:
            return []
        
        obs_group = f[obs_path]
        obs_keys = list(obs_group.keys())  # type: ignore[union-attr]
        if "vision" not in obs_keys:
            return []
        
        vision_group = obs_group["vision"]  # type: ignore[index]
        # Check for common camera names
        available_cameras = []
        vision_keys = list(vision_group.keys())  # type: ignore[union-attr]
        for camera_name in ["external_camera", "wrist_camera", "wrist_cam", "table_cam"]:
            if camera_name in vision_keys:
                available_cameras.append(camera_name)
        
        return available_cameras


def main():
    """Render videos for each episode in the dataset."""
    # Check dataset file exists
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args_cli.dataset_file}")
    
    # Get episode info
    episode_names = get_episode_names(args_cli.dataset_file)
    episode_count = len(episode_names)
    
    if episode_count == 0:
        print("No episodes found in the dataset.")
        return
    
    print(f"Found {episode_count} episodes in dataset")
    
    # Determine which episodes to render
    if args_cli.select_episodes:
        episode_indices = [i for i in args_cli.select_episodes if i < episode_count]
    else:
        # Randomly sample 10 episodes (or all if fewer than 10)
        num_to_sample = min(10, episode_count)
        episode_indices = sorted(random.sample(range(episode_count), num_to_sample))
        print(f"Randomly selected {len(episode_indices)} episode(s) to render: {episode_indices}")
    
    if not episode_indices:
        print("No valid episodes to render.")
        return
    
    # Set up output directory
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args_cli.dataset_file), "videos")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving videos to: {output_dir}")
    
    # Check which cameras are available in the first episode
    first_episode_name = episode_names[episode_indices[0]]
    available_cameras = get_available_cameras(args_cli.dataset_file, first_episode_name)
    
    # Check if both external_camera and wrist_camera are available
    has_external = "external_camera" in available_cameras
    has_wrist = "wrist_camera" in available_cameras
    
    if not (has_external and has_wrist):
        print("Warning: Both external_camera and wrist_camera must be available for side-by-side rendering.")
        if not available_cameras:
            print("No camera observations found in dataset. Cannot render videos.")
            return
        print(f"Available cameras: {available_cameras}")
        return
    
    print("Using external_camera and wrist_camera for side-by-side rendering")
    
    camera_names_to_use = ["external_camera", "wrist_camera"]
    rendered_count = 0
    
    for episode_idx in episode_indices:
        episode_name = episode_names[episode_idx]
        
        print(f"Rendering episode {episode_idx} ({episode_name})")
        
        # Load camera observations from dataset
        camera_obs = load_episode_camera_observations(args_cli.dataset_file, episode_name, camera_names_to_use)
        
        if not camera_obs or len(camera_obs) != 2:
            print(f"Warning: Could not load both camera observations for episode '{episode_name}', skipping")
            continue
        
        # Get the episode length from camera data
        episode_length = min(len(camera_obs["external_camera"]), len(camera_obs["wrist_camera"]))
        
        if episode_length == 0:
            print(f"Warning: No camera data found for episode '{episode_name}', skipping")
            continue
        
        print(f"  Found {episode_length} frames")
        
        # Create a single video recorder for the combined side-by-side video
        # Combined width will be 2x the specified width (each camera gets half)
        output_path = os.path.join(output_dir, f"{episode_name}_combined.mp4")
        video_recorder = VideoRecorder(
            output_path,
            args_cli.video_width * 2,  # Double width for side-by-side
            args_cli.video_height,
            args_cli.framerate
        )
        
        # Render each step by combining camera observations side by side
        for step_idx in range(episode_length):
            external_frame = camera_obs["external_camera"][step_idx].copy()
            wrist_frame = camera_obs["wrist_camera"][step_idx].copy()
            
            # Ensure frames are in correct format (H, W, C) and uint8
            if external_frame.dtype != np.uint8:
                if external_frame.max() <= 1.0:
                    external_frame = (external_frame * 255).astype(np.uint8)
                else:
                    external_frame = external_frame.astype(np.uint8)
            
            if wrist_frame.dtype != np.uint8:
                if wrist_frame.max() <= 1.0:
                    wrist_frame = (wrist_frame * 255).astype(np.uint8)
                else:
                    wrist_frame = wrist_frame.astype(np.uint8)
            
            # Resize each frame to half the target width and full height
            external_resized = media.resize_image(external_frame, (args_cli.video_height, args_cli.video_width))
            wrist_resized = media.resize_image(wrist_frame, (args_cli.video_height, args_cli.video_width))
            
            # Combine frames side by side (horizontally)
            combined_frame = np.concatenate([external_resized, wrist_resized], axis=1)
            
            video_recorder.add_frame(combined_frame)
        
        # Save the combined video
        video_recorder.save()
        rendered_count += 1
    
    plural_s = "s" if rendered_count != 1 else ""
    print(f"\nFinished rendering {rendered_count} episode{plural_s}.")


if __name__ == "__main__":
    main()
