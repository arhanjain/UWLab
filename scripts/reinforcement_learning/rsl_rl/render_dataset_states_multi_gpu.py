#!/usr/bin/env python3
# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-GPU launcher for render_dataset_states.py.

It shards episode indices across detected GPUs and spawns one subprocess per GPU,
pinning each worker with CUDA_VISIBLE_DEVICES. Each worker writes its logs and
intermediate observations into a per-worker directory, and the launcher merges
those intermediates back into a single HDF5 once all workers finish.

Example:
  python scripts/reinforcement_learning/rsl_rl/render_dataset_states_multi_gpu.py \
    --hdf5_file rollout_dataset/cube_on_plate_1k.hdf5 \
    --task OmniReset-DROID-IkRelative-Vision-Play-v0 \
    --gpus auto \
    -- \
    --num_envs 1 --headless --enable_cameras \
    env.scene.insertive_object=cube env.scene.receptive_object=plate

Notes:
  - Workers never write directly to the main HDF5; they dump .npz files containing
    vision observations. The parent process then merges those into the HDF5, which
    avoids concurrent writers.
  - Each run gets its own directory, with one subdirectory per worker containing
    logs and intermediate observation files.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def _detect_gpu_count() -> int:
    """Detect number of NVIDIA GPUs available (best-effort)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if r.returncode == 0:
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith("GPU ")]
            if lines:
                return len(lines)
    except FileNotFoundError:
        pass

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _parse_gpu_list(gpus: str, detected: int) -> list[int]:
    if gpus in ("auto", ""):
        return list(range(detected))
    out: list[int] = []
    for part in gpus.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    # de-dupe preserving order
    seen = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    # filter to non-negative
    return [x for x in uniq if x >= 0]


def _chunk_evenly(items: list[int], k: int) -> list[list[int]]:
    if k <= 0:
        return []
    chunks = [[] for _ in range(k)]
    for i, it in enumerate(items):
        chunks[i % k].append(it)
    return chunks


def _get_episode_count_and_names(hdf5_file: str) -> tuple[int, list[str]]:
    with h5py.File(hdf5_file, "r") as f:
        if "data" not in f:
            return 0, []
        episode_names = list(f["data"].keys())
        return len(episode_names), episode_names


@dataclass(frozen=True)
class WorkerSpec:
    gpu_id: int
    episode_indices: list[int]
    episode_names: list[str]
    work_dir: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch render_dataset_states.py across multiple GPUs by sharding episodes."
    )
    parser.add_argument("--hdf5_file", type=str, required=True, help="Path to the HDF5 dataset file.")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task id/name (forwarded to worker). Not required with --merge_only.",
    )
    parser.add_argument("--gpus", type=str, default="auto", help="GPU ids, e.g. 'auto', '0,1,3', '0-3'.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Total workers to spawn. Default 0 => one per GPU listed.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode indices to process: 'all' or e.g. '0,1,2,10-19'.",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default=None,
        help="Root directory for run outputs (logs + intermediates). Defaults to ./render_dataset_runs.",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        default=False,
        help=(
            "Skip launching workers and only merge intermediate .npz observation dumps into the HDF5. "
            "Requires --run_dir pointing at an existing run directory."
        ),
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run directory to merge from when using --merge_only (contains worker_*/obs/*.npz).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print worker commands and exit without launching.",
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Everything after '--' is forwarded to render_dataset_states.py (including Hydra overrides).",
    )

    args = parser.parse_args()

    if args.merge_only:
        if not args.run_dir:
            raise SystemExit("--merge_only requires --run_dir pointing at an existing run directory.")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise SystemExit(f"--run_dir does not exist: {run_dir}")
        workers: list[WorkerSpec] = []
        for wdir in sorted(run_dir.glob("worker_*")):
            obs_dir = wdir / "obs"
            if obs_dir.exists():
                workers.append(WorkerSpec(gpu_id=-1, episode_indices=[], episode_names=[], work_dir=wdir))
        if not workers:
            raise SystemExit(f"No worker directories with obs/ found under: {run_dir}")

        # Merge intermediate .npz observation files back into main HDF5.
        print(f"Merging intermediate observations into HDF5 from run_dir: {run_dir}")
        merged = 0
        open_deadline_s = time.time() + 120.0
        last_err: Exception | None = None
        h5f = None
        while time.time() < open_deadline_s:
            try:
                h5f = h5py.File(args.hdf5_file, "r+")
                break
            except BlockingIOError as e:
                last_err = e
                time.sleep(0.5)
            except OSError as e:
                last_err = e
                msg = str(e).lower()
                if "unable to lock file" in msg or "resource temporarily unavailable" in msg:
                    time.sleep(0.5)
                else:
                    raise

        if h5f is None:
            raise SystemExit(
                f"Failed to open HDF5 for merge (still locked after 120s): {args.hdf5_file}\n"
                f"Underlying error: {last_err}\n"
                "This usually means another process still has the dataset open. Ensure:\n"
                "  - no other render/train/viz job is using the same .hdf5\n"
                "  - all worker processes from a previous run have exited\n"
                "If you understand the risks, you can disable HDF5 locking by setting:\n"
                "  HDF5_USE_FILE_LOCKING=FALSE\n"
                "before running this script."
            )

        with h5f:
            for w in workers:
                obs_dir = w.work_dir / "obs"
                for npz_path in sorted(obs_dir.glob("*.npz")):
                    episode_name = npz_path.stem.removesuffix("_vision")
                    if "data" not in h5f or episode_name not in h5f["data"]:
                        print(f"  [merge] Episode {episode_name} not found in HDF5, skipping {npz_path}")
                        continue
                    episode_path = f"data/{episode_name}"
                    ep_group = h5f[episode_path]
                    if "obs" not in ep_group:
                        ep_group.create_group("obs")
                    obs_group = ep_group["obs"]
                    if "vision" not in obs_group:
                        obs_group.create_group("vision")
                    vision_group = obs_group["vision"]

                    with np.load(npz_path) as data:
                        for camera_name in data.files:
                            if camera_name in vision_group:
                                del vision_group[camera_name]
                            vision_group.create_dataset(camera_name, data=data[camera_name], compression="gzip")
                            print(f"  [merge] {episode_name}/{camera_name}: {data[camera_name].shape}")
                    merged += 1
            h5f.flush()

        print(f"Merge complete. Wrote observations for {merged} episode(s).")
        print(f"Run artifacts are in: {run_dir}")
        return 0

    detected = _detect_gpu_count()
    gpu_ids = _parse_gpu_list(args.gpus, detected)
    if not gpu_ids:
        raise SystemExit(
            f"No GPUs detected/selected (detected={detected}, gpus='{args.gpus}'). "
            "If you intend to run on CPU, just run render_dataset_states.py directly."
        )
    if not args.task:
        raise SystemExit("--task is required unless you use --merge_only.")

    num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else len(gpu_ids)
    if num_workers <= 0:
        raise SystemExit("num_workers must be > 0")

    # Assign workers to GPUs round-robin if num_workers > len(gpu_ids)
    worker_gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(num_workers)]

    # Figure out which episode indices to shard.
    episode_count, episode_names = _get_episode_count_and_names(args.hdf5_file)
    if episode_count <= 0:
        raise SystemExit(f"No episodes found in dataset or missing 'data' group: {args.hdf5_file}")

    if args.episodes == "all":
        episode_indices = list(range(episode_count))
    else:
        episode_indices = _parse_gpu_list(args.episodes, detected=0)  # same parser supports ranges
        episode_indices = [i for i in episode_indices if 0 <= i < episode_count]
        if not episode_indices:
            raise SystemExit(f"No valid episode indices after filtering (episode_count={episode_count}).")

    shards = _chunk_evenly(episode_indices, num_workers)

    # Prepare run directories
    if args.run_root is None:
        run_root = Path.cwd() / "render_dataset_runs"
    else:
        run_root = Path(args.run_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    workers: list[WorkerSpec] = []
    for i in range(num_workers):
        wdir = run_dir / f"worker_{i}"
        wdir.mkdir(parents=True, exist_ok=True)
        idxs = shards[i]
        names = [episode_names[j] for j in idxs] if idxs else []
        workers.append(WorkerSpec(gpu_id=worker_gpu_ids[i], episode_indices=idxs, episode_names=names, work_dir=wdir))

    # Worker script path relative to this file.
    worker_script = os.path.join(os.path.dirname(__file__), "render_dataset_states.py")

    # Parse remainder: allow optional leading '--'
    remainder = list(args.remainder)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]

    procs: list[tuple[subprocess.Popen, WorkerSpec]] = []
    for wi, w in enumerate(workers):
        if not w.episode_indices:
            continue

        obs_dir = w.work_dir / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            worker_script,
            "--hdf5_file",
            args.hdf5_file,
            "--task",
            args.task,
            "--device",
            "cuda:0",
            "--obs_dump_dir",
            str(obs_dir),
            "--select_episodes",
            *[str(x) for x in w.episode_indices],
        ]
        cmd.extend(remainder)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(w.gpu_id)

        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"[worker {wi}] GPU={w.gpu_id} episodes={len(w.episode_indices)}")
        print(f"[worker {wi}] {printable}")

        if args.dry_run:
            continue

        log_path = w.work_dir / "worker.log"
        log_f = open(log_path, "w", buffering=1)
        procs.append(
            (
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                ),
                w,
            )
        )

    if args.dry_run:
        return 0

    if not procs:
        print("No workers launched (no episodes assigned).")
        return 0

    # Wait for all workers; return non-zero if any failed.
    rc = 0
    for p, _w in procs:
        r = p.wait()
        rc = rc or r

    # Merge intermediate .npz observation files back into main HDF5.
    # This is done in a single process to avoid concurrent writers.
    print("Merging intermediate observations into HDF5...")
    merged = 0
    # HDF5 uses file locks. Opening with "r+" requires an exclusive lock, so if any
    # worker (or another run) still has the file open, this can fail transiently.
    # We retry briefly to accommodate slow shutdown/cleanup, and print actionable
    # guidance if the lock is persistent.
    open_deadline_s = time.time() + 120.0
    last_err: Exception | None = None
    h5f = None
    while time.time() < open_deadline_s:
        try:
            h5f = h5py.File(args.hdf5_file, "r+")
            break
        except BlockingIOError as e:
            last_err = e
            time.sleep(0.5)
        except OSError as e:
            # Some HDF5 builds raise generic OSError for locking issues.
            last_err = e
            msg = str(e).lower()
            if "unable to lock file" in msg or "resource temporarily unavailable" in msg:
                time.sleep(0.5)
            else:
                raise

    if h5f is None:
        raise SystemExit(
            f"Failed to open HDF5 for merge (still locked after 120s): {args.hdf5_file}\n"
            f"Underlying error: {last_err}\n"
            "This usually means another process still has the dataset open. Ensure:\n"
            "  - no other render/train/viz job is using the same .hdf5\n"
            "  - all worker processes from a previous run have exited\n"
            "If you understand the risks, you can disable HDF5 locking by setting:\n"
            "  HDF5_USE_FILE_LOCKING=FALSE\n"
            "before running this script."
        )

    with h5f:
        for _p, w in procs:
            obs_dir = w.work_dir / "obs"
            if not obs_dir.exists():
                continue
            for npz_path in sorted(obs_dir.glob("*.npz")):
                episode_name = npz_path.stem.removesuffix("_vision")
                if "data" not in h5f or episode_name not in h5f["data"]:
                    print(f"  [merge] Episode {episode_name} not found in HDF5, skipping {npz_path}")
                    continue
                episode_path = f"data/{episode_name}"
                ep_group = h5f[episode_path]
                if "obs" not in ep_group:
                    ep_group.create_group("obs")
                obs_group = ep_group["obs"]
                if "vision" not in obs_group:
                    obs_group.create_group("vision")
                vision_group = obs_group["vision"]

                with np.load(npz_path) as data:
                    for camera_name in data.files:
                        if camera_name in vision_group:
                            del vision_group[camera_name]
                        vision_group.create_dataset(camera_name, data=data[camera_name], compression="gzip")
                        print(f"  [merge] {episode_name}/{camera_name}: {data[camera_name].shape}")
                merged += 1
        h5f.flush()

    print(f"Merge complete. Wrote observations for {merged} episode(s).")
    print(f"Run artifacts are in: {run_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


