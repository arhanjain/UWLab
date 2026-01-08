# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import hashlib
import io
import logging
import numpy as np
import os
import random
import tempfile
import torch
import trimesh
import yaml
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from urllib.parse import urlparse

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
import omni
import warp as wp
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.warp import convert_to_warp_mesh
from pxr import UsdGeom
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from .rigid_object_hasher import RigidObjectHasher

# Initialize logger
logger = logging.getLogger(__name__)

# Try to import boto3 for S3 access
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# ---- module-scope caches ----
_PRIM_SAMPLE_CACHE: dict[tuple[str, int], np.ndarray] = {}  # (prim_hash, num_points) -> (N,3) in root frame
_FINAL_SAMPLE_CACHE: dict[str, np.ndarray] = {}  # env_hash -> (num_points,3) in root frame


def clear_pointcloud_caches():
    _PRIM_SAMPLE_CACHE.clear()
    _FINAL_SAMPLE_CACHE.clear()


@lru_cache(maxsize=None)
def _load_mesh_tensors(prim):
    tm = prim_to_trimesh(prim)
    verts = torch.from_numpy(tm.vertices.astype("float32"))
    faces = torch.from_numpy(tm.faces.astype("int64"))
    return verts, faces


def sample_object_point_cloud(
    num_envs: int,
    num_points: int,
    prim_path_pattern: str,
    device: str = "cuda",  # assume GPU
    rigid_object_hasher: RigidObjectHasher | None = None,
    seed: int = 42,
) -> torch.Tensor | None:
    """Generating point cloud given the path regex expression. This methood samples point cloud on ALL colliders
    falls under the prim path pattern. It is robust even if there are different numbers of colliders under the same
    regex expression. e.g. envs_0/object has 2 colliders, while envs_1/object has 4 colliders. This method will ensure
    each object has exactly num_points pointcloud regardless of number of colliders. If detected 0 collider, this method
    will return None, indicating no pointcloud can be sampled.

    To save memory and time, this method utilize RigidObjectHasher to make sure collider that hash to the same key will
    only be sampled once. It worths noting there are two kinds of hash:

    collider hash, and root hash. As name suggest, collider hash describes the uniqueness of collider from the view of root,
    collider hash is generated at atomic level and can not be representing aggregated. The root hash describes the
    uniqueness of aggregate of root, and can be hash that represent aggregate of multiple components that composes root.

    Be mindful that root's transform: translation, quaternion, scale, do no account for root's hash

    Args:
        num_envs (int): _description_
        num_points (int): _description_
        prim_path_pattern (str): _description_
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        torch.Tensor | None: _description_
    """
    hasher = (
        rigid_object_hasher
        if rigid_object_hasher is not None
        else RigidObjectHasher(num_envs, prim_path_pattern, device=device)
    )

    if hasher.num_root == 0:
        return None

    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    if replicated_env:
        # Pick env 0’s colliders
        mask_env0 = hasher.collider_prim_env_ids == 0
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p, m in zip(hasher.collider_prims, mask_env0) if m])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms[mask_env0]
    else:
        # Build all envs's colliders
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p in hasher.collider_prims])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms
    with temporary_seed(seed):
        # Uniform‐surface sample then scale to root
        samp = sample_points_from_meshes(meshes, num_points * 2)
        local, _ = sample_farthest_points(samp, K=num_points)
        t_rel, q_rel, s_rel = rel_tf[:, :3].unsqueeze(1), rel_tf[:, 3:7].unsqueeze(1), rel_tf[:, 7:].unsqueeze(1)
        # here is apply_forward not apply_inverse, because when mesh loaded, it is unscaled. But inorder to view it from
        # root, you need to apply forward transformation of root->child, which is exactly tqs_root_child.
        root = math_utils.quat_apply(q_rel.expand(-1, num_points, -1), local * s_rel) + t_rel

        # Merge Colliders
        if replicated_env:
            buf = root.reshape(1, -1, 3)
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged.view(1, num_points, 3).expand(num_envs, -1, -1) * hasher.root_prim_scales.unsqueeze(1)
        else:
            # 4) Scatter each collider into a padded per‐root buffer
            env_ids = hasher.collider_prim_env_ids.to(device)  # (M,)
            counts = torch.bincount(env_ids, minlength=hasher.num_root)  # (num_root,)
            max_c = int(counts.max().item())
            buf = torch.zeros((hasher.num_root, max_c * num_points, 3), device=device, dtype=root.dtype)
            # track how many placed in each root
            placed = torch.zeros_like(counts)
            for i in range(len(hasher.collider_prims)):
                r = int(env_ids[i].item())
                start = placed[r].item() * num_points
                buf[r, start : start + num_points] = root[i]
                placed[r] += 1
            # 5) One batch‐FPS to merge per‐root
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged * hasher.root_prim_scales.unsqueeze(1)

    return result


def _triangulate_faces(prim) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def prim_to_trimesh(prim, relative_to_world=False) -> trimesh.Trimesh:
    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # shape (4,4)
        mesh_tm.apply_transform(tf)

    return mesh_tm


def fps(points: torch.Tensor, n_samples: int, memory_threashold=2 * 1024**3) -> torch.Tensor:  # 2 GiB
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float("inf"), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def prim_to_warp_mesh(prim, device, relative_to_world=False) -> wp.Mesh:
    if prim.GetTypeName() == "Mesh":
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # (4,4)
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh


@wp.kernel
def get_signed_distance(
    queries: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * n_points, 3]
    mesh_handles: wp.array(dtype=wp.uint64),  # [n_obstacles * E_bad * max_prims]
    prim_counts: wp.array(dtype=wp.int32),  # [n_obstacles * E_bad]
    coll_rel_pos: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * max_prims, 3]
    coll_rel_quat: wp.array(dtype=wp.quat),  # [n_obstacles * E_bad * max_prims, 4]
    coll_rel_scale: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * max_prims, 3]
    max_dist: float,
    check_dist: bool,
    num_envs: int,
    num_points: int,
    max_prims: int,
    signs: wp.array(dtype=float),  # [E_bad * n_points]
):
    tid = wp.tid()
    per_obstacle_stride = num_envs * num_points
    obstacle_idx = tid // per_obstacle_stride
    rem = tid - obstacle_idx * per_obstacle_stride
    env_id = rem // num_points  # this env_id is index of arange(0, len(env_id)), its sequence, not selective indexing
    q = queries[tid]
    # accumulator for the lowest‐sign (start large)
    best_signed_dist = max_dist
    obstacle_env_base = obstacle_idx * num_envs * max_prims + env_id * max_prims
    prim_id = obstacle_idx * num_envs + env_id

    for p in range(prim_counts[prim_id]):
        index = obstacle_env_base + p
        mid = mesh_handles[index]
        if mid != 0:
            q1 = q - coll_rel_pos[index]
            q2 = wp.quat_rotate_inv(coll_rel_quat[index], q1)
            crs = coll_rel_scale[index]
            q3 = wp.vec3(q2.x / crs.x, q2.y / crs.y, q2.z / crs.z)
            mp = wp.mesh_query_point(mid, q3, max_dist)
            if mp.result:
                if check_dist:
                    closest = wp.mesh_eval_position(mid, mp.face, mp.u, mp.v)
                    local_dist = q3 - closest
                    unscaled_local_dist = wp.vec3(local_dist.x * crs.x, local_dist.y * crs.y, local_dist.z * crs.z)
                    delta_root = wp.quat_rotate(coll_rel_quat[index], unscaled_local_dist)
                    dist = wp.length(delta_root)
                    signed_dist = dist * mp.sign
                else:
                    signed_dist = mp.sign
                if signed_dist < best_signed_dist:
                    best_signed_dist = signed_dist
    signs[tid] = best_signed_dist


@contextmanager
def temporary_seed(seed: int, restore_numpy: bool = True, restore_python: bool = True):
    # snapshot states
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state() if restore_numpy else None
    py_state = random.getstate() if restore_python else None

    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            torch_utils.set_seed(seed)
        yield
    finally:
        # restore everything
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if np_state is not None:
            np.random.set_state(np_state)
        if py_state is not None:
            random.setstate(py_state)


def read_metadata_from_usd_directory(usd_path: str) -> dict:
    """Read metadata from metadata.yaml in the same directory as the USD file."""
    # Get the directory containing the USD file
    usd_dir = os.path.dirname(usd_path)

    # Look for metadata.yaml in the same directory
    metadata_path = os.path.join(usd_dir, "metadata.yaml")
    rank = int(os.getenv("RANK", "0"))
    download_dir = os.path.join(tempfile.gettempdir(), f"rank_{rank}")
    with open(retrieve_file_path(metadata_path, download_dir=download_dir)) as f:
        metadata_file = yaml.safe_load(f)

    return metadata_file


def compute_assembly_hash(*usd_paths: str) -> str:
    """Compute a hash for an assembly based on the USD file paths.

    Args:
        *usd_paths: Variable number of USD file paths

    Returns:
        A hash string that uniquely identifies the combination of objects
    """
    # Extract path suffixes and sort to ensure consistent hash regardless of input order
    sorted_paths = sorted(urlparse(path).path for path in usd_paths)
    combined = "|".join(sorted_paths)

    full_hash = hashlib.md5(combined.encode()).hexdigest()
    return full_hash


def is_s3_url(url: str) -> bool:
    """Check if a URL/URI is an S3 URL or URI.
    
    Supports both:
    - S3 URLs: https://bucket.s3.region.amazonaws.com/path/to/file
    - S3 URIs: s3://bucket-name/path/to/file
    
    Args:
        url: The URL/URI to check
        
    Returns:
        True if the URL/URI is an S3 URL or URI, False otherwise
    """
    return url.startswith("s3://") or (url.startswith("https://") and ".s3." in url)


def download_from_s3(s3_path: str, download_dir: str, force_download: bool = True) -> str:
    """Download a file from S3 using boto3 with authentication.
    
    This function supports both S3 URLs and S3 URIs:
    - S3 URLs: https://bucket.s3.region.amazonaws.com/path/to/file.pt
    - S3 URIs: s3://bucket-name/path/to/file.pt
    
    This function supports:
    - SageMaker IAM roles (automatic credential detection)
    - Local SSO profiles (via AWS_PROFILE environment variable or default profile)
    - Standard AWS credentials (via ~/.aws/credentials or environment variables)
    
    Args:
        s3_path: The S3 URL or URI
        download_dir: Directory where the file should be downloaded
        force_download: Whether to force re-download even if file exists locally
        
    Returns:
        The local path to the downloaded file
        
    Raises:
        ImportError: If boto3 is not installed
        FileNotFoundError: If the file cannot be downloaded or accessed
    """
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required for S3 downloads. Install it with: pip install boto3"
        )
    
    # Handle S3 URI format (s3://bucket/key)
    if s3_path.startswith("s3://"):
        # Parse S3 URI: s3://bucket-name/path/to/file or s3://bucket-name/file
        uri_path = s3_path[5:]  # Remove "s3://" prefix
        if not uri_path:
            raise ValueError(f"Invalid S3 URI (empty path): {s3_path}")
        
        parts = uri_path.split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
        elif len(parts) == 1:
            # Only bucket name provided, which is invalid (need at least a key)
            raise ValueError(f"Invalid S3 URI (no key specified): {s3_path}")
        else:
            raise ValueError(f"Unable to parse S3 URI: {s3_path}")
        region = None  # Region will be auto-detected by boto3
    else:
        # Parse the S3 URL to extract bucket and key
        parsed = urlparse(s3_path)
        # Handle both virtual-hosted and path-style URLs
        # e.g., https://bucket.s3.region.amazonaws.com/key or https://s3.region.amazonaws.com/bucket/key
        hostname = parsed.netloc
        
        if ".s3." in hostname:
            # Virtual-hosted style: bucket.s3.region.amazonaws.com
            bucket = hostname.split(".s3.")[0]
            key = parsed.path.lstrip("/")
        else:
            # Path-style: s3.region.amazonaws.com/bucket/key
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) == 2:
                bucket, key = path_parts
            else:
                raise ValueError(f"Unable to parse S3 URL: {s3_path}")
        
        # Extract region from hostname if possible
        region = None
        if ".s3." in hostname:
            # Try to extract region from hostname like bucket.s3.region.amazonaws.com
            parts = hostname.split(".s3.")
            if len(parts) > 1:
                region_part = parts[1].split(".")[0]
                # Check if it's a valid region format (e.g., us-west-2)
                if "-" in region_part:
                    region = region_part
    
    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    
    # Determine local file path
    if s3_path.startswith("s3://"):
        file_name = os.path.basename(key) if key else "file"
    else:
        parsed = urlparse(s3_path)
        file_name = os.path.basename(key) if key else os.path.basename(parsed.path)
    local_file_path = os.path.join(download_dir, file_name)
    
    # Check if file already exists
    if os.path.exists(local_file_path) and not force_download:
        return os.path.abspath(local_file_path)
    
    # Initialize boto3 S3 client
    # boto3 will automatically use credentials from:
    # 1. IAM role (in SageMaker/EC2)
    # 2. AWS_PROFILE environment variable
    # 3. Default profile
    # 4. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # 5. ~/.aws/credentials file
    try:
        if region:
            s3_client = boto3.client("s3", region_name=region)
        else:
            # Try to get region from environment or use default
            s3_client = boto3.client("s3")
    except NoCredentialsError:
        raise FileNotFoundError(
            f"No AWS credentials found. Please configure credentials using one of:\n"
            f"  - IAM role (for SageMaker/EC2)\n"
            f"  - AWS_PROFILE environment variable\n"
            f"  - AWS credentials file (~/.aws/credentials)\n"
            f"  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"
        )
    
    # Download the file
    try:
        logger.info(f"Downloading {s3_path} to {local_file_path}")
        s3_client.download_file(bucket, key, local_file_path)
        logger.info(f"Successfully downloaded {s3_path} to {local_file_path}")
        return os.path.abspath(local_file_path)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"S3 object not found: {s3_path}")
        elif error_code == "403":
            raise FileNotFoundError(
                f"Access denied to S3 object: {s3_path}. "
                f"Please check your AWS credentials and permissions."
            )
        else:
            raise FileNotFoundError(f"Failed to download from S3: {s3_path}. Error: {e}")
    except Exception as e:
        raise FileNotFoundError(f"Unexpected error downloading from S3: {s3_path}. Error: {e}")


def retrieve_file_path_with_s3_support(
    path: str, download_dir: str | None = None, force_download: bool = True
) -> str:
    """Retrieve file path with support for S3 URLs and URIs using authenticated downloads.
    
    This is a wrapper around retrieve_file_path that adds S3 support. If the path
    is an S3 URL or URI, it will use boto3 to download with authentication. Otherwise,
    it falls back to the standard retrieve_file_path function.
    
    Supports:
    - S3 URLs: https://bucket.s3.region.amazonaws.com/path/to/file
    - S3 URIs: s3://bucket-name/path/to/file
    
    Args:
        path: The path to the file (local, Nucleus, S3 URL, or S3 URI)
        download_dir: Directory where files should be downloaded
        force_download: Whether to force re-download
        
    Returns:
        The local path to the file
    """
    # Check if it's an S3 URL or URI
    if is_s3_url(path):
        if download_dir is None:
            download_dir = tempfile.gettempdir()
        return download_from_s3(path, download_dir, force_download)
    else:
        # Use the standard retrieve_file_path for local/Nucleus files
        return retrieve_file_path(path, download_dir, force_download)
