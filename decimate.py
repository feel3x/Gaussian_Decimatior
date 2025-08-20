
# Gaussian Splat Decimation Tool
# Author: Felix Hirt
# License: MIT License (see LICENSE file for details)

# Note:
# This file contains original code by Felix Hirt, licensed under MIT.
# Portions of this project may incorporate code from:
# https://github.com/graphdeco-inria/gaussian-splatting
# All rights for that software are held by Inria and the Max Planck Institute for Informatics (MPII),
# and its use is subject to the original licensing terms.


import numpy as np
import argparse
import helpers.PointModel as GS
import torch
from tqdm import tqdm
import time
from plyfile import PlyData, PlyElement
    
import torch_scatter


def decimate(base_radius, gaussian_model, density_aware=False):
    xyz = gaussian_model._xyz
    scaling = gaussian_model.get_scaling
    rotation = gaussian_model.get_rotation  # assumed quaternion-like
    features_dc = gaussian_model._features_dc
    features_rest = gaussian_model._features_rest
    opacity = gaussian_model.get_opacity

    #Density-aware radius assignment
    if density_aware:
        print("Computing density-aware radii...")
        radii = compute_density_aware_radius_fast(xyz, base_radius)
    else:
        radii = torch.full((xyz.shape[0],), base_radius, device=xyz.device)

    #luster assignment
    voxel_idx = torch.floor(xyz / radii.unsqueeze(-1)).long()
    voxel_keys = (voxel_idx * torch.tensor(
        [73856093, 19349663, 83492791], device=xyz.device)).sum(1)
    unique_keys, inverse, counts = torch.unique(voxel_keys, return_inverse=True, return_counts=True)

    print("Merging {} splats into {} clusters...".format(len(xyz), len(unique_keys)))

    #center
    mean_xyz = torch_scatter.scatter_mean(xyz, inverse, dim=0)

    #Scaling Fix ---
    dists = torch.norm(xyz - mean_xyz[inverse], dim=1)
    max_dists, _ = torch_scatter.scatter_max(dists, inverse, dim=0)

    diffs = xyz - mean_xyz[inverse]
    outer = diffs.unsqueeze(-1) * diffs.unsqueeze(-2)  # (N,3,3)

    cov_sum = torch_scatter.scatter_sum(outer, inverse, dim=0)  # (C,3,3)
    counts = torch_scatter.scatter_sum(
        torch.ones(xyz.size(0), device=xyz.device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1,3,3),
        inverse, dim=0
    )
    cov = cov_sum / counts.clamp_min(1.0)

    #sanitize
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = cov + torch.eye(3, device=cov.device).unsqueeze(0) * 1e-8
    #double Sanitize covariance
    mask_invalid = torch.isnan(cov).any(dim=(1,2)) | torch.isinf(cov).any(dim=(1,2))
    if mask_invalid.any():
        cov[mask_invalid] = torch.eye(3, device=cov.device).unsqueeze(0)

    eigvals_list, eigvecs_list = [], []
    chunk = 20000  # tune for your GPU
    for i in range(0, cov.size(0), chunk):
        cov_chunk = cov[i:i+chunk]
        # sanitize inside chunk
        cov_chunk = torch.nan_to_num(cov_chunk, nan=0.0, posinf=1e6, neginf=-1e6)
        cov_chunk += torch.eye(3, device=cov.device).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(cov_chunk)
        eigvals_list.append(ev); eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, 0)
    eigvecs = torch.cat(eigvecs_list, 0)
    scale_from_cov = torch.sqrt(torch.clamp(eigvals, min=1e-6))

    #combine conservative estimates
    new_scaling = torch.maximum(torch_scatter.scatter_mean(scaling, inverse, dim=0),
                                scale_from_cov.max(dim=1).values.unsqueeze(1))
    new_scaling = torch.maximum(new_scaling, max_dists.unsqueeze(1) * 0.5)  # inflate a bit

    #normalize input quaternions
    rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)
    mean_rot = quaternion_mean_markley(rotation, inverse, len(unique_keys))


    #Features weighted by opacity
    w = opacity.unsqueeze(-1)
    mean_features_dc = torch_scatter.scatter_sum(features_dc * w, inverse, dim=0) / \
                       torch_scatter.scatter_sum(w, inverse, dim=0)
    mean_features_rest = torch_scatter.scatter_sum(features_rest * w, inverse, dim=0) / \
                         torch_scatter.scatter_sum(w, inverse, dim=0)

    #Opacity fusion 
    one_minus_alpha = 1 - opacity
    prod = torch_scatter.scatter_mul(one_minus_alpha, inverse, dim=0)
    new_opacity = 1 - prod

    #Update Gaussian model
    print("Updating model...")
    gaussian_model._xyz = mean_xyz
    gaussian_model._scaling = gaussian_model.scaling_inverse_activation(new_scaling)
    gaussian_model._rotation = mean_rot
    gaussian_model._features_dc = mean_features_dc
    gaussian_model._features_rest = mean_features_rest
    gaussian_model._opacity = gaussian_model.inverse_opacity_activation(new_opacity)

    return gaussian_model

def quaternion_mean_markley(quats, cluster_ids, num_clusters, chunk: int = 20000):
    """
    Vectorized Markley quaternion averaging.
    """
    N = quats.size(0)

    #Outer product q q^T -> (N,4,4)
    outer = quats.unsqueeze(2) * quats.unsqueeze(1)  # (N,4,4)

    #Sum per cluster
    M_sum = torch_scatter.scatter_sum(outer, cluster_ids, dim=0, dim_size=num_clusters)  # (C,4,4)

    #Count per cluster
    ones = torch.ones((N,1,1), device=quats.device, dtype=quats.dtype).expand(-1,4,4)
    counts = torch_scatter.scatter_sum(ones, cluster_ids, dim=0, dim_size=num_clusters)  # (C,4,4)

    #Normalize
    M = M_sum / counts.clamp_min(1.0)

    #Sanitize
    M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    M = M + torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8
    mask_invalid = torch.isnan(M).any(dim=(1,2)) | torch.isinf(M).any(dim=(1,2))
    if mask_invalid.any():
        M[mask_invalid] = torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0)

    #Chunked eigendecomposition
    eigvals_list, eigvecs_list = [], []
    for i in range(0, M.size(0), chunk):
        M_chunk = M[i:i+chunk]
        M_chunk = torch.nan_to_num(M_chunk, nan=0.0, posinf=1e6, neginf=-1e6)
        M_chunk += torch.eye(4, device=M.device, dtype=M.dtype).unsqueeze(0) * 1e-8
        ev, V = torch.linalg.eigh(M_chunk)  # safe in chunk
        eigvals_list.append(ev); eigvecs_list.append(V)
    eigvals = torch.cat(eigvals_list, 0)
    eigvecs = torch.cat(eigvecs_list, 0)                 # (C,4), (C,4,4)

    max_ids = eigvals.argmax(dim=1)                            # (C,)
    mean_quats = eigvecs[torch.arange(num_clusters, device=quats.device), :, max_ids]  # (C,4)

    #Normalize outputs
    mean_quats = mean_quats / (mean_quats.norm(dim=1, keepdim=True) + 1e-12)
    return mean_quats

def progressive_decimate(gaussian_model, target_count, base_radius=0.01, growth=1.2, max_iter=20):
    """
    Progressive decimation of a Gaussian Splatting model until target count is reached.
    """
    radius = base_radius
    xyz = gaussian_model._xyz

    for it in range(max_iter):
        print(f"\n[Iteration {it+1}] Radius = {radius:.5f}")

        # --- Run smart decimation step ---
        decimate(radius, gaussian_model)

        new_count = gaussian_model._xyz.shape[0]
        print(f" → Reduced to {new_count} splats")

        if new_count <= target_count:
            print("✅ Target reached")
            break

        # Increase radius for next round
        radius *= growth

    return gaussian_model

def compute_density_aware_radius_fast(xyz, base_radius, voxel_size=None, show_progress=False):
    """Approximate density"""
    if voxel_size is None:
        voxel_size = base_radius  #fine grid for density estimation
    
    #Assign to fine voxels
    voxel_idx = torch.floor(xyz / voxel_size).long()
    
    #Create unique voxel keys (hash function for uniqueness)
    voxel_keys = voxel_idx[:, 0] + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2] * 1000000
    
    #Count points per voxel
    unique_keys, counts = torch.unique(voxel_keys, return_counts=True)
    
    #Map back counts to original points
    key_to_count = dict(zip(unique_keys.tolist(), counts.tolist()))
    
    if show_progress:
        density_map = torch.zeros_like(voxel_keys, dtype=torch.float32)
        for i in tqdm(range(len(voxel_keys)), desc="Mapping densities"):
            density_map[i] = key_to_count[voxel_keys[i].item()]
    else:
        #Faster vectorized lookup
        inv_idx = torch.bucketize(voxel_keys, unique_keys)
        density_map = counts[inv_idx - 1].float()
    
    #Normalize by median
    median_density = torch.median(counts.float())
    density_scale = (median_density / density_map).clamp(0.3, 3.0)
    return base_radius * density_scale

def load_model(path):
    model = GS.PointModel()
    model.load_ply(path)
    return model

def save_model(path, model):
    model.save_ply(path)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Decimate a Gaussian PLY Model with a specific radius in which points are merged."
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        help="Path pointing to the Gaussian model.",
        required=True
    )
    parser.add_argument(
        "--decimate_radius",
        type=float,
        default=0.01,
        help="Decimate radius factor. Radius that should only contain Gaussian."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the processed model. (Full path with .ply extension)",
        required=True
    )
    # parser.add_argument(
    #     "--dense_aware",
    #     help="Sparse regions are decimated with a higher radius. (SLOWER! Especially for Large Scenes)",
    #     default=False, 
    #     action='store_true'
    # )

    args = parser.parse_args()

    print("Loading Model...")
    
    #load model
    new_gaussian_model = load_model(args.path_to_model)
    
    start_time = time.time()
    #start process
    print("Start Decimation...")
    decimate(float(args.decimate_radius), new_gaussian_model, False)
    #progressive_decimate(new_gaussian_model, 1000)
    print("--- %s seconds ---" % (time.time() - start_time))

    #save model
    print("Saving...")
    save_model(args.save_path, new_gaussian_model)
    
