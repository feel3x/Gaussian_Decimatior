import sys
import argparse
import scene.gaussian_model as GS
import torch
import typing
from tqdm import tqdm

#SLOWER decimate even across voxel borders
def decimateWITHBORDERS(merge_radius, model : GS.GaussianModel, voxel_size = 0.01, max_scale = 100000):
    """
    Merge Gaussian points using voxel binning for speed but only merge points
    within a true radius. Clamp final scale to `max_scale`.

    merge_radius: float, max distance to merge points
    max_scale: float, clamp merged scale so Gaussians don't blow up
    """
    # Extract attributes
    xyz = model.get_xyz
    scaling = model.get_scaling
    rotation = model._rotation
    features_dc = model._features_dc
    features_rest = model._features_rest
    opacity = model._opacity

    device = xyz.device

    #decouple eventually
    voxel_size = merge_radius

    # Step 1 — assign points to voxels
    voxel_indices = torch.floor(xyz / voxel_size).to(torch.int64)
    voxel_keys = voxel_indices[:, 0] * 73856093 ^ voxel_indices[:, 1] * 19349663 ^ voxel_indices[:, 2] * 83492791

    # Step 2 — sort points by voxel key
    sorted_keys, sort_idx = torch.sort(voxel_keys)
    xyz = xyz[sort_idx]
    scaling = scaling[sort_idx]
    rotation = rotation[sort_idx]
    features_dc = features_dc[sort_idx]
    features_rest = features_rest[sort_idx]
    opacity = opacity[sort_idx]

    # Step 3 — iterate voxel groups and merge within radius
    new_xyz, new_scaling, new_rotation = [], [], []
    new_features_dc, new_features_rest, new_opacity, new_tmp_radii = [], [], [], []
    unique_keys = torch.unique_consecutive(sorted_keys, return_counts=False)
    progress_bar = tqdm(total=len(unique_keys), desc="Merging Gaussians", unit="voxel")
    start = 0
    N = xyz.shape[0]
    while start < N:
        # Find end of this voxel group
        end = start + 1
        while end < N and sorted_keys[end] == sorted_keys[start]:
            end += 1

        # Extract points in voxel
        voxel_points = xyz[start:end]

        # Optional: Also check neighboring voxels
        # Get neighbor voxel keys (27 including self)
        vi = voxel_indices[sort_idx[start]]
        neighbor_offsets = torch.tensor(
            [[dx, dy, dz] for dx in (-1, 0, 1)
                          for dy in (-1, 0, 1)
                          for dz in (-1, 0, 1)],
            device=device, dtype=torch.int64
        )
        neighbor_keys = (vi + neighbor_offsets)[:, 0] * 73856093 ^ \
                        (vi + neighbor_offsets)[:, 1] * 19349663 ^ \
                        (vi + neighbor_offsets)[:, 2] * 83492791

        # Mask for points in neighboring voxels
        mask_neighbors = torch.isin(sorted_keys, neighbor_keys)
        neighbor_points_idx = torch.where(mask_neighbors)[0]

        # Gather candidate points
        candidates_xyz = xyz[neighbor_points_idx]
        candidates_scaling = scaling[neighbor_points_idx]
        candidates_rotation = rotation[neighbor_points_idx]
        candidates_features_dc = features_dc[neighbor_points_idx]
        candidates_features_rest = features_rest[neighbor_points_idx]
        candidates_opacity = opacity[neighbor_points_idx]

        # Compute distances to the voxel's first point
        dist = torch.norm(candidates_xyz - voxel_points[0], dim=1)
        merge_mask = dist <= merge_radius

        cluster_xyz = candidates_xyz[merge_mask]
        cluster_scaling = candidates_scaling[merge_mask]
        cluster_rotation = candidates_rotation[merge_mask]
        cluster_features_dc = candidates_features_dc[merge_mask]
        cluster_features_rest = candidates_features_rest[merge_mask]
        cluster_opacity = candidates_opacity[merge_mask]

        # Merge
        merged_xyz = cluster_xyz.mean(dim=0)
        extent = (cluster_xyz - merged_xyz).norm(dim=1).max()
        merged_scale = (cluster_scaling.max(dim=0).values + extent).clamp(max=max_scale)
        merged_rotation = cluster_rotation.mean(dim=0)
        merged_features_dc = cluster_features_dc.mean(dim=0, keepdim=True)
        merged_features_rest = cluster_features_rest.mean(dim=0, keepdim=True)
        merged_opacity = cluster_opacity.mean(dim=0, keepdim=True)

        # Append
        new_xyz.append(merged_xyz)
        new_scaling.append(merged_scale)
        new_rotation.append(merged_rotation)
        new_features_dc.append(merged_features_dc)
        new_features_rest.append(merged_features_rest)
        new_opacity.append(merged_opacity)

        start = end
        progress_bar.update(1)

    progress_bar.close()

    # Step 4 — stack and replace
    model._xyz = torch.stack(new_xyz)
    model._scaling = model.scaling_inverse_activation(torch.stack(new_scaling))
    model._rotation = torch.stack(new_rotation)
    model._features_dc = torch.cat(new_features_dc, dim=0)
    model._features_rest = torch.cat(new_features_rest, dim=0)
    model._opacity = torch.cat(new_opacity, dim=0)


def decimate(radius, gaussian_model : GS.GaussianModel):
    xyz = gaussian_model._xyz
    scaling = gaussian_model.get_scaling
    rotation = gaussian_model._rotation
    features_dc = gaussian_model._features_dc
    features_rest = gaussian_model._features_rest
    opacity = gaussian_model._opacity

    #Assign points to voxels
    voxel_size = radius
    voxel_indices = torch.floor(xyz / voxel_size).to(torch.int64)

    #Make voxel keys (hash)
    #primes to reduce collisions
    voxel_keys = voxel_indices[:,0]*73856093 ^ voxel_indices[:,1]*19349663 ^ voxel_indices[:,2]*83492791

    #Sort by voxel key
    sorted_keys, sort_idx = torch.sort(voxel_keys)
    xyz = xyz[sort_idx]
    scaling = scaling[sort_idx]
    rotation = rotation[sort_idx]
    features_dc = features_dc[sort_idx]
    features_rest = features_rest[sort_idx]
    opacity = opacity[sort_idx]

    #Merge within each voxel group
    new_xyz, new_scaling, new_rotation = [], [], []
    new_features_dc, new_features_rest, new_opacity, new_tmp_radii = [], [], [], []
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    progress_bar = tqdm(total=len(unique_keys), desc="Merging Gaussians", unit="voxel")
    start = 0
    while start < len(xyz):
        end = start + 1
        while end < len(xyz) and sorted_keys[end] == sorted_keys[start]:
            end += 1

        #Points in this voxel
        cluster_xyz = xyz[start:end]
        cluster_scaling = scaling[start:end]
        cluster_rotation = rotation[start:end]
        cluster_features_dc = features_dc[start:end]
        cluster_features_rest = features_rest[start:end]
        cluster_opacity = opacity[start:end]

        merged_xyz = cluster_xyz.mean(dim=0)
        #extent = (cluster_xyz - merged_xyz).norm(dim=1).max()
        extent = min((cluster_xyz - merged_xyz).norm(dim=1).max().item(), radius)
        merged_scale = (cluster_scaling.max(dim=0).values + extent).clamp(max=1, min=0.00001)
        merged_rotation = cluster_rotation.mean(dim=0)
        merged_features_dc = cluster_features_dc.mean(dim=0, keepdim=True)
        merged_features_rest = cluster_features_rest.mean(dim=0, keepdim=True)
        merged_opacity = cluster_opacity.mean(dim=0, keepdim=True)

        new_xyz.append(merged_xyz)
        new_scaling.append(merged_scale)
        new_rotation.append(merged_rotation)
        new_features_dc.append(merged_features_dc)
        new_features_rest.append(merged_features_rest)
        new_opacity.append(merged_opacity)

        start = end
        progress_bar.update(1)

    progress_bar.close()

    #Stack
    gaussian_model._xyz = torch.stack(new_xyz)
    gaussian_model._scaling = gaussian_model.scaling_inverse_activation(torch.stack(new_scaling))
    gaussian_model._rotation = torch.stack(new_rotation)
    gaussian_model._features_dc = torch.cat(new_features_dc, dim=0)
    gaussian_model._features_rest = torch.cat(new_features_rest, dim=0)
    gaussian_model._opacity = torch.cat(new_opacity, dim=0)

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

    args = parser.parse_args()

    new_gaussian_model = GS.GaussianModel(3)
    new_gaussian_model.load_ply(args.path_to_model)
    print("Start Downscale...")
    decimate(float(args.decimate_radius), new_gaussian_model)

    print("Saving...")
    new_gaussian_model.save_ply(args.save_path)