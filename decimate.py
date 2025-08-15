import sys
import os
import scene.gaussian_model as GS
import torch
import typing
from tqdm import tqdm

def decimate(radius, gaussian_model : GS.GaussianModel):
    print("Starting Downscaling with radius " + str(radius))
    xyz = gaussian_model.get_xyz
    scaling = gaussian_model.get_scaling
    rotation = gaussian_model._rotation
    features_dc = gaussian_model._features_dc
    features_rest = gaussian_model._features_rest
    opacity = gaussian_model._opacity

    N = xyz.shape[0]
    dist_matrix = torch.cdist(xyz, xyz)  # [N, N]
    visited = torch.zeros(N, dtype=torch.bool, device=xyz.device)

    new_xyz, new_scaling, new_rotation = [], [], []
    new_features_dc, new_features_rest, new_opacity, new_tmp_radii = [], [], [], []

    for i in range(N):
        if visited[i]:
            continue

        cluster_mask = (dist_matrix[i] <= radius) & (~visited)
        visited[cluster_mask] = True

        cluster_xyz = xyz[cluster_mask]
        cluster_scaling = scaling[cluster_mask]
        cluster_rotation = rotation[cluster_mask]
        cluster_features_dc = features_dc[cluster_mask]
        cluster_features_rest = features_rest[cluster_mask]
        cluster_opacity = opacity[cluster_mask]

        # Merge: position = mean
        merged_xyz = cluster_xyz.mean(dim=0)

        # Scale = cover cluster extent
        extent = (cluster_xyz - merged_xyz).norm(dim=1).max()
        merged_scale = cluster_scaling.max(dim=0).values + extent

        # Rotation/features/opacity = simple average
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
    new_tmp_radii = 0

    # Stack and replace
    gaussian_model._xyz = torch.stack(new_xyz)
    gaussian_model._scaling = gaussian_model.scaling_inverse_activation(torch.stack(new_scaling))
    gaussian_model._rotation = torch.stack(new_rotation)
    gaussian_model._features_dc = torch.cat(new_features_dc, dim=0)
    gaussian_model._features_rest = torch.cat(new_features_rest, dim=0)
    gaussian_model._opacity = torch.cat(new_opacity, dim=0)

if __name__ == "__main__":

    url_to_model = sys.argv[1]
    downscale_radius = sys.argv[2]
    save_path = sys.argv[3]

    new_gaussian_model = GS.GaussianModel(3)
    new_gaussian_model.load_ply(url_to_model)
    print("Start Downscale...")
    decimate(float(downscale_radius), new_gaussian_model)

    print("Saving...")
    new_gaussian_model.save_ply(save_path)