import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device='cuda')

    
        # TODO (Q1.4): Sample points from z values
        O = ray_bundle.origins.shape[0]   
        N =  z_vals.shape[0]
        # print("11111 directions shape:", ray_bundle.directions.shape)
        # print("11111 origins shape:", ray_bundle.origins.shape)
        # print("11111 Z_vals shape:", z_vals.shape)
        
        origins= ray_bundle.origins.unsqueeze(1).repeat(1, N, 1)         # [batch_size, 3] ->  [batch_size, n_pts_per_ray, 3]
        directions = ray_bundle.directions.unsqueeze(1).repeat(1, N, 1)  # [batch_size, 3] ->  [batch_size, n_pts_per_ray, 3]
        # print("directions shape:", directions.shape)
        # print("origins shape:", origins.shape)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(O, 1, 1)       # [n_pts_per_ray] ->  [batch_size, n_pts_per_ray, 1]
        # print("Z_vals shape:", z_vals.shape)
        sample_points = z_vals * directions + origins                    # [batch_size, n_pts_per_ray, 3]
        # print("final sample points:", sample_points.shape)
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}