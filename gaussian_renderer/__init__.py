#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer # CUDA version
from .cpu_rasterizer import CPUGaussian, rasterize_gaussians_cpu # CPU version
from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh # Not directly used by CPU rasterizer for SH evaluation
import numpy as np

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene using CPU rasterizer.
    
    Background tensor (bg_color) must be on CPU for CPU rasterizer!
    """

    bg_color_cpu = bg_color.cpu()

    xyz_np = pc.get_xyz.cpu().numpy()
    scales_np = pc.get_scaling.cpu().numpy()
    rotations_np = pc.get_rotation.cpu().numpy() 
    opacities_np = pc.get_opacity.cpu().numpy()
    shs_np = pc.get_features.cpu().numpy()
    # Get object IDs from the GaussianModel
    # pc.get_objects typically returns a tensor of shape (N, 1) with dtype int32
    object_ids_tensor = pc.get_objects
    if object_ids_tensor is None:
        # If get_objects is not implemented or returns None, create dummy IDs (e.g., all zeros)
        # This might happen if the model wasn't trained with object IDs or if it's a base 3DGS model.
        # For gaussian-grouping, this should be available.
        print("Warning: pc.get_objects returned None. Using dummy object IDs (all 0).")
        object_ids_np = np.zeros(xyz_np.shape[0], dtype=np.int32)
    else:
        object_ids_np = object_ids_tensor.cpu().numpy().flatten() # Flatten from (N,1) to (N,)
    
    num_gaussians = xyz_np.shape[0]
    cpu_gaussians = []
    for i in range(num_gaussians):
        cpu_gaussians.append(CPUGaussian(
            pos=xyz_np[i],
            scale=scales_np[i],
            rot_quat=rotations_np[i], 
            opacity=opacities_np[i],
            sh=shs_np[i].reshape(-1), 
            object_id=object_ids_np[i], # Pass the object ID
            scale_modifier=scaling_modifier
        ))

    raster_results = rasterize_gaussians_cpu(cpu_gaussians, viewpoint_camera, bg_color_cpu, pipe_args=None)
    
    rendered_image_cpu = raster_results["render"] 
    rendered_objects_cpu = raster_results["render_object"] # This is the object_id_map

    # For compatibility with the original return structure, provide placeholders if needed.
    screenspace_points_cpu = torch.zeros_like(pc.get_xyz.cpu(), dtype=pc.get_xyz.dtype, requires_grad=False, device="cpu")
    num_points = pc.get_xyz.shape[0]
    radii_cpu = torch.ones(num_points, device="cpu") 
    visibility_filter_cpu = radii_cpu > 0 

    # Ensure returned tensors are on the same device as input if required by calling code.
    # However, for a CPU pipeline, keeping them on CPU is usually fine.
    # Let's assume the caller can handle CPU tensors or move them as needed.
    return {
        "render": rendered_image_cpu, # Already a CPU tensor
        "viewspace_points": screenspace_points_cpu,
        "visibility_filter": visibility_filter_cpu,
        "radii": radii_cpu,
        "render_object": rendered_objects_cpu # CPU tensor (segmentation map)
    }


