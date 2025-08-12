#!/usr/bin/env python3
"""
Fixed CPU test pipeline for 3DGS with instance segmentation
"""
import torch
import numpy as np
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

class MockViewpointCamera:
    def __init__(self, W, H, K, W2C, Znear, Zfar):
        self.image_width = W
        self.image_height = H
        self.znear = Znear
        self.zfar = Zfar

        fx = K[0,0]
        fy = K[1,1]
        self.FoVx = 2 * np.arctan(W / (2 * fx))
        self.FoVy = 2 * np.arctan(H / (2 * fy))
        self.tanfovx = np.tan(self.FoVx * 0.5)
        self.tanfovy = np.tan(self.FoVy * 0.5)

        self.world_view_transform = torch.tensor(W2C, dtype=torch.float32)
        self.camera_center = torch.inverse(self.world_view_transform)[:3, 3]

        proj = torch.zeros(4, 4, dtype=torch.float32)
        proj[0, 0] = 1.0 / self.tanfovx
        proj[1, 1] = 1.0 / self.tanfovy
        proj[2, 2] = (Zfar + Znear) / (Znear - Zfar)
        proj[2, 3] = (2 * Zfar * Znear) / (Znear - Zfar)
        proj[3, 2] = -1.0
        self.projection_matrix = proj
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform

    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

def main():
    print("Starting FIXED test for CPU 3DGS with instance segmentation...")
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create Gaussian model
    gaussians = GaussianModel(sh_degree=3)
    
    # Create simple test data
    num_points = 2
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    colors = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32)
    opacities = torch.tensor([[2.0], [2.0]], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    object_ids = torch.tensor([[1], [2]], dtype=torch.float32)

    # Initialize model
    gaussians._xyz = xyz
    gaussians._features_dc = colors
    gaussians._opacity = opacities
    gaussians._scaling = scales
    gaussians._rotation = rotations
    gaussians._objects_dc = object_ids

    print("Gaussian model created with 2 Gaussians.")
    print(f"  XYZ:\n{gaussians.get_xyz.detach().cpu().numpy()}")
    print(f"  Object IDs (internal _objects_dc):\n{gaussians._objects_dc.detach().cpu().numpy()}")
    print(f"  Object IDs (get_objects):\n{gaussians.get_objects.detach().cpu().numpy()}")

    # Save PLY with object IDs
    ply_path = os.path.join(output_dir, "test_model_with_ids.ply")
    gaussians.save_ply_with_object_id(ply_path)
    print(f"PLY file saved to {ply_path} with object IDs.")

    # Verify PLY file exists (don't try to read as text)
    if os.path.exists(ply_path):
        file_size = os.path.getsize(ply_path)
        print(f"Saved PLY file: {ply_path} (size: {file_size} bytes)")
    else:
        print("Error: PLY file was not created")

    # Create mock camera
    W, H = 100, 100
    K = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float32)
    W2C = np.eye(4, dtype=np.float32)
    W2C[2, 3] = 2.0  # Move camera back
    viewpoint_camera = MockViewpointCamera(W, H, K, W2C, 0.1, 100.0)
    print("Mock camera created.")

    # Render
    try:
        print("Rendering scene with CPU rasterizer...")
        render_result = render(viewpoint_camera, gaussians, None, None)
        
        rendered_image = render_result["render"]
        object_id_map = render_result.get("render_object", torch.zeros(H, W))
        
        print(f"Rendering complete. Image shape: {rendered_image.shape}, Object ID map shape: {object_id_map.shape}")

        # Save rendered image
        try:
            from torchvision.utils import save_image
            rendered_image_path = os.path.join(output_dir, "rendered_image_cpu.png")
            save_image(rendered_image, rendered_image_path)
            print(f"Rendered image saved to: {rendered_image_path}")
        except ImportError:
            print("torchvision not available, cannot save rendered image.")

        # Save object_id_map
        try:
            object_id_map_path_raw = os.path.join(output_dir, "object_id_map_raw.pt")
            torch.save(object_id_map, object_id_map_path_raw)
            print(f"Object ID map (raw tensor) saved to: {object_id_map_path_raw}")
            
            if object_id_map.max() > 0:
                from torchvision.utils import save_image
                object_id_map_vis = object_id_map.float() / object_id_map.max()
                object_id_map_vis_path = os.path.join(output_dir, "object_id_map_vis.png")
                save_image(object_id_map_vis.unsqueeze(0), object_id_map_vis_path)
                print(f"Object ID map (visualized) saved to: {object_id_map_vis_path}")
            else:
                print("Object ID map is all zeros, skipping visualization save.")

        except Exception as e_map:
            print(f"Error saving object_id_map: {e_map}")
            
        # Check object_id_map content
        unique_ids = torch.unique(object_id_map)
        print(f"Unique IDs in object_id_map: {unique_ids.cpu().numpy()}")

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

    print("Fixed test script finished successfully.")

if __name__ == "__main__":
    main()
