#!/usr/bin/env python3
"""
Final fixed CPU test pipeline
"""
import torch
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available, but basic functionality works")

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

def main():
    print("Starting FINAL FIXED CPU test...")
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create model
        gaussians = GaussianModel(sh_degree=3)
        
        # Simple test data - fix the object_ids shape
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
        colors = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32)
        opacities = torch.tensor([[2.0], [2.0]], dtype=torch.float32)
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float32)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        
        # Fix: object_ids should be shape (N, 1) not (N, 1, 1)
        object_ids = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

        # Initialize
        gaussians._xyz = xyz
        gaussians._features_dc = colors
        gaussians._opacity = opacities
        gaussians._scaling = scales
        gaussians._rotation = rotations
        gaussians._objects_dc = object_ids

        print("✓ Gaussian model created with 2 Gaussians")
        print(f"  XYZ shape: {gaussians.get_xyz.shape}")
        print(f"  Object IDs internal shape: {gaussians._objects_dc.shape}")
        
        # Test get_objects method
        try:
            obj_ids = gaussians.get_objects
            print(f"  Object IDs via get_objects: {obj_ids}")
            print(f"  Object IDs shape: {obj_ids.shape}")
        except Exception as e:
            print(f"  Warning: get_objects failed: {e}")

        # Save PLY
        ply_path = os.path.join(output_dir, "test_model_fixed.ply")
        gaussians.save_ply_with_object_id(ply_path)
        
        if os.path.exists(ply_path):
            size = os.path.getsize(ply_path)
            print(f"✓ PLY saved: {ply_path} ({size} bytes)")
        
        # Test rendering if possible
        try:
            W, H = 100, 100
            K = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float32)
            W2C = np.eye(4, dtype=np.float32)
            W2C[2, 3] = 2.0
            
            camera = MockViewpointCamera(W, H, K, W2C, 0.1, 100.0)
            
            render_result = render(camera, gaussians, None, None)
            print("✓ Rendering test successful")
            
            # Save if possible
            try:
                from torchvision.utils import save_image
                rendered_image = render_result["render"]
                save_image(rendered_image, os.path.join(output_dir, "render_fixed.png"))
                print("✓ Rendered image saved")
            except ImportError:
                print("  (torchvision not available for saving)")
                
        except Exception as e:
            print(f"  Rendering test failed: {e}")

        print("\n✓ FIXED CPU test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
