# Test script for CPU-based 3DGS with instance segmentation and PLY export
import torch
import numpy as np
import sys
import os

# Add project root to sys.path to allow imports from scene and gaussian_renderer
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render # This should be our modified render function

class MockViewpointCamera:
    def __init__(self, W, H, K, W2C, Znear, Zfar):
        self.image_width = W
        self.image_height = H
        self.znear = Znear
        self.zfar = Zfar

        # K is [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
        # FoVx = 2 * atan(W / (2 * fx))
        # FoVy = 2 * atan(H / (2 * fy))
        fx = K[0,0]
        fy = K[1,1]
        self.FoVx = 2 * np.arctan(W / (2 * fx))
        self.FoVy = 2 * np.arctan(H / (2 * fy))
        self.tanfovx = np.tan(self.FoVx * 0.5)
        self.tanfovy = np.tan(self.FoVy * 0.5)

        self.world_view_transform = torch.tensor(W2C, dtype=torch.float32) # World to View
        self.camera_center = torch.inverse(self.world_view_transform)[:3, 3] # Cam position in world

        # Projection matrix (OpenGL style)
        # Simplified perspective projection matrix
        # full_proj_transform is view_to_clip * world_to_view
        # Let's construct a simple perspective projection matrix (view to clip)
        proj = torch.zeros(4, 4, dtype=torch.float32)
        proj[0, 0] = 1.0 / self.tanfovx
        proj[1, 1] = 1.0 / self.tanfovy
        proj[2, 2] = (Zfar + Znear) / (Znear - Zfar)
        proj[2, 3] = (2 * Zfar * Znear) / (Znear - Zfar)
        proj[3, 2] = -1.0
        self.projection_matrix = proj # View to Clip
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform # World to Clip

    # Add other methods if the renderer or model expects them, e.g., for device
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

def main():
    print("Starting test for CPU 3DGS with instance segmentation...")
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create and populate GaussianModel
    sh_degree = 0 # Simplest SH degree (DC only)
    model = GaussianModel(sh_degree)

    # Define 2 sample Gaussians
    # Gaussian 1: red, object_id 1
    # Gaussian 2: blue, object_id 2
    num_gaussians = 2
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ], dtype=torch.float32)

    # SH features (DC only for sh_degree 0)
    # RGB2SH for red (1,0,0) and blue (0,0,1)
    # RGB2SH(torch.tensor([1.0,0.0,0.0])) approx [0.282, 0, 0] * some_factor if normalized
    # For simplicity, let's use values that would result in visible red/blue after SH to RGB conversion (+0.5, clip)
    # If color = SH_C0 * c0 + 0.5, then c0 = (target_color - 0.5) / SH_C0
    # Target red (1,0,0) -> c0_r = (1-0.5)/0.282 = 1.77, c0_g = (0-0.5)/0.282 = -1.77, c0_b = (0-0.5)/0.282 = -1.77
    sh_c0 = 0.28209479177387814
    features_dc_g1 = torch.tensor([[[ (1.0-0.5)/sh_c0, (0.0-0.5)/sh_c0, (0.0-0.5)/sh_c0 ]]], dtype=torch.float32)
    features_dc_g2 = torch.tensor([[[ (0.0-0.5)/sh_c0, (0.0-0.5)/sh_c0, (1.0-0.5)/sh_c0 ]]], dtype=torch.float32)
    features_dc = torch.cat((features_dc_g1, features_dc_g2), dim=0) # (2, 1, 3)

    features_rest = torch.empty((num_gaussians, 0, 3), dtype=torch.float32) # Empty for sh_degree 0

    scales = torch.tensor([
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1]
    ], dtype=torch.float32)
    # Log scales for _scaling
    log_scales = torch.log(scales)

    rotations = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], # Quaternion (w, x, y, z)
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    opacities = torch.tensor([
        [0.9],
        [0.9]
    ], dtype=torch.float32)
    # Inverse sigmoid for _opacity
    logit_opacities = torch.log(opacities / (1 - opacities))

    object_ids = torch.tensor([
        [[1]], # Object ID 1
        [[2]]  # Object ID 2
    ], dtype=torch.int32).float() # GaussianModel stores as float then converts with .int()

    # Manually set the private attributes of the model (ensure they are Parameters if model expects)
    # The model's create_from_pcd initializes these as nn.Parameter.
    # For direct setting, we might need to wrap them or ensure the model can handle raw tensors for export/render.
    # Let's assume direct setting works for CPU path for now, or adapt if errors occur.
    model._xyz = xyz
    model._features_dc = features_dc
    model._features_rest = features_rest
    model._scaling = log_scales
    model._rotation = rotations
    model._opacity = logit_opacities
    model._objects_dc = object_ids # (N, 1, 1) as float
    model.active_sh_degree = sh_degree

    print(f"Gaussian model created with {num_gaussians} Gaussians.")
    print(f"  XYZ:\n{model.get_xyz.cpu().numpy()}")
    print(f"  Object IDs (internal _objects_dc):\n{model._objects_dc.cpu().numpy()}")
    print(f"  Object IDs (get_objects):\n{model.get_objects.cpu().numpy()}")

    # 2. Save PLY with object ID
    ply_output_path = os.path.join(output_dir, "test_model_with_ids.ply")
    try:
        model.save_ply_with_object_id(ply_output_path)
        print(f"Saved PLY file with object IDs to: {ply_output_path}")
        # Verify PLY content (manual check or basic parsing if needed later)
        with open(ply_output_path, 'r') as f:
            ply_content_sample = f.read(500)
        print(f"  PLY file sample:\n{ply_content_sample}...")
    except Exception as e:
        print(f"Error saving PLY file: {e}")
        import traceback
        traceback.print_exc()

    # 3. Create Mock Camera
    W, H = 100, 100 # Small image for testing
    fx, fy = 50.0, 50.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # Simple World to Camera: look at origin from z=1, y-up
    # Camera at (0,0,1), looking towards (0,0,0), up (0,1,0)
    # View matrix W2C = LookAt(eye, center, up)
    eye = np.array([0.0, 0.0, 1.0])
    center = np.array([0.0, 0.0, 0.0])
    up_dir = np.array([0.0, 1.0, 0.0])
    
    F = center - eye
    F = F / np.linalg.norm(F)
    R = np.cross(F, up_dir)
    R = R / np.linalg.norm(R)
    U = np.cross(R, F) # F is already normalized
    
    w2c_rot = np.eye(4)
    w2c_rot[0, :3] = R
    w2c_rot[1, :3] = U # OpenGL convention: Y is up
    w2c_rot[2, :3] = -F # OpenGL convention: Z is backward from camera
    
    w2c_trans = np.eye(4)
    w2c_trans[:3, 3] = -eye
    
    W2C_np = w2c_rot @ w2c_trans # This is R then T. View = T_view * R_view
                                # Or more directly: first translate world so camera is at origin, then rotate.
                                # Standard LookAt: R = [r_x, r_y, r_z]^T, t = -R @ eye
                                # R_view = [[R_x, U_x, -F_x], [R_y, U_y, -F_y], [R_z, U_z, -F_z]]
                                # T_view = [-dot(R,eye), -dot(U,eye), -dot(-F,eye)]
    # Let's use a simpler view matrix construction for testing
    # Camera at (0, 0, 1), looking at origin, Y up.
    # View matrix should transform world point (0,0,0) to (0,0,-1) in camera space (if Z is depth axis)
    # And (0.5,0,0) to (0.5,0,-1)
    # And (0,0,0.5) to (0,0,-0.5)
    # Standard lookAt matrix:
    z_axis = (eye - center) / np.linalg.norm(eye - center) # Forward vector of camera (points opposite to viewing direction)
    x_axis = np.cross(up_dir, z_axis) / np.linalg.norm(np.cross(up_dir, z_axis)) # Right vector
    y_axis = np.cross(z_axis, x_axis) # Up vector
    
    rot_mat = np.eye(4, dtype=np.float32)
    rot_mat[0, :3] = x_axis
    rot_mat[1, :3] = y_axis
    rot_mat[2, :3] = z_axis
    
    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[:3, 3] = -eye
    
    W2C_np = rot_mat @ trans_mat # This is a common view matrix form
    # The CPU rasterizer expects world_view_transform where +Z is into the screen (camera looks along -Z)
    # The above W2C_np makes camera look along its -Z axis. So Z_cam = Z_world - 1 for points on Z axis if eye=(0,0,1)
    # This seems consistent. Point (0,0,0)world -> (0,0,-1)view. Point (0.5,0,0)world -> (0.5,0,-1)view.

    Znear, Zfar = 0.1, 100.0
    viewpoint_camera = MockViewpointCamera(W, H, K, W2C_np, Znear, Zfar)
    viewpoint_camera.to(torch.device("cpu")) # Ensure camera tensors are on CPU

    print("Mock camera created.")

    # 4. Render scene
    bg_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device="cpu") # Dark grey background
    pipe_args = None # Not used by our CPU rasterizer's current version

    try:
        print("Rendering scene with CPU rasterizer...")
        render_output = render(viewpoint_camera, model, pipe_args, bg_color)
        rendered_image = render_output["render"] # (3, H, W)
        object_id_map = render_output["render_object"] # (H, W)

        print(f"Rendering complete. Image shape: {rendered_image.shape}, Object ID map shape: {object_id_map.shape}")

        # Save rendered image (requires a library like Pillow or torchvision)
        try:
            from torchvision.utils import save_image
            rendered_image_path = os.path.join(output_dir, "rendered_image_cpu.png")
            save_image(rendered_image, rendered_image_path)
            print(f"Rendered image saved to: {rendered_image_path}")
        except ImportError:
            print("torchvision not available, cannot save rendered image.")
        except Exception as e_img:
            print(f"Error saving rendered image: {e_img}")

        # Save object_id_map (e.g., as a grayscale image or raw tensor)
        try:
            # Normalize object_id_map for visualization if needed, or save raw
            object_id_map_path_raw = os.path.join(output_dir, "object_id_map_raw.pt")
            torch.save(object_id_map, object_id_map_path_raw)
            print(f"Object ID map (raw tensor) saved to: {object_id_map_path_raw}")
            
            # Visualize as image (requires matplotlib or Pillow)
            if object_id_map.max() > 0:
                object_id_map_vis = object_id_map.float() / object_id_map.max()
                object_id_map_vis_path = os.path.join(output_dir, "object_id_map_vis.png")
                save_image(object_id_map_vis.unsqueeze(0), object_id_map_vis_path) # Add channel dim
                print(f"Object ID map (visualized) saved to: {object_id_map_vis_path}")
            else:
                print("Object ID map is all zeros, skipping visualization save.")

        except ImportError:
            print("torchvision not available, cannot save object_id_map as image.")
        except Exception as e_map:
            print(f"Error saving object_id_map: {e_map}")
            
        # Basic check of object_id_map content
        unique_ids = torch.unique(object_id_map)
        print(f"Unique IDs in object_id_map: {unique_ids.cpu().numpy()}")
        # Expected: [0, 1, 2] if background is 0 and both Gaussians are visible.

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

    print("Test script finished.")

if __name__ == "__main__":
    main()


