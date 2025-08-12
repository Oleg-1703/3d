# CPU-based Gaussian rasterizer
# Adapted from thomasantony/splat (https://github.com/thomasantony/splat)

import numpy as np
import torch
import scipy as sp
from scipy.spatial.transform import Rotation as ScipyRotation

# Constants for Spherical Harmonics (SH)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

class CPUGaussian:
    def __init__(self, pos, scale, rot_quat, opacity, sh, object_id, scale_modifier=1.0):
        self.pos = np.array(pos, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32) * scale_modifier
        self.rot = ScipyRotation.from_quat([rot_quat[1], rot_quat[2], rot_quat[3], rot_quat[0]])
        self.opacity = opacity[0] if isinstance(opacity, (list, np.ndarray)) else opacity
        self.sh = np.array(sh, dtype=np.float32)
        self.object_id = int(object_id) # Store object ID
        self.cov3D = self._compute_cov3d()

    def _compute_cov3d(self):
        S = np.diag(self.scale**2)
        R = self.rot.as_matrix()
        cov3D = R @ S @ R.T
        return cov3D

    def get_cov2d_and_depth(self, viewpoint_camera):
        view_matrix = viewpoint_camera.world_view_transform.cpu().numpy()
        pos_w = np.append(self.pos, 1.0)
        pos_v = view_matrix @ pos_w
        depth = pos_v[2]

        focal_x = viewpoint_camera.image_width / (2 * viewpoint_camera.tanfovx)
        focal_y = viewpoint_camera.image_height / (2 * viewpoint_camera.tanfovy)
        
        tx, ty, tz = pos_v[0], pos_v[1], pos_v[2]
        if tz < 1e-5: tz = 1e-5

        J = np.array([
            [focal_x / tz, 0.0, -(focal_x * tx) / (tz * tz)],
            [0.0, focal_y / tz, -(focal_y * ty) / (tz * tz)],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)

        cov3D_cam = view_matrix[:3, :3] @ self.cov3D @ view_matrix[:3, :3].T
        cov2d_intermediate = J @ cov3D_cam @ J.T
        cov2d = cov2d_intermediate[:2, :2]
        cov2d[0, 0] += 0.3
        cov2d[1, 1] += 0.3
        return cov2d, depth, pos_v[:3]

    def get_color(self, view_dir_w):
        dir_w_norm = view_dir_w / (np.linalg.norm(view_dir_w) + 1e-8)
        x, y, z = dir_w_norm[0], dir_w_norm[1], dir_w_norm[2]

        color = SH_C0 * self.sh[0:3]
        num_sh_coeffs_per_color = self.sh.shape[0] // 3
        sh_reshaped = self.sh.reshape(num_sh_coeffs_per_color, 3)

        if num_sh_coeffs_per_color > 1:
            # Degree 1 SH (indices 1,2,3 in reshaped SH array)
            # Original gaussian_grouping sh order: DC, then 3 per degree.
            # self.sh[0:3] is DC (index 0 of sh_reshaped)
            # self.sh[3:6], self.sh[6:9], self.sh[9:12] are the 3 components of 1st degree SH (indices 1,2,3 of sh_reshaped)
            # The formula uses sh1_r, sh1_g, sh1_b. If sh_reshaped[1] is (r,g,b) for 1st component of 1st degree.
            # The original paper formula is: c_out = c_dc + sum(c_i * Y_i(d))
            # Y_1^-1 = -SH_C1 * y; Y_1^0 = SH_C1 * z; Y_1^1 = -SH_C1 * x
            # So, color += SH_C1 * (-y * sh_reshaped[1] + z * sh_reshaped[2] - x * sh_reshaped[3])
            # This matches the structure if sh_reshaped[1] = c1, sh_reshaped[2]=c2, sh_reshaped[3]=c3 in the original code.
            # Let's assume sh_reshaped[1] = (sh_y_r, sh_y_g, sh_y_b), sh_reshaped[2] = (sh_z_r, sh_z_g, sh_z_b) etc.
            color = color - SH_C1 * y * sh_reshaped[1] + SH_C1 * z * sh_reshaped[2] - SH_C1 * x * sh_reshaped[3]

        if num_sh_coeffs_per_color > 4: # DC (1) + Deg1 (3) = 4 basis functions
            # Degree 2 SH (indices 4 to 8 in reshaped SH array)
            xy, yz, zz = x * y, y * z, z * z
            xx, yy = x * x, y * y
            # sh_reshaped[4] to sh_reshaped[8] are the 5 components of 2nd degree SH
            color = color + \
                SH_C2[0] * xy * sh_reshaped[4] + \
                SH_C2[1] * yz * sh_reshaped[5] + \
                SH_C2[2] * (2.0 * zz - xx - yy) * sh_reshaped[6] + \
                SH_C2[3] * (x * z) * sh_reshaped[7] + \
                SH_C2[4] * (xx - yy) * sh_reshaped[8]

        color += 0.5
        return np.clip(color, 0.0, 1.0)

def rasterize_gaussians_cpu(gaussians, viewpoint_camera, bg_color, pipe_args):
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)
    
    rendered_image = bg_color.view(3, 1, 1).expand(3, image_height, image_width).clone() # Initialize with bg_color
    object_id_map = torch.full((image_height, image_width), 0, dtype=torch.int32, device="cpu") # For instance IDs
    # Initialize with a background ID, e.g., 0 or -1 if 0 is a valid object ID.
    # The original gaussian-grouping uses 0 for background/unassigned in rendered_objects.
    
    # The rendered_image is already initialized with bg_color (torch.Tensor)
    # Removed redundant and problematic assignment from bg_color_np
    
    # Using a depth buffer to correctly assign object_id for the foremost Gaussian
    # Initialize depth buffer with infinity
    depth_buffer = torch.full((image_height, image_width), float('inf'), device="cpu")

    gaussian_depths = []
    view_matrix_np = viewpoint_camera.world_view_transform.cpu().numpy()
    for i, gau in enumerate(gaussians):
        pos_w = np.append(gau.pos, 1.0)
        pos_v = view_matrix_np @ pos_w
        gaussian_depths.append((pos_v[2], i))
    
    # Sort Gaussians by depth (far to near for alpha blending)
    sorted_gaussians_indices = [idx for depth, idx in sorted(gaussian_depths, key=lambda x: x[0], reverse=True)]

    for gau_idx in sorted_gaussians_indices:
        gau = gaussians[gau_idx]
        cov2d, depth_gau, pos_v = gau.get_cov2d_and_depth(viewpoint_camera)

        if depth_gau < viewpoint_camera.znear or depth_gau > viewpoint_camera.zfar:
            continue

        fx = viewpoint_camera.image_width / (2 * viewpoint_camera.tanfovx)
        fy = viewpoint_camera.image_height / (2 * viewpoint_camera.tanfovy)
        cx = viewpoint_camera.image_width / 2
        cy = viewpoint_camera.image_height / 2

        x_ndc = (pos_v[0] * fx) / pos_v[2]
        y_ndc = (pos_v[1] * fy) / pos_v[2]
        u_mean = x_ndc + cx
        # Adjust v_mean for top-left origin: (height - (y_ndc + cy)) or (cy - y_ndc)
        # If y_ndc is positive upwards from center, then cy - y_ndc is from top.
        # Let\u0027s assume standard image coords: y positive downwards.
        # If pos_v[1] is positive upwards in camera frame, then y_ndc is positive upwards.
        # Screen v = cy - y_ndc (if cy is center and v is from top)
        # Or, if y_ndc is already in screen space (y down), then v_mean = y_ndc + cy is wrong.
        # The original CUDA rasterizer uses tanfov for projection, which implies a coordinate system.
        # Let\u0027s stick to the previous u_mean = x_ndc + cx, v_mean = y_ndc + cy and assume it works out or adjust later.
        # thomasantony/splat uses: (ndc_x+1)*W/2, (1-ndc_y)*H/2. If ndc_y is -1 (bottom) to 1 (top).
        # Our y_ndc = (pos_v[1]*fy)/pos_v[2]. If pos_v[1] is up, y_ndc is up.
        # So v_pixel = H/2 - y_ndc_pixel_scale = H/2 - (pos_v[1]*fy)/pos_v[2]
        v_mean = cy - y_ndc # Assuming y_ndc was calculated with y positive upwards from optical axis.

        try:
            eigvals, eigvecs = np.linalg.eigh(cov2d)
        except np.linalg.LinAlgError: continue
        eigvals = np.maximum(eigvals, 1e-6)
        r1, r2 = 3 * np.sqrt(eigvals[0]), 3 * np.sqrt(eigvals[1])

        # More precise bounding box using rotated ellipse
        # For simplicity, using axis-aligned bounding box based on max radius for now
        max_radius = max(r1,r2) # A simpler but larger bounding box
        min_u = int(np.floor(u_mean - max_radius))
        max_u = int(np.ceil(u_mean + max_radius))
        min_v = int(np.floor(v_mean - max_radius))
        max_v = int(np.ceil(v_mean + max_radius))

        min_u = max(0, min_u); max_u = min(image_width, max_u)
        min_v = max(0, min_v); max_v = min(image_height, max_v)

        if min_u >= max_u or min_v >= max_v: continue

        cam_center_w = viewpoint_camera.camera_center.cpu().numpy()
        view_dir_w = gau.pos - cam_center_w
        color = gau.get_color(view_dir_w)
        
        try: inv_cov2d = np.linalg.inv(cov2d)
        except np.linalg.LinAlgError: continue

        for v_px in range(min_v, max_v):
            for u_px in range(min_u, max_u):
                x = u_px + 0.5 - u_mean
                y = v_px + 0.5 - v_mean
                pt_vec = np.array([x,y])
                mahalanobis_sq = pt_vec.T @ inv_cov2d @ pt_vec
                alpha_val = gau.opacity * np.exp(-0.5 * mahalanobis_sq)
                
                if alpha_val < 1e-3: continue
                
                # Depth test for object ID assignment (closest Gaussian writes its ID)
                # Alpha blending is far-to-near, so current_pixel_color is accumulation from farther Gs.
                # For object ID, we want the ID of the Gaussian that is most visible / closest at this pixel.
                # The CUDA rasterizer does per-pixel sorting/accumulation.
                # With global sort and far-to-near blending, the *last* Gaussian (closest) to significantly contribute
                # should ideally set the object ID if its alpha is high enough and it passes a depth test against prior *significant* contributions.
                # A simpler approach for now: if this Gaussian is closer than what is in depth_buffer AND its alpha is significant,
                # update color, depth_buffer, and object_id_map.
                
                # We are blending far-to-near. If we use a traditional z-buffer for object_id:
                # The object_id should be from the Gaussian that is finally visible (highest accumulated alpha from front, or simply closest significant).
                # Let\u0027s use a simple depth test for object_id. If this gau is closer, it writes its ID.
                # This might not be perfect with transparency but is a common approach.
                
                # Perform alpha blending for color
                current_pixel_color_tensor = rendered_image[:, v_px, u_px]
                current_pixel_color_np = current_pixel_color_tensor.numpy()
                blended_color = alpha_val * color + (1 - alpha_val) * current_pixel_color_np
                rendered_image[:, v_px, u_px] = torch.from_numpy(blended_color)

                # Update object_id_map if this Gaussian is closer and significantly contributes
                # We need a threshold for alpha_val to be considered significant for object ID.
                # And it must be closer than the current occupant of the depth buffer for this pixel.
                if alpha_val > 1e-3 and depth_gau < depth_buffer[v_px, u_px]: # Alpha threshold can be tuned
                    depth_buffer[v_px, u_px] = depth_gau
                    object_id_map[v_px, u_px] = gau.object_id

    return {"render": rendered_image, "render_object": object_id_map}

