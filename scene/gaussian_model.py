import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH #, SH2RGB
# from simple_knn._C import distCUDA2 # Commented out for CPU version
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial import KDTree

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # (N, 1, 3)
        self._features_rest = torch.empty(0) # (N, C, 3) C = (max_sh_degree+1)**2 - 1
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._objects_dc = torch.empty(0) # This stores object IDs, shape (N, 1, 1) or similar
        self.num_objects = 16 # Default, can be overridden
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        dc = self._features_dc.squeeze(1) # (N, 3)
        rest = self._features_rest.reshape(self._features_rest.shape[0], -1) # (N, C*3)
        return torch.cat((dc, rest), dim=1) # (N, 3 + C*3)
    
    @property
    def get_objects(self):
        if self._objects_dc.nelement() == 0:
             return torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device, dtype=torch.int32)
        if self._objects_dc.shape[1] > 1 and self._objects_dc.shape[2] == 1:
            return torch.argmax(self._objects_dc.squeeze(-1), dim=1, keepdim=True).int()
        elif self._objects_dc.shape[1] == 1 and self._objects_dc.shape[2] == 1:
            return self._objects_dc.squeeze(-1).int()
        else:
            return self._objects_dc.int()

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_objects: int = 16):
        self.spatial_lr_scale = spatial_lr_scale
        self.num_objects = num_objects
        # Ensure tensors are created on CPU if CUDA is not available or not intended for this part
        # For now, assuming this function might still be called in a context where CUDA is attempted.
        # If strictly CPU, .cuda() calls should be replaced with .to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).to(device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float32).to(device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32).to(device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        if hasattr(pcd, 'object_ids') and pcd.object_ids is not None:
            print("Using object IDs from PCD.")
            fused_object_ids = torch.tensor(np.asarray(pcd.object_ids), dtype=torch.int32).float().to(device).unsqueeze(-1)
        else:
            print(f"Randomly initializing object IDs (1 to {self.num_objects}).")
            raw_ids = torch.randint(1, self.num_objects + 1, (fused_point_cloud.shape[0], 1), device=device).float()
            fused_object_ids = raw_ids.unsqueeze(-1)
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # distCUDA2 is commented out. Need a CPU alternative if this path is used.
        # For now, initialize scales differently if distCUDA2 is not available.
        # For example, based on average nearest neighbor distance using scipy.spatial.KDTree
        if fused_point_cloud.shape[0] > 1:
            try:
                from scipy.spatial import cKDTree
                kdtree = cKDTree(fused_point_cloud.cpu().numpy())
                # Query for 2 nearest neighbors (first is self)
                dist, _ = kdtree.query(fused_point_cloud.cpu().numpy(), k=min(2, fused_point_cloud.shape[0]))
                if dist.ndim > 1 and dist.shape[1] > 1:
                     dist2 = torch.tensor(dist[:, 1]**2, dtype=torch.float32).to(device)
                else: # Handle case with very few points
                    dist2 = torch.ones(fused_point_cloud.shape[0], dtype=torch.float32).to(device) * 0.01**2
                dist2 = torch.clamp_min(dist2, 0.0000001)
            except ImportError:
                print("Warning: scipy not available for KDTree based scale initialization. Using default small scale.")
                dist2 = torch.ones(fused_point_cloud.shape[0], dtype=torch.float32).to(device) * 0.01**2 # Default small scale
        else:
            dist2 = torch.ones(fused_point_cloud.shape[0], dtype=torch.float32).to(device) * 0.01**2

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self._objects_dc = nn.Parameter(fused_object_ids.requires_grad_(False))

    def training_setup(self, training_args):
        # This setup is for training, which we are simplifying away for CPU version.
        # If training is ever adapted, device handling needs care.
        device = self._xyz.device # Use device of existing tensors
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self._objects_dc.requires_grad:
             l.append({'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def save_ply_with_object_id(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) # Normals are not used by 3DGS but PLY format often includes them
        
        # features_dc is (N,1,3), transpose to (N,3,1), then flatten to (N,3)
        f_dc = self._features_dc.detach().cpu().transpose(1,2).flatten(start_dim=1).numpy()
        
        # features_rest is (N,C,3), transpose to (N,3,C), then flatten to (N, 3*C)
        f_rest = self._features_rest.detach().cpu().transpose(1,2).flatten(start_dim=1).numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        object_ids_tensor = self.get_objects.detach().cpu() # (N,1) or (N,)
        if object_ids_tensor.ndim == 1:
            object_ids = object_ids_tensor.unsqueeze(1).numpy() # Ensure (N,1)
        else:
            object_ids = object_ids_tensor.numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_with_object_id()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # Ensure all parts are (N, K) for concatenation
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, object_ids), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"PLY file saved to {path} with object IDs.")

    def construct_list_of_attributes_with_object_id(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # f_dc is 3 features
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        
        # f_rest is C*3 features, where C = (max_sh_degree+1)**2 - 1
        num_rest_coeffs = (self.max_sh_degree + 1)**2 - 1
        for i in range(num_rest_coeffs * 3):
            l.append('f_rest_{}'.format(i))
            
        l.append('opacity')
        # scale is 3 features
        for i in range(3):
            l.append('scale_{}'.format(i))
        # rotation is 4 features
        for i in range(4):
            l.append('rot_{}'.format(i))
        l.append('object_id')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().transpose(1,2).flatten(start_dim=1).numpy()
        f_rest = self._features_rest.detach().cpu().transpose(1,2).flatten(start_dim=1).numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        num_rest_coeffs = (self.max_sh_degree + 1)**2 - 1
        for i in range(num_rest_coeffs * 3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if not self._xyz.requires_grad: return
        device = viewspace_point_tensor.device
        if self.xyz_gradient_accum.device != device:
            self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
            self.denom = self.denom.to(device)
            self.max_radii2D = self.max_radii2D.to(device)

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _prune_points(self, mask):
        valid_points_mask = ~mask
        self._xyz = nn.Parameter(self._xyz[valid_points_mask].requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask].requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask].requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[valid_points_mask].requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[valid_points_mask].requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[valid_points_mask].requires_grad_(True))
        if self._objects_dc.nelement() > 0:
            self._objects_dc = nn.Parameter(self._objects_dc[valid_points_mask].requires_grad_(self._objects_dc.requires_grad))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return valid_points_mask

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self._objects_dc.nelement() > 0:
            new_objects_dc = self._objects_dc[selected_pts_mask]
        else:
            new_objects_dc = torch.empty(0, device=self._xyz.device)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], dtype=bool, device=self._xyz.device)))
        self._prune_points(prune_filter)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self._objects_dc.nelement() > 0:
            new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)
        else:
            new_objects_dc = torch.empty(0, device=self._xyz.device)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], dtype=bool, device=self._xyz.device)))
        self._prune_points(prune_filter)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if new_objects_dc.nelement() > 0:
             d["obj_dc"] = new_objects_dc

        optimizable_tensors = {} 
        for group in self.optimizer.param_groups:
            if group["name"] not in d: continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(d[group["name"]])), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(d[group["name"]])), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], d[group["name"]]), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], d[group["name"]]), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "obj_dc" in optimizable_tensors and new_objects_dc.nelement() > 0:
            self._objects_dc = optimizable_tensors["obj_dc"]
        elif new_objects_dc.nelement() > 0:
            self._objects_dc = nn.Parameter(torch.cat((self._objects_dc, new_objects_dc), dim=0).requires_grad_(self._objects_dc.requires_grad))

        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)



    def update_learning_rate(self, iteration):
        """Update learning rate for all optimizer groups"""
        # Простая реализация - можно расширить при необходимости
        pass
