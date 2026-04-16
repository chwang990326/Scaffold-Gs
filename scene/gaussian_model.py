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

import copy
import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding


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


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        # [新增] 用于存储语义特征 (512 或 128 维)
        self._semantic_features = torch.empty(0, device="cuda") 
        self.semantic_adapter = None
        self.semantic_adapter_scheduler_args = None
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    @property
    def semantic_features(self):
        return self._semantic_features

    @semantic_features.setter
    def semantic_features(self, features):
        self._semantic_features = features

    def init_semantic_adapter(self, semantic_dim: int) -> None:
        if semantic_dim <= 0:
            self.semantic_adapter = None
            return

        needs_init = (
            self.semantic_adapter is None or
            not hasattr(self.semantic_adapter, "in_features") or
            not hasattr(self.semantic_adapter, "out_features") or
            self.semantic_adapter.in_features != semantic_dim or
            self.semantic_adapter.out_features != self.feat_dim
        )

        if needs_init:
            self.semantic_adapter = nn.Linear(semantic_dim, self.feat_dim).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.semantic_adapter is not None:
            self.semantic_adapter.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.semantic_adapter is not None:
            self.semantic_adapter.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                  
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._anchor_feat,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.offset_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_features, # 增加语义捕获
            self.semantic_adapter.state_dict() if self.semantic_adapter is not None else None,
        )

    def _build_aligned_semantic_features(self, num_anchors: int, source_features=None):
        if source_features is not None and source_features.dim() == 2 and source_features.shape[1] > 0:
            aligned_features = torch.zeros((num_anchors, source_features.shape[1]), device="cuda")
            min_len = min(num_anchors, source_features.shape[0])
            aligned_features[:min_len] = source_features[:min_len]
            return aligned_features
        return torch.zeros((num_anchors, 128), device="cuda")

    def _load_optimizer_state_compat(self, opt_dict):
        if opt_dict is None:
            return

        try:
            self.optimizer.load_state_dict(opt_dict)
            return
        except ValueError:
            current_state = self.optimizer.state_dict()
            saved_groups = opt_dict.get("param_groups", [])
            current_groups = current_state.get("param_groups", [])
            saved_names = [group.get("name") for group in saved_groups]
            current_names = [group.get("name") for group in current_groups]

            if len(saved_groups) > len(current_groups) or saved_names != current_names[:len(saved_names)]:
                raise

            merged_state = copy.deepcopy(opt_dict)
            merged_state["param_groups"] = copy.deepcopy(saved_groups)
            merged_state["param_groups"].extend(copy.deepcopy(current_groups[len(saved_groups):]))
            self.optimizer.load_state_dict(merged_state)
    
    def restore(self, model_args, training_args):
        semantic_adapter_state = None
        previous_semantic = self._semantic_features if self._semantic_features.dim() == 2 else None

        # [关键修复] 兼容检查点加载，防止尝试加载之前没有保存语义的检查点时发生 unpack 报错
        if len(model_args) == 12:
            (self._anchor, 
            self._offset,
            self._anchor_feat,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            offset_denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._semantic_features,
            semantic_adapter_state) = model_args
        elif len(model_args) == 11:
            (self._anchor, 
            self._offset,
            self._anchor_feat,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            offset_denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._semantic_features) = model_args # 恢复语义
        else:
            # 向后兼容没有语义的原始 Scaffold-GS checkpoint
            (self._anchor, 
            self._offset,
            self._anchor_feat,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            offset_denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            self._semantic_features = self._build_aligned_semantic_features(self._anchor.shape[0], previous_semantic)

        if self._semantic_features.shape[0] != self._anchor.shape[0]:
            self._semantic_features = self._build_aligned_semantic_features(self._anchor.shape[0], self._semantic_features)

        if semantic_adapter_state is not None:
            self.init_semantic_adapter(semantic_adapter_state["weight"].shape[1])
            self.semantic_adapter.load_state_dict(semantic_adapter_state)
        elif self._semantic_features.dim() == 2 and self._semantic_features.shape[1] > 0:
            self.init_semantic_adapter(self._semantic_features.shape[1])

        self.training_setup(training_args)
        self.offset_denom = offset_denom
        self._load_optimizer_state_compat(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]

        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})
        
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        if self.semantic_adapter is not None:
            l.append({'params': self.semantic_adapter.parameters(), 'lr': training_args.semantic_adapter_lr, "name": "semantic_adapter"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.offset_lr_delay_mult,
                                                        max_steps=training_args.offset_lr_max_steps)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                        lr_final=training_args.mlp_opacity_lr_final,
                                                        lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                        max_steps=training_args.mlp_opacity_lr_max_steps)
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                lr_final=training_args.mlp_featurebank_lr_final,
                                                                lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                                lr_final=training_args.appearance_lr_final,
                                                                lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                                max_steps=training_args.appearance_lr_max_steps)
        if self.semantic_adapter is not None:
            self.semantic_adapter_scheduler_args = get_expon_lr_func(lr_init=training_args.semantic_adapter_lr,
                                                                    lr_final=training_args.semantic_adapter_lr,
                                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                param_group['lr'] = self.offset_scheduler_args(iteration)
            if param_group["name"] == "anchor":
                param_group['lr'] = self.anchor_scheduler_args(iteration)
            if param_group["name"] == "mlp_opacity":
                param_group['lr'] = self.mlp_opacity_scheduler_args(iteration)
            if param_group["name"] == "mlp_cov":
                param_group['lr'] = self.mlp_cov_scheduler_args(iteration)
            if param_group["name"] == "mlp_color":
                param_group['lr'] = self.mlp_color_scheduler_args(iteration)
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                param_group['lr'] = self.mlp_featurebank_scheduler_args(iteration)
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                param_group['lr'] = self.appearance_scheduler_args(iteration)
            if self.semantic_adapter is not None and param_group["name"] == "semantic_adapter":
                param_group['lr'] = self.semantic_adapter_scheduler_args(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)
        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")], key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        anchor_feat_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")], key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offset_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")], key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'embedding' in group['name'] or group["name"] == "semantic_adapter":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        # [新增] 同步增加语义特征
        # [关键修复] 当 `_semantic_features` 是空或一维时，直接赋值，避免 torch.cat 对 1D 和 2D 拼接时报错
        if "semantic_features" in tensors_dict:
            if self._semantic_features.dim() < 2 or self._semantic_features.shape[0] == 0:
                self._semantic_features = tensors_dict["semantic_features"]
            else:
                self._semantic_features = torch.cat((self._semantic_features, tensors_dict["semantic_features"]), dim=0)

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1
        anchor_visible_mask_expanded = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask_expanded] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'embedding' in group['name'] or group["name"] == "semantic_adapter":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    group["params"][0].data[:,3:] = torch.clamp(group["params"][0].data[:,3:], max=0.05)
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    group["params"][0].data[:,3:] = torch.clamp(group["params"][0].data[:,3:], max=0.05)
                optimizable_tensors[group["name"]] = group["params"][0]
        
        # [新增] 同步删除语义特征
        if self._semantic_features.shape[0] > 0:
            self._semantic_features = self._semantic_features[mask]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = torch.logical_and((grads >= cur_threshold), offset_mask)
            rand_mask = (torch.rand_like(candidate_mask.float()) > (0.5**(i+1))).cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc > 0:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            elif i > 0 and length_inc == 0:
                continue

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            cur_size = self.voxel_size * (self.update_init_factor // (self.update_hierachy_factor**i))
            grid_coords = torch.round(self.get_anchor / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for j in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[j*chunk_size:(j+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = ~reduce(torch.logical_or, remove_duplicates_list)
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.log(torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device="cuda").float(); new_rotation[:,0] = 1.0
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), device="cuda"))
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_offsets = torch.zeros((candidate_anchor.shape[0], self.n_offsets, 3), device="cuda").float()

                # [新增] 为新锚点计算语义特征（继承父代）
                # =================== [FIX START: 维度自适应与安全性优化] ===================
                # 获取当前特征维度 (默认回退 128 而不是 512，与修改后的维度一致)
                feat_dim = self._semantic_features.shape[1] if (self._semantic_features.dim() > 1 and self._semantic_features.shape[0] > 0) else 128
                
                new_semantic = torch.zeros((candidate_anchor.shape[0], feat_dim), device='cuda')
                
                if self._semantic_features.shape[0] > 0:
                    candidate_indices = torch.nonzero(candidate_mask).view(-1) # 获取所有需要生长的索引
                    parent_anchor_indices = candidate_indices // self.n_offsets # 计算属于哪个父锚点
                    parent_semantic = self._semantic_features[parent_anchor_indices] # 直接索引提取
                    
                    # [关键修复] 使用 feat_dim 替代硬编码维度
                    new_semantic = scatter_max(parent_semantic, inverse_indices.unsqueeze(1).expand(-1, feat_dim), dim=0)[0][remove_duplicates]
                # =================== [FIX END] ===================

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "semantic_features": new_semantic,
                }
                
                self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda')], dim=0)
                self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda')], dim=0)
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor, self._scaling, self._rotation = optimizable_tensors["anchor"], optimizable_tensors["scaling"], optimizable_tensors["rotation"]
                self._anchor_feat, self._offset, self._opacity = optimizable_tensors["anchor_feat"], optimizable_tensors["offset"], optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        grads = (self.offset_gradient_accum / self.offset_denom).nan_to_num(0.0)
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        self.offset_denom[offset_mask] = 0
        padding = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1], dtype=torch.int32, device='cuda')
        self.offset_denom = torch.cat([self.offset_denom, padding], dim=0)
        self.offset_gradient_accum[offset_mask] = 0
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, torch.zeros_like(padding)], dim=0)
        
        prune_mask = torch.logical_and((self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1), (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1))
        
        self.offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask].view([-1, 1])
        self.offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask].view([-1, 1])
        
        if prune_mask.any():
            self.opacity_accum[~prune_mask] # Just to keep logic flow, actual reset below
        self.opacity_accum[~prune_mask] = self.opacity_accum[~prune_mask] # Placeholder
        
        # Reset accumulators for remaining anchors
        remaining_mask = ~prune_mask
        self.opacity_accum = torch.zeros((remaining_mask.sum(), 1), device='cuda')
        self.anchor_demon = torch.zeros((remaining_mask.sum(), 1), device='cuda')

        if prune_mask.any():
            self.prune_anchor(prune_mask)
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            for name, mlp, inp_dim in [('opacity_mlp', self.mlp_opacity, self.feat_dim+3+self.opacity_dist_dim),
                                       ('cov_mlp', self.mlp_cov, self.feat_dim+3+self.cov_dist_dim),
                                       ('color_mlp', self.mlp_color, self.feat_dim+3+self.color_dist_dim+self.appearance_dim)]:
                mlp.eval()
                torch.jit.trace(mlp, torch.rand(1, inp_dim).cuda()).save(os.path.join(path, f'{name}.pt'))
                mlp.train()
            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                torch.jit.trace(self.mlp_feature_bank, torch.rand(1, 4).cuda()).save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()
            if self.appearance_dim > 0:
                self.embedding_appearance.eval()
                torch.jit.trace(self.embedding_appearance, torch.zeros((1,), dtype=torch.long).cuda()).save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()
        elif mode == 'unite':
            data = {'opacity_mlp': self.mlp_opacity.state_dict(), 'cov_mlp': self.mlp_cov.state_dict(), 'color_mlp': self.mlp_color.state_dict()}
            if self.use_feat_bank: data['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0: data['appearance'] = self.embedding_appearance.state_dict()
            torch.save(data, os.path.join(path, 'checkpoints.pth'))

    def load_mlp_checkpoints(self, path, mode = 'split'):
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank: self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0: self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            ckpt = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(ckpt['opacity_mlp'])
            self.mlp_cov.load_state_dict(ckpt['cov_mlp'])
            self.mlp_color.load_state_dict(ckpt['color_mlp'])
            if self.use_feat_bank: self.mlp_feature_bank.load_state_dict(ckpt['feature_bank_mlp'])
            if self.appearance_dim > 0: self.embedding_appearance.load_state_dict(ckpt['appearance'])
