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
                 semantic_num_experts: int = 4,
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
        self.semantic_num_experts = max(1, int(semantic_num_experts))

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._anchor_sem_feat = torch.empty(0)
        
        # [新增] 用于存储语义特征 (512 或 128 维)
        self._semantic_features = torch.empty(0, device="cuda") 
        self._semantic_confidence = torch.empty(0, device="cuda")
        self._semantic_cluster_ids = torch.empty(0, device="cuda", dtype=torch.long)
        self._semantic_valid_mask = torch.empty(0, device="cuda", dtype=torch.bool)
        self.semantic_routing_initialized = False
        self.semantic_adapter = None
        self.semantic_adapter_scheduler_args = None
        self._triangle_ctrl = nn.Parameter(torch.empty((0, 3, 3), device="cuda").requires_grad_(True))
        self._triangle_alpha_logit = nn.Parameter(torch.empty((0, 1), device="cuda").requires_grad_(True))
        self._triangle_thickness_logit = nn.Parameter(torch.empty((0, 1), device="cuda").requires_grad_(True))
        self._triangle_rgb_residual = nn.Parameter(torch.empty((0, 3), device="cuda").requires_grad_(True))
        self._triangle_semantic_feat = nn.Parameter(torch.empty((0, 128), device="cuda").requires_grad_(True))
        self._triangle_semantic_target = torch.empty((0, 128), device="cuda")
        self._triangle_parent_anchor_idx = torch.empty((0,), device="cuda", dtype=torch.long)
        self._triangle_confidence = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_active_mask = torch.empty((0,), device="cuda", dtype=torch.bool)
        self._triangle_init_ctrl = torch.empty((0, 3, 3), device="cuda")
        self._triangle_target_center = torch.empty((0, 3), device="cuda")
        self._triangle_view_count = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_angle_stability = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_depth_stability = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_center_weight = torch.empty((0,), device="cuda", dtype=torch.float32)
        self.triangle_initialized = False
        self.triangle_semantic_dim = 128
        self.triangle_thickness_min = 1.0
        self.triangle_thickness_max = 6.0
        self.triangle_ctrl_scheduler_args = None
        self.triangle_alpha_scheduler_args = None
        self.triangle_thickness_scheduler_args = None
        self.triangle_color_scheduler_args = None
        self.triangle_semantic_scheduler_args = None
        
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
        self.init_semantic_color_modules()

    def _create_color_mlp(self):
        return nn.Sequential(
            nn.Linear(self.feat_dim+3+self.color_dist_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def init_semantic_color_modules(self):
        self.mlp_color_fallback = self._create_color_mlp()
        self.mlp_color_experts = nn.ModuleList()

    @property
    def semantic_features(self):
        return self._semantic_features

    @semantic_features.setter
    def semantic_features(self, features):
        self._semantic_features = features

    @property
    def anchor_sem_feat(self):
        return self._anchor_sem_feat

    @anchor_sem_feat.setter
    def anchor_sem_feat(self, features):
        self._anchor_sem_feat = features

    @property
    def semantic_confidence(self):
        return self._semantic_confidence

    @semantic_confidence.setter
    def semantic_confidence(self, confidence):
        self._semantic_confidence = confidence

    @property
    def semantic_cluster_ids(self):
        return self._semantic_cluster_ids

    @semantic_cluster_ids.setter
    def semantic_cluster_ids(self, cluster_ids):
        self._semantic_cluster_ids = cluster_ids

    @property
    def semantic_valid_mask(self):
        return self._semantic_valid_mask

    @semantic_valid_mask.setter
    def semantic_valid_mask(self, valid_mask):
        self._semantic_valid_mask = valid_mask

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

    def _build_aligned_semantic_cluster_ids(self, num_anchors: int, source_ids=None):
        aligned_ids = torch.full((num_anchors,), -1, device="cuda", dtype=torch.long)
        if source_ids is not None and source_ids.dim() == 1 and source_ids.shape[0] > 0:
            min_len = min(num_anchors, source_ids.shape[0])
            aligned_ids[:min_len] = source_ids[:min_len].to(device="cuda", dtype=torch.long)
        return aligned_ids

    def _build_aligned_semantic_valid_mask(self, num_anchors: int, source_mask=None):
        aligned_mask = torch.zeros((num_anchors,), device="cuda", dtype=torch.bool)
        if source_mask is not None and source_mask.dim() == 1 and source_mask.shape[0] > 0:
            min_len = min(num_anchors, source_mask.shape[0])
            aligned_mask[:min_len] = source_mask[:min_len].to(device="cuda", dtype=torch.bool)
        return aligned_mask

    def _build_aligned_semantic_confidence(self, num_anchors: int, source_confidence=None):
        aligned_confidence = torch.zeros((num_anchors,), device="cuda", dtype=torch.float32)
        if source_confidence is not None and source_confidence.dim() == 1 and source_confidence.shape[0] > 0:
            min_len = min(num_anchors, source_confidence.shape[0])
            aligned_confidence[:min_len] = source_confidence[:min_len].to(device="cuda", dtype=torch.float32)
        return aligned_confidence

    def _build_aligned_anchor_sem_feat(self, num_anchors: int, source_features=None):
        if source_features is not None and source_features.dim() == 2 and source_features.shape[1] == self.feat_dim:
            aligned_features = torch.zeros((num_anchors, self.feat_dim), device="cuda")
            min_len = min(num_anchors, source_features.shape[0])
            aligned_features[:min_len] = source_features[:min_len].to(device="cuda", dtype=torch.float32)
            return nn.Parameter(aligned_features.requires_grad_(True))
        return nn.Parameter(torch.zeros((num_anchors, self.feat_dim), device="cuda").requires_grad_(True))

    def set_semantic_routing(self, cluster_ids=None, valid_mask=None, initialized=True):
        num_anchors = self._anchor.shape[0] if self._anchor.dim() > 0 else 0
        self._semantic_cluster_ids = self._build_aligned_semantic_cluster_ids(num_anchors, cluster_ids)
        self._semantic_valid_mask = self._build_aligned_semantic_valid_mask(num_anchors, valid_mask)
        self._semantic_cluster_ids[~self._semantic_valid_mask] = -1
        self.semantic_routing_initialized = initialized

    def get_semantic_routing_state(self):
        return {
            "semantic_cluster_ids": self._semantic_cluster_ids,
            "semantic_valid_mask": self._semantic_valid_mask,
            "semantic_confidence": self._semantic_confidence,
            "semantic_num_experts": self.semantic_num_experts,
            "semantic_routing_initialized": self.semantic_routing_initialized,
        }

    def load_semantic_routing_state(self, routing_state):
        if routing_state is None:
            default_valid_mask = None
            if (
                self._semantic_features.dim() == 2 and
                self._anchor.dim() > 0 and
                self._semantic_features.shape[0] == self._anchor.shape[0]
            ):
                default_valid_mask = self._semantic_features.abs().sum(dim=1) > 0
            self.set_semantic_routing(valid_mask=default_valid_mask, initialized=False)
            self._semantic_confidence = self._build_aligned_semantic_confidence(self._anchor.shape[0], self._semantic_confidence)
            return

        saved_num_experts = routing_state.get("semantic_num_experts")
        if saved_num_experts is not None and int(saved_num_experts) != self.semantic_num_experts:
            self.semantic_num_experts = max(1, int(saved_num_experts))
            self.init_semantic_color_modules()

        self.set_semantic_routing(
            cluster_ids=routing_state.get("semantic_cluster_ids"),
            valid_mask=routing_state.get("semantic_valid_mask"),
            initialized=bool(routing_state.get("semantic_routing_initialized", True)),
        )
        confidence_source = routing_state.get("semantic_confidence")
        if confidence_source is None:
            confidence_source = self._semantic_confidence
        self._semantic_confidence = self._build_aligned_semantic_confidence(
            self._anchor.shape[0],
            confidence_source,
        )

    def warm_start_color_modules(self, source_state_dict):
        if source_state_dict is None:
            return
        self.mlp_color_fallback.load_state_dict(source_state_dict)

    def get_visible_semantic_cluster_ids(self, visible_mask=None):
        return None

    def predict_color(self, color_input, semantic_cluster_ids=None):
        if color_input.shape[0] == 0:
            return torch.empty((0, 3 * self.n_offsets), device=color_input.device, dtype=color_input.dtype)
        return self.mlp_color_fallback(color_input)

    def reset_triangle_state(self, semantic_dim=None):
        if semantic_dim is None:
            semantic_dim = self.triangle_semantic_dim
        semantic_dim = max(1, int(semantic_dim))
        self.triangle_semantic_dim = semantic_dim
        self._triangle_ctrl = nn.Parameter(torch.empty((0, 3, 3), device="cuda").requires_grad_(True))
        self._triangle_alpha_logit = nn.Parameter(torch.empty((0, 1), device="cuda").requires_grad_(True))
        self._triangle_thickness_logit = nn.Parameter(torch.empty((0, 1), device="cuda").requires_grad_(True))
        self._triangle_rgb_residual = nn.Parameter(torch.empty((0, 3), device="cuda").requires_grad_(True))
        self._triangle_semantic_feat = nn.Parameter(torch.empty((0, semantic_dim), device="cuda").requires_grad_(True))
        self._triangle_semantic_target = torch.empty((0, semantic_dim), device="cuda")
        self._triangle_parent_anchor_idx = torch.empty((0,), device="cuda", dtype=torch.long)
        self._triangle_confidence = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_active_mask = torch.empty((0,), device="cuda", dtype=torch.bool)
        self._triangle_init_ctrl = torch.empty((0, 3, 3), device="cuda")
        self._triangle_target_center = torch.empty((0, 3), device="cuda")
        self._triangle_view_count = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_angle_stability = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_depth_stability = torch.empty((0,), device="cuda", dtype=torch.float32)
        self._triangle_center_weight = torch.empty((0,), device="cuda", dtype=torch.float32)
        self.triangle_initialized = False

    @property
    def has_boundary_triangles(self):
        if self._triangle_ctrl.dim() != 3 or self._triangle_ctrl.shape[0] == 0:
            return False
        if self._triangle_active_mask.dim() != 1 or self._triangle_active_mask.shape[0] != self._triangle_ctrl.shape[0]:
            return False
        return bool(self._triangle_active_mask.any().item())

    def set_triangle_hyperparams(self, training_args):
        self.triangle_thickness_min = float(getattr(training_args, "triangle_thickness_min", 1.0))
        self.triangle_thickness_max = max(
            self.triangle_thickness_min + 1e-3,
            float(getattr(training_args, "triangle_thickness_max", 6.0)),
        )
        self.triangle_semantic_dim = max(
            1,
            int(getattr(training_args, "triangle_feature_dim", self.triangle_semantic_dim)),
        )

    def get_triangle_alpha(self):
        return torch.sigmoid(self._triangle_alpha_logit)

    def get_triangle_thickness(self):
        if self._triangle_thickness_logit.shape[0] == 0:
            return torch.empty((0, 1), device="cuda")
        return self.triangle_thickness_min + (
            self.triangle_thickness_max - self.triangle_thickness_min
        ) * torch.sigmoid(self._triangle_thickness_logit)

    def get_active_triangle_state(self, visible_mask=None):
        if not self.has_boundary_triangles:
            return None

        active_mask = self._triangle_active_mask.clone()
        if (
            visible_mask is not None and
            visible_mask.dim() == 1 and
            self._triangle_parent_anchor_idx.shape[0] == active_mask.shape[0] and
            visible_mask.shape[0] == self.get_anchor.shape[0]
        ):
            active_mask &= visible_mask[self._triangle_parent_anchor_idx]

        if not active_mask.any():
            return None

        return {
            "ctrl": self._triangle_ctrl[active_mask],
            "alpha": self.get_triangle_alpha()[active_mask],
            "thickness": self.get_triangle_thickness()[active_mask],
            "rgb_residual": self._triangle_rgb_residual[active_mask],
            "semantic_feat": self._triangle_semantic_feat[active_mask],
            "semantic_target": self._triangle_semantic_target[active_mask],
            "parent_anchor_idx": self._triangle_parent_anchor_idx[active_mask],
            "confidence": self._triangle_confidence[active_mask],
            "target_center": self._triangle_target_center[active_mask],
            "init_ctrl": self._triangle_init_ctrl[active_mask],
            "view_count": self._triangle_view_count[active_mask],
            "angle_stability": self._triangle_angle_stability[active_mask],
            "depth_stability": self._triangle_depth_stability[active_mask],
            "center_weight": self._triangle_center_weight[active_mask],
            "active_mask": active_mask,
        }

    def get_triangle_state(self):
        return {
            "triangle_ctrl": self._triangle_ctrl.detach(),
            "triangle_alpha_logit": self._triangle_alpha_logit.detach(),
            "triangle_thickness_logit": self._triangle_thickness_logit.detach(),
            "triangle_rgb_residual": self._triangle_rgb_residual.detach(),
            "triangle_semantic_feat": self._triangle_semantic_feat.detach(),
            "triangle_semantic_target": self._triangle_semantic_target.detach(),
            "triangle_parent_anchor_idx": self._triangle_parent_anchor_idx.detach(),
            "triangle_confidence": self._triangle_confidence.detach(),
            "triangle_active_mask": self._triangle_active_mask.detach(),
            "triangle_init_ctrl": self._triangle_init_ctrl.detach(),
            "triangle_target_center": self._triangle_target_center.detach(),
            "triangle_view_count": self._triangle_view_count.detach(),
            "triangle_angle_stability": self._triangle_angle_stability.detach(),
            "triangle_depth_stability": self._triangle_depth_stability.detach(),
            "triangle_center_weight": self._triangle_center_weight.detach(),
            "triangle_semantic_dim": self.triangle_semantic_dim,
            "triangle_initialized": self.triangle_initialized,
            "triangle_thickness_min": self.triangle_thickness_min,
            "triangle_thickness_max": self.triangle_thickness_max,
        }

    def load_triangle_state(self, triangle_state):
        if triangle_state is None:
            self.reset_triangle_state(self.triangle_semantic_dim)
            return

        semantic_feat = triangle_state.get("triangle_semantic_feat")
        semantic_dim = semantic_feat.shape[1] if semantic_feat is not None and semantic_feat.dim() == 2 else triangle_state.get("triangle_semantic_dim", self.triangle_semantic_dim)
        self.reset_triangle_state(semantic_dim)
        self.triangle_thickness_min = float(triangle_state.get("triangle_thickness_min", self.triangle_thickness_min))
        self.triangle_thickness_max = float(triangle_state.get("triangle_thickness_max", self.triangle_thickness_max))

        self._triangle_ctrl = nn.Parameter(triangle_state.get("triangle_ctrl", self._triangle_ctrl).detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._triangle_alpha_logit = nn.Parameter(triangle_state.get("triangle_alpha_logit", self._triangle_alpha_logit).detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._triangle_thickness_logit = nn.Parameter(triangle_state.get("triangle_thickness_logit", self._triangle_thickness_logit).detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._triangle_rgb_residual = nn.Parameter(triangle_state.get("triangle_rgb_residual", self._triangle_rgb_residual).detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._triangle_semantic_feat = nn.Parameter(triangle_state.get("triangle_semantic_feat", self._triangle_semantic_feat).detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        self._triangle_semantic_target = triangle_state.get("triangle_semantic_target", self._triangle_semantic_target).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_parent_anchor_idx = triangle_state.get("triangle_parent_anchor_idx", self._triangle_parent_anchor_idx).detach().to(device="cuda", dtype=torch.long)
        self._triangle_confidence = triangle_state.get("triangle_confidence", self._triangle_confidence).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_active_mask = triangle_state.get("triangle_active_mask", self._triangle_active_mask).detach().to(device="cuda", dtype=torch.bool)
        self._triangle_init_ctrl = triangle_state.get("triangle_init_ctrl", self._triangle_init_ctrl).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_target_center = triangle_state.get("triangle_target_center", self._triangle_target_center).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_view_count = triangle_state.get("triangle_view_count", self._triangle_view_count).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_angle_stability = triangle_state.get("triangle_angle_stability", self._triangle_angle_stability).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_depth_stability = triangle_state.get("triangle_depth_stability", self._triangle_depth_stability).detach().to(device="cuda", dtype=torch.float32)
        self._triangle_center_weight = triangle_state.get("triangle_center_weight", self._triangle_center_weight).detach().to(device="cuda", dtype=torch.float32)
        self.triangle_initialized = bool(triangle_state.get("triangle_initialized", self._triangle_ctrl.shape[0] > 0))

    def initialize_boundary_triangles(self, boundary_candidates, training_args, logger=None):
        self.set_triangle_hyperparams(training_args)
        if boundary_candidates is None:
            self.reset_triangle_state(self.triangle_semantic_dim)
            self.triangle_initialized = True
            if logger is not None:
                logger.info("[TRIANGLE] Boundary candidate cache is empty. Triangle branch stays inactive.")
            return 0

        parent_anchor_indices = boundary_candidates.get("parent_anchor_indices")
        if parent_anchor_indices is None or parent_anchor_indices.numel() == 0:
            self.reset_triangle_state(self.triangle_semantic_dim)
            self.triangle_initialized = True
            if logger is not None:
                logger.info("[TRIANGLE] No valid boundary candidates passed the confidence filter.")
            return 0

        parent_anchor_indices = parent_anchor_indices.to(device="cuda", dtype=torch.long)
        valid_mask = (parent_anchor_indices >= 0) & (parent_anchor_indices < self.get_anchor.shape[0])
        confidence = boundary_candidates.get("confidence", torch.zeros_like(parent_anchor_indices, dtype=torch.float32)).to(device="cuda", dtype=torch.float32)
        view_count = boundary_candidates.get("view_counts", torch.zeros_like(confidence)).to(device="cuda", dtype=torch.float32)
        valid_mask &= confidence >= float(getattr(training_args, "triangle_confidence_threshold", 0.55))
        valid_mask &= view_count >= float(getattr(training_args, "triangle_min_views", 2))

        if not valid_mask.any():
            self.reset_triangle_state(self.triangle_semantic_dim)
            self.triangle_initialized = True
            if logger is not None:
                logger.info("[TRIANGLE] Boundary candidates exist, but none satisfy the triangle activation rule.")
            return 0

        raw_positions = boundary_candidates.get("positions")
        if raw_positions is None:
            raw_positions = self.get_anchor[parent_anchor_indices]
        raw_semantic_features = boundary_candidates.get("semantic_features")
        if raw_semantic_features is None:
            raw_semantic_features = torch.zeros((parent_anchor_indices.shape[0], self.triangle_semantic_dim), device="cuda")
        raw_angle_stability = boundary_candidates.get("angle_stability")
        if raw_angle_stability is None:
            raw_angle_stability = torch.ones_like(confidence)
        raw_depth_stability = boundary_candidates.get("depth_stability")
        if raw_depth_stability is None:
            raw_depth_stability = torch.ones_like(confidence)
        raw_center_weight = boundary_candidates.get("center_weight")
        if raw_center_weight is None:
            raw_center_weight = torch.ones_like(confidence)

        parent_anchor_indices = parent_anchor_indices[valid_mask]
        confidence = confidence[valid_mask]
        view_count = view_count[valid_mask]
        positions = raw_positions.to(device="cuda", dtype=torch.float32)[valid_mask]
        semantic_features = raw_semantic_features.to(device="cuda", dtype=torch.float32)[valid_mask]
        angle_stability = raw_angle_stability.to(device="cuda", dtype=torch.float32)[valid_mask]
        depth_stability = raw_depth_stability.to(device="cuda", dtype=torch.float32)[valid_mask]
        center_weight = raw_center_weight.to(device="cuda", dtype=torch.float32)[valid_mask]

        max_candidates = int(getattr(training_args, "triangle_max_candidates", 0))
        if max_candidates > 0 and parent_anchor_indices.numel() > max_candidates:
            topk = torch.topk(confidence, k=max_candidates, largest=True).indices
            parent_anchor_indices = parent_anchor_indices[topk]
            confidence = confidence[topk]
            view_count = view_count[topk]
            positions = positions[topk]
            semantic_features = semantic_features[topk]
            angle_stability = angle_stability[topk]
            depth_stability = depth_stability[topk]
            center_weight = center_weight[topk]

        semantic_dim = semantic_features.shape[1] if semantic_features.dim() == 2 and semantic_features.shape[1] > 0 else self.triangle_semantic_dim
        self.reset_triangle_state(semantic_dim)
        self.set_triangle_hyperparams(training_args)

        parent_scaling = self.get_scaling[parent_anchor_indices, :3]
        parent_offsets = self._offset[parent_anchor_indices] * parent_scaling.unsqueeze(1)
        dir_a = parent_offsets[:, 0, :]
        dir_b = parent_offsets[:, min(1, self.n_offsets - 1), :]

        fallback_a = torch.tensor([1.0, 0.0, 0.0], device="cuda").expand_as(dir_a)
        fallback_b = torch.tensor([0.0, 1.0, 0.0], device="cuda").expand_as(dir_b)
        dir_a_norm = dir_a.norm(dim=-1, keepdim=True)
        dir_a = torch.where(dir_a_norm > 1e-6, dir_a / dir_a_norm.clamp_min(1e-6), fallback_a)
        dir_b = dir_b - (dir_b * dir_a).sum(dim=-1, keepdim=True) * dir_a
        dir_b_norm = dir_b.norm(dim=-1, keepdim=True)
        dir_b = torch.where(dir_b_norm > 1e-6, dir_b / dir_b_norm.clamp_min(1e-6), fallback_b)

        base_scale = parent_scaling.mean(dim=-1, keepdim=True).clamp_min(max(self.voxel_size, 1e-4) * 0.25)
        base_scale = base_scale * float(getattr(training_args, "triangle_init_scale", 1.0))

        ctrl = torch.stack([
            positions + base_scale * dir_a,
            positions - base_scale * dir_a,
            positions + base_scale * dir_b,
        ], dim=1)

        alpha_init = torch.clamp(
            confidence.unsqueeze(1) * float(getattr(training_args, "triangle_init_alpha", 0.35)),
            1e-3,
            0.95,
        )
        thickness_value = float(getattr(training_args, "triangle_init_thickness", 2.0))
        thickness_norm = (thickness_value - self.triangle_thickness_min) / max(self.triangle_thickness_max - self.triangle_thickness_min, 1e-6)
        thickness_norm = min(max(thickness_norm, 1e-3), 1.0 - 1e-3)
        thickness_logit = inverse_sigmoid(torch.full((ctrl.shape[0], 1), thickness_norm, device="cuda"))

        self._triangle_ctrl = nn.Parameter(ctrl.detach().requires_grad_(True))
        self._triangle_alpha_logit = nn.Parameter(inverse_sigmoid(alpha_init).detach().requires_grad_(True))
        self._triangle_thickness_logit = nn.Parameter(thickness_logit.detach().requires_grad_(True))
        self._triangle_rgb_residual = nn.Parameter(torch.zeros((ctrl.shape[0], 3), device="cuda").requires_grad_(True))
        self._triangle_semantic_feat = nn.Parameter(semantic_features.detach().clone().requires_grad_(True))
        self._triangle_semantic_target = semantic_features.detach().clone()
        self._triangle_parent_anchor_idx = parent_anchor_indices.detach().clone()
        self._triangle_confidence = confidence.detach().clone()
        self._triangle_active_mask = torch.ones((ctrl.shape[0],), device="cuda", dtype=torch.bool)
        self._triangle_init_ctrl = ctrl.detach().clone()
        self._triangle_target_center = positions.detach().clone()
        self._triangle_view_count = view_count.detach().clone()
        self._triangle_angle_stability = angle_stability.detach().clone()
        self._triangle_depth_stability = depth_stability.detach().clone()
        self._triangle_center_weight = center_weight.detach().clone()
        self.triangle_initialized = True

        if logger is not None:
            logger.info(
                f"[TRIANGLE] Initialized {ctrl.shape[0]} sparse boundary triangles "
                f"(semantic_dim={self.triangle_semantic_dim})."
            )
        return ctrl.shape[0]

    def set_main_branch_trainable(self, trainable=True):
        main_params = [
            self._anchor,
            self._offset,
            self._anchor_feat,
            self._anchor_sem_feat,
            self._opacity,
            self._scaling,
            self._rotation,
        ]
        for param in main_params:
            param.requires_grad_(trainable)
        for module in [self.mlp_opacity, self.mlp_cov, self.mlp_color_fallback]:
            for param in module.parameters():
                param.requires_grad_(trainable)
        if self.semantic_adapter is not None:
            for param in self.semantic_adapter.parameters():
                param.requires_grad_(trainable)
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            for param in self.embedding_appearance.parameters():
                param.requires_grad_(trainable)
        if self.use_feat_bank:
            for param in self.mlp_feature_bank.parameters():
                param.requires_grad_(trainable)

    def set_triangle_branch_trainable(self, trainable=True):
        for param in [
            self._triangle_ctrl,
            self._triangle_alpha_logit,
            self._triangle_thickness_logit,
            self._triangle_rgb_residual,
            self._triangle_semantic_feat,
        ]:
            param.requires_grad_(trainable)

    def _configure_triangle_scheduler_args(self, training_args):
        self.set_triangle_hyperparams(training_args)
        self.triangle_ctrl_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "triangle_ctrl_lr_init", 0.0005),
            lr_final=getattr(training_args, "triangle_ctrl_lr_final", 0.00005),
            lr_delay_mult=getattr(training_args, "triangle_ctrl_lr_delay_mult", 0.01),
            max_steps=getattr(training_args, "triangle_ctrl_lr_max_steps", getattr(training_args, "iterations", 30000)),
        )
        self.triangle_alpha_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "triangle_alpha_lr", 0.0005),
            lr_final=getattr(training_args, "triangle_alpha_lr", 0.0005),
            max_steps=getattr(training_args, "triangle_ctrl_lr_max_steps", getattr(training_args, "iterations", 30000)),
        )
        self.triangle_thickness_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "triangle_thickness_lr", 0.0005),
            lr_final=getattr(training_args, "triangle_thickness_lr", 0.0005),
            max_steps=getattr(training_args, "triangle_ctrl_lr_max_steps", getattr(training_args, "iterations", 30000)),
        )
        self.triangle_color_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "triangle_color_lr", 0.001),
            lr_final=getattr(training_args, "triangle_color_lr", 0.001),
            max_steps=getattr(training_args, "triangle_ctrl_lr_max_steps", getattr(training_args, "iterations", 30000)),
        )
        self.triangle_semantic_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "triangle_semantic_lr", 0.0005),
            lr_final=getattr(training_args, "triangle_semantic_lr", 0.0005),
            max_steps=getattr(training_args, "triangle_ctrl_lr_max_steps", getattr(training_args, "iterations", 30000)),
        )

    def add_triangle_optimizer_groups(self, training_args):
        if self.optimizer is None or self._triangle_ctrl.shape[0] == 0:
            return

        existing_names = {group.get("name") for group in self.optimizer.param_groups}
        self._configure_triangle_scheduler_args(training_args)
        triangle_groups = [
            ("triangle_ctrl", [self._triangle_ctrl], getattr(training_args, "triangle_ctrl_lr_init", 0.0005)),
            ("triangle_alpha", [self._triangle_alpha_logit], getattr(training_args, "triangle_alpha_lr", 0.0005)),
            ("triangle_thickness", [self._triangle_thickness_logit], getattr(training_args, "triangle_thickness_lr", 0.0005)),
            ("triangle_rgb", [self._triangle_rgb_residual], getattr(training_args, "triangle_color_lr", 0.001)),
            ("triangle_semantic", [self._triangle_semantic_feat], getattr(training_args, "triangle_semantic_lr", 0.0005)),
        ]
        for name, params, lr in triangle_groups:
            if name in existing_names:
                continue
            self.optimizer.add_param_group({"params": params, "lr": lr, "name": name})

    def get_module_state(self):
        module_state = {
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "mlp_cov": self.mlp_cov.state_dict(),
            "color_mlp": self.mlp_color_fallback.state_dict(),
        }
        if self.use_feat_bank:
            module_state["feature_bank_mlp"] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0:
            module_state["appearance"] = self.embedding_appearance.state_dict()
        return module_state

    def load_module_state(self, module_state):
        if module_state is None:
            return

        if "mlp_opacity" in module_state:
            self.mlp_opacity.load_state_dict(module_state["mlp_opacity"])
        if "mlp_cov" in module_state:
            self.mlp_cov.load_state_dict(module_state["mlp_cov"])

        if "color_mlp" in module_state:
            self.warm_start_color_modules(module_state["color_mlp"])
        elif "mlp_color_fallback" in module_state:
            self.warm_start_color_modules(module_state["mlp_color_fallback"])

        if self.use_feat_bank and "feature_bank_mlp" in module_state:
            self.mlp_feature_bank.load_state_dict(module_state["feature_bank_mlp"])
        if self.appearance_dim > 0 and "appearance" in module_state:
            self.embedding_appearance.load_state_dict(module_state["appearance"])

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color_fallback.eval()
        if self.semantic_adapter is not None:
            self.semantic_adapter.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color_fallback.train()
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
            self.get_semantic_routing_state(),
            self.get_module_state(),
            self._anchor_sem_feat,
            self._semantic_confidence,
            self.get_triangle_state(),
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
                return

            merged_state = copy.deepcopy(opt_dict)
            merged_state["param_groups"] = copy.deepcopy(saved_groups)
            merged_state["param_groups"].extend(copy.deepcopy(current_groups[len(saved_groups):]))
            self.optimizer.load_state_dict(merged_state)
    
    def restore(self, model_args, training_args):
        semantic_adapter_state = None
        routing_state = None
        module_state = None
        triangle_state = None
        previous_semantic = self._semantic_features if self._semantic_features.dim() == 2 else None
        previous_semantic_confidence = self._semantic_confidence if self._semantic_confidence.dim() == 1 else None
        previous_anchor_sem_feat = self._anchor_sem_feat if self._anchor_sem_feat.dim() == 2 else None
        anchor_sem_feat = None
        semantic_confidence = None

        # [关键修复] 兼容检查点加载，防止尝试加载之前没有保存语义的检查点时发生 unpack 报错
        if len(model_args) >= 12:
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
            semantic_adapter_state) = model_args[:12]
            if len(model_args) >= 13:
                routing_state = model_args[12]
            if len(model_args) >= 14:
                module_state = model_args[13]
            if len(model_args) >= 15:
                anchor_sem_feat = model_args[14]
            if len(model_args) >= 16:
                semantic_confidence = model_args[15]
            if len(model_args) >= 17:
                triangle_state = model_args[16]
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

        if anchor_sem_feat is not None and anchor_sem_feat.dim() == 2 and anchor_sem_feat.shape[0] == self._anchor.shape[0]:
            self._anchor_sem_feat = nn.Parameter(anchor_sem_feat.detach().to(device="cuda", dtype=torch.float32).requires_grad_(True))
        else:
            self._anchor_sem_feat = self._build_aligned_anchor_sem_feat(self._anchor.shape[0], previous_anchor_sem_feat)

        if semantic_confidence is not None and semantic_confidence.dim() == 1 and semantic_confidence.shape[0] == self._anchor.shape[0]:
            self._semantic_confidence = semantic_confidence.to(device="cuda", dtype=torch.float32)
        else:
            self._semantic_confidence = self._build_aligned_semantic_confidence(self._anchor.shape[0], previous_semantic_confidence)

        if semantic_adapter_state is not None:
            self.init_semantic_adapter(semantic_adapter_state["weight"].shape[1])
            self.semantic_adapter.load_state_dict(semantic_adapter_state)
        elif self._semantic_features.dim() == 2 and self._semantic_features.shape[1] > 0:
            self.init_semantic_adapter(self._semantic_features.shape[1])

        self.load_semantic_routing_state(routing_state)
        self.load_module_state(module_state)
        self.load_triangle_state(triangle_state)
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
        return self.mlp_color_fallback
    
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
        self._anchor_sem_feat = nn.Parameter(torch.zeros_like(anchors_feat).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self._semantic_confidence = torch.zeros((self.get_anchor.shape[0],), device="cuda", dtype=torch.float32)
        self.load_semantic_routing_state(None)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.set_triangle_hyperparams(training_args)
        if self._anchor_sem_feat.dim() != 2 or self._anchor_sem_feat.shape[0] != self.get_anchor.shape[0]:
            self._anchor_sem_feat = self._build_aligned_anchor_sem_feat(self.get_anchor.shape[0], self._anchor_sem_feat)
        if self._semantic_confidence.dim() != 1 or self._semantic_confidence.shape[0] != self.get_anchor.shape[0]:
            self._semantic_confidence = self._build_aligned_semantic_confidence(self.get_anchor.shape[0], self._semantic_confidence)
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
            {'params': self.mlp_color_fallback.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color_fallback"},
        ]

        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})
        
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        if self.semantic_adapter is not None:
            l.append({'params': self.semantic_adapter.parameters(), 'lr': training_args.semantic_adapter_lr, "name": "semantic_adapter"})
        l.append({'params': [self._anchor_sem_feat], 'lr': training_args.feature_lr, "name": "anchor_sem_feat"})
        if self._triangle_ctrl.shape[0] > 0:
            self._configure_triangle_scheduler_args(training_args)
            l.extend([
                {'params': [self._triangle_ctrl], 'lr': training_args.triangle_ctrl_lr_init, "name": "triangle_ctrl"},
                {'params': [self._triangle_alpha_logit], 'lr': training_args.triangle_alpha_lr, "name": "triangle_alpha"},
                {'params': [self._triangle_thickness_logit], 'lr': training_args.triangle_thickness_lr, "name": "triangle_thickness"},
                {'params': [self._triangle_rgb_residual], 'lr': training_args.triangle_color_lr, "name": "triangle_rgb"},
                {'params': [self._triangle_semantic_feat], 'lr': training_args.triangle_semantic_lr, "name": "triangle_semantic"},
            ])

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
        if self._triangle_ctrl.shape[0] > 0 and self.triangle_ctrl_scheduler_args is None:
            self._configure_triangle_scheduler_args(training_args)

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
            if param_group["name"] == "mlp_color_fallback":
                param_group['lr'] = self.mlp_color_scheduler_args(iteration)
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                param_group['lr'] = self.mlp_featurebank_scheduler_args(iteration)
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                param_group['lr'] = self.appearance_scheduler_args(iteration)
            if self.semantic_adapter is not None and param_group["name"] == "semantic_adapter":
                param_group['lr'] = self.semantic_adapter_scheduler_args(iteration)
            if self._triangle_ctrl.shape[0] > 0 and param_group["name"] == "triangle_ctrl":
                param_group['lr'] = self.triangle_ctrl_scheduler_args(iteration)
            if self._triangle_ctrl.shape[0] > 0 and param_group["name"] == "triangle_alpha":
                param_group['lr'] = self.triangle_alpha_scheduler_args(iteration)
            if self._triangle_ctrl.shape[0] > 0 and param_group["name"] == "triangle_thickness":
                param_group['lr'] = self.triangle_thickness_scheduler_args(iteration)
            if self._triangle_ctrl.shape[0] > 0 and param_group["name"] == "triangle_rgb":
                param_group['lr'] = self.triangle_color_scheduler_args(iteration)
            if self._triangle_ctrl.shape[0] > 0 and param_group["name"] == "triangle_semantic":
                param_group['lr'] = self.triangle_semantic_scheduler_args(iteration)

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
        self._anchor_sem_feat = nn.Parameter(torch.zeros((anchor.shape[0], self.feat_dim), dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self._semantic_confidence = torch.zeros((self.get_anchor.shape[0],), device="cuda", dtype=torch.float32)
        self.load_semantic_routing_state(None)

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

        if "semantic_cluster_ids" in tensors_dict:
            if self._semantic_cluster_ids.dim() != 1 or self._semantic_cluster_ids.shape[0] == 0:
                self._semantic_cluster_ids = tensors_dict["semantic_cluster_ids"]
            else:
                self._semantic_cluster_ids = torch.cat((self._semantic_cluster_ids, tensors_dict["semantic_cluster_ids"]), dim=0)

        if "semantic_valid_mask" in tensors_dict:
            if self._semantic_valid_mask.dim() != 1 or self._semantic_valid_mask.shape[0] == 0:
                self._semantic_valid_mask = tensors_dict["semantic_valid_mask"]
            else:
                self._semantic_valid_mask = torch.cat((self._semantic_valid_mask, tensors_dict["semantic_valid_mask"]), dim=0)

        if "semantic_confidence" in tensors_dict:
            if self._semantic_confidence.dim() != 1 or self._semantic_confidence.shape[0] == 0:
                self._semantic_confidence = tensors_dict["semantic_confidence"]
            else:
                self._semantic_confidence = torch.cat((self._semantic_confidence, tensors_dict["semantic_confidence"]), dim=0)

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
        if self._semantic_cluster_ids.shape[0] > 0:
            self._semantic_cluster_ids = self._semantic_cluster_ids[mask]
        if self._semantic_valid_mask.shape[0] > 0:
            self._semantic_valid_mask = self._semantic_valid_mask[mask]
        if self._semantic_confidence.shape[0] > 0:
            self._semantic_confidence = self._semantic_confidence[mask]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._anchor_sem_feat = optimizable_tensors["anchor_sem_feat"]
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
                new_anchor_sem_feat = torch.zeros((candidate_anchor.shape[0], self.feat_dim), device="cuda")
                new_offsets = torch.zeros((candidate_anchor.shape[0], self.n_offsets, 3), device="cuda").float()

                # [新增] 为新锚点计算语义特征（继承父代）
                # =================== [FIX START: 维度自适应与安全性优化] ===================
                # 获取当前特征维度 (默认回退 128 而不是 512，与修改后的维度一致)
                feat_dim = self._semantic_features.shape[1] if (self._semantic_features.dim() > 1 and self._semantic_features.shape[0] > 0) else 128
                
                new_semantic = torch.zeros((candidate_anchor.shape[0], feat_dim), device='cuda')
                new_semantic_cluster_ids = torch.full((candidate_anchor.shape[0],), -1, device='cuda', dtype=torch.long)
                new_semantic_valid_mask = torch.zeros((candidate_anchor.shape[0],), device='cuda', dtype=torch.bool)
                new_semantic_confidence = torch.zeros((candidate_anchor.shape[0],), device='cuda', dtype=torch.float32)
                candidate_indices = torch.nonzero(candidate_mask).view(-1)
                parent_anchor_indices = candidate_indices // self.n_offsets
                
                if self._semantic_features.shape[0] > 0:
                    candidate_indices = torch.nonzero(candidate_mask).view(-1) # 获取所有需要生长的索引
                    parent_anchor_indices = candidate_indices // self.n_offsets # 计算属于哪个父锚点
                    parent_semantic = self._semantic_features[parent_anchor_indices] # 直接索引提取
                    
                    # [关键修复] 使用 feat_dim 替代硬编码维度
                    new_semantic = scatter_max(parent_semantic, inverse_indices.unsqueeze(1).expand(-1, feat_dim), dim=0)[0][remove_duplicates]

                if self._semantic_cluster_ids.shape[0] > 0:
                    parent_cluster_ids = self._semantic_cluster_ids[parent_anchor_indices].float().unsqueeze(1)
                    reduced_cluster_ids = scatter_max(parent_cluster_ids, inverse_indices.unsqueeze(1), dim=0)[0][remove_duplicates]
                    new_semantic_cluster_ids = reduced_cluster_ids.squeeze(1).long()

                if self._semantic_valid_mask.shape[0] > 0:
                    parent_valid_mask = self._semantic_valid_mask[parent_anchor_indices].float().unsqueeze(1)
                    reduced_valid_mask = scatter_max(parent_valid_mask, inverse_indices.unsqueeze(1), dim=0)[0][remove_duplicates]
                    new_semantic_valid_mask = reduced_valid_mask.squeeze(1) > 0
                    new_semantic_cluster_ids[~new_semantic_valid_mask] = -1

                if self._semantic_confidence.shape[0] > 0:
                    parent_confidence = self._semantic_confidence[parent_anchor_indices].unsqueeze(1)
                    reduced_confidence = scatter_max(parent_confidence, inverse_indices.unsqueeze(1), dim=0)[0][remove_duplicates]
                    new_semantic_confidence = reduced_confidence.squeeze(1)
                # =================== [FIX END] ===================

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_sem_feat": new_anchor_sem_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "semantic_features": new_semantic,
                    "semantic_cluster_ids": new_semantic_cluster_ids,
                    "semantic_valid_mask": new_semantic_valid_mask,
                    "semantic_confidence": new_semantic_confidence,
                }
                
                self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda')], dim=0)
                self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda')], dim=0)
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor, self._scaling, self._rotation = optimizable_tensors["anchor"], optimizable_tensors["scaling"], optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_sem_feat = optimizable_tensors["anchor_sem_feat"]
                self._offset, self._opacity = optimizable_tensors["offset"], optimizable_tensors["opacity"]

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
                                       ('color_mlp', self.mlp_color_fallback, self.feat_dim+3+self.color_dist_dim+self.appearance_dim)]:
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
            torch.save(self.get_triangle_state(), os.path.join(path, 'triangle_state.pth'))
        elif mode == 'unite':
            data = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color_fallback.state_dict(),
                'triangle_state': self.get_triangle_state(),
            }
            if self.use_feat_bank: data['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0: data['appearance'] = self.embedding_appearance.state_dict()
            torch.save(data, os.path.join(path, 'checkpoints.pth'))

    def load_mlp_checkpoints(self, path, mode = 'split'):
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            shared_color_path = os.path.join(path, 'color_mlp.pt')
            fallback_path = os.path.join(path, 'color_mlp_fallback.pt')
            if os.path.exists(shared_color_path):
                self.warm_start_color_modules(torch.jit.load(shared_color_path).cuda().state_dict())
            elif os.path.exists(fallback_path):
                self.warm_start_color_modules(torch.jit.load(fallback_path).cuda().state_dict())
            else:
                raise FileNotFoundError(f"Neither {shared_color_path} nor {fallback_path} exists.")

            self.load_semantic_routing_state(None)
            triangle_state_path = os.path.join(path, 'triangle_state.pth')
            if os.path.exists(triangle_state_path):
                self.load_triangle_state(torch.load(triangle_state_path, map_location='cuda'))
            else:
                self.reset_triangle_state(self.triangle_semantic_dim)
            if self.use_feat_bank: self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0: self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            ckpt = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.load_module_state(ckpt)
            self.load_semantic_routing_state(None)
            self.load_triangle_state(ckpt.get('triangle_state'))
            if self.use_feat_bank: self.mlp_feature_bank.load_state_dict(ckpt['feature_bank_mlp'])
            if self.appearance_dim > 0: self.embedding_appearance.load_state_dict(ckpt['appearance'])
