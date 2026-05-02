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


import os
import numpy as np
import subprocess
import math

# [保留] 自动选择显存占用最低的 GPU
#cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
#result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
#os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
import torch.nn.functional as F
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene
from scene.gaussian_model_seg import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from semantic_init_main import SemanticVoter

# [新增] 导入用于降维可视化点云的库
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

# [新增] 导入 matplotlib 用于生成曲线图
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False
    print("Warning: matplotlib not found. Please `pip install matplotlib` to generate PSNR curves.")

# [修改 1] 注释掉全局初始化，防止提前锁定 GPU 0
# lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

# [修复] 改进的备份逻辑，显式排除大文件夹防止递归错误
def saveRuntimeCode(dst: str) -> None:
    # 必须排除 'output' 文件夹以防止递归死循环
    # 同时排除 'data' 和 'weights' 以免备份过于臃肿
    additionalIgnorePatterns = ['.git', '.gitignore', 'output', 'data', 'weights', '__pycache__']
    ignorePatterns = set()
    ROOT = '.'
    if os.path.exists(os.path.join(ROOT, '.gitignore')):
        with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
            for line in gitIgnoreFile:
                if not line.startswith('#'):
                    clean_line = line.strip()
                    if clean_line:
                        if clean_line.endswith('/'): clean_line = clean_line[:-1]
                        ignorePatterns.add(clean_line)
    
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern) if isinstance(ignorePatterns, list) else ignorePatterns.add(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()
    
    # 如果目标文件夹已存在，先清理
    if os.path.exists(dst):
        shutil.rmtree(dst)
        
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*list(ignorePatterns)))
    print(f'Backup Finished! Source code saved to {dst}')

# [新增] 将语义特征降维并保存为可直接查看的彩色点云 (.ply)
def export_semantic_point_cloud(anchors, semantic_features, output_path, logger=None):
    """
    将 128 维的语义特征使用 PCA 降维至 3 维，并映射到 RGB 颜色空间，
    最后导出为 .ply 点云文件，方便在 MeshLab 中查看。
    """
    try:
        anchors_np = anchors.detach().cpu().numpy()
        features_np = semantic_features.detach().cpu().numpy()
        
        if features_np.shape[1] < 3:
            if logger: logger.warning("[WARN] 语义特征维度小于3，跳过PCA可视化。")
            return
            
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features_np)
        
        # 使用 2% 和 98% 分位数截断，去除极端的离群点，使得映射出的颜色对比度更自然
        p2, p98 = np.percentile(pca_features, [2, 98], axis=0)
        pca_features = np.clip(pca_features, p2, p98)
        
        pca_min = pca_features.min(axis=0)
        pca_max = pca_features.max(axis=0)
        
        # 归一化到 0~1 范围，然后放大到 0~255 RGB 值
        pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)
        colors = (pca_normalized * 255).astype(np.uint8)
        
        # 构建适合点云查看软件的 PLY 数据结构
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                 
        elements = np.empty(anchors_np.shape[0], dtype=dtype)
        attributes = np.concatenate((anchors_np, colors), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
        if logger: logger.info(f"[INFO] 成功导出 PCA 降维语义彩色点云至: {output_path}")
    except Exception as e:
        if logger: logger.error(f"[ERROR] 导出语义点云失败: {e}")


def get_semantic_supervision_start_iter(opt):
    return max(opt.semantic_loss_start_iter, opt.update_until + 1)


def get_semantic_loss_scale(iteration, opt):
    start_iter = get_semantic_supervision_start_iter(opt)
    if iteration < start_iter:
        return 0.0
    if opt.semantic_loss_ramp_iters <= 0:
        return 1.0
    progress = (iteration - start_iter + 1) / float(opt.semantic_loss_ramp_iters)
    return min(max(progress, 0.0), 1.0)


def compute_semantic_loss(gaussians, visible_mask, opt):
    semantic_features = gaussians.semantic_features
    semantic_adapter = gaussians.semantic_adapter
    anchor_sem_feat = gaussians.anchor_sem_feat

    if semantic_adapter is None or semantic_features.dim() != 2 or semantic_features.shape[0] == 0:
        return None
    if semantic_features.shape[0] != gaussians.get_anchor.shape[0]:
        return None
    if anchor_sem_feat.dim() != 2 or anchor_sem_feat.shape[0] != gaussians.get_anchor.shape[0]:
        return None
    if hasattr(semantic_adapter, "in_features") and semantic_adapter.in_features != semantic_features.shape[1]:
        return None

    if gaussians.semantic_valid_mask.dim() == 1 and gaussians.semantic_valid_mask.shape[0] == gaussians.get_anchor.shape[0]:
        semantic_valid_mask = gaussians.semantic_valid_mask
    else:
        semantic_valid_mask = semantic_features.abs().sum(dim=1) > 0

    if gaussians.semantic_confidence.dim() == 1 and gaussians.semantic_confidence.shape[0] == gaussians.get_anchor.shape[0]:
        semantic_confidence = gaussians.semantic_confidence
    else:
        semantic_confidence = semantic_valid_mask.float()

    confidence_threshold = getattr(opt, "semantic_confidence_threshold", 0.6)
    high_conf_mask = semantic_confidence >= confidence_threshold
    candidate_mask = visible_mask & semantic_valid_mask & high_conf_mask
    valid_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)

    if valid_indices.numel() < opt.semantic_min_count:
        return None

    if valid_indices.numel() > opt.semantic_sample_size:
        sampled = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:opt.semantic_sample_size]
        valid_indices = valid_indices[sampled]

    anchor_feat = F.normalize(anchor_sem_feat[valid_indices], p=2, dim=-1)
    semantic_targets = semantic_adapter(semantic_features[valid_indices].detach())
    semantic_targets = F.normalize(semantic_targets, p=2, dim=-1)

    return (1.0 - F.cosine_similarity(anchor_feat, semantic_targets, dim=-1)).mean()


def get_boundary_supervision_start_iter(opt):
    ratio_start_iter = int(float(getattr(opt, "boundary_loss_start_ratio", 0.6)) * float(opt.iterations))
    return max(opt.update_until + 1, ratio_start_iter)


def get_boundary_loss_scale(iteration, opt):
    start_iter = get_boundary_supervision_start_iter(opt)
    if iteration < start_iter:
        return 0.0
    if getattr(opt, "boundary_loss_ramp_iters", 0) <= 0:
        return 1.0
    progress = (iteration - start_iter + 1) / float(opt.boundary_loss_ramp_iters)
    return min(max(progress, 0.0), 1.0)


def has_stable_edge_masks(boundary_mask_dir):
    if not os.path.isdir(boundary_mask_dir):
        return False
    return any(Path(boundary_mask_dir).rglob("*_edge.png"))


def load_stable_edge_mask(boundary_mask_dir, image_name, cache, target_hw, device, dtype):
    if image_name not in cache:
        direct_path = os.path.join(boundary_mask_dir, f"{image_name}_edge.png")
        edge_path = direct_path if os.path.exists(direct_path) else None
        if edge_path is None:
            matches = list(Path(boundary_mask_dir).rglob(f"{image_name}_edge.png"))
            edge_path = str(matches[0]) if matches else None

        if edge_path is None:
            cache[image_name] = None
        else:
            edge_image = Image.open(edge_path).convert("L")
            edge_array = np.array(edge_image, dtype=np.float32) / 255.0
            cache[image_name] = torch.from_numpy(edge_array)

    edge_mask = cache[image_name]
    if edge_mask is None:
        return None

    edge_mask = edge_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    if edge_mask.shape[-2:] != target_hw:
        edge_mask = F.interpolate(edge_mask, size=target_hw, mode="nearest")
    return (edge_mask > 0.5).to(dtype=dtype)


def rgb_to_grayscale(image):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[1] == 1:
        return image
    return (
        0.2989 * image[:, 0:1] +
        0.5870 * image[:, 1:2] +
        0.1140 * image[:, 2:3]
    )


def sobel_gradient_magnitude(gray_image):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray_image.device,
        dtype=gray_image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray_image.device,
        dtype=gray_image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(gray_image, kernel_x, padding=1)
    grad_y = F.conv2d(gray_image, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def build_center_axis_weights(length, inner_margin, outer_margin, device, dtype):
    inner_margin = min(max(float(inner_margin), 0.0), 0.49)
    outer_margin = min(max(float(outer_margin), 0.0), 0.49)
    if outer_margin > inner_margin:
        outer_margin, inner_margin = inner_margin, outer_margin
    if abs(inner_margin - outer_margin) < 1e-6:
        outer_margin = max(0.0, inner_margin - 1e-3)

    coords = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
    weights = torch.zeros_like(coords)
    inner_left = inner_margin
    inner_right = 1.0 - inner_margin
    outer_left = outer_margin
    outer_right = 1.0 - outer_margin

    inner_mask = (coords >= inner_left) & (coords <= inner_right)
    weights[inner_mask] = 1.0

    if inner_left > outer_left:
        left_mask = (coords >= outer_left) & (coords < inner_left)
        weights[left_mask] = (coords[left_mask] - outer_left) / max(inner_left - outer_left, 1e-6)

        right_mask = (coords > inner_right) & (coords <= outer_right)
        weights[right_mask] = (outer_right - coords[right_mask]) / max(outer_right - inner_right, 1e-6)

    return weights.clamp_(0.0, 1.0)


def get_center_weight_map(height, width, opt, device, dtype, cache):
    cache_key = (height, width, str(device), str(dtype))
    if cache_key not in cache:
        x_weights = build_center_axis_weights(
            width,
            getattr(opt, "boundary_center_inner_ratio", 0.25),
            getattr(opt, "boundary_center_outer_ratio", 0.15),
            device,
            dtype,
        )
        y_weights = build_center_axis_weights(
            height,
            getattr(opt, "boundary_center_inner_ratio", 0.25),
            getattr(opt, "boundary_center_outer_ratio", 0.15),
            device,
            dtype,
        )
        cache[cache_key] = (y_weights[:, None] * x_weights[None, :]).unsqueeze(0).unsqueeze(0)
    return cache[cache_key]


def compute_boundary_sharpen_loss(rendered_image, gt_image, image_name, boundary_mask_dir, opt, mask_cache, center_weight_cache):
    height, width = rendered_image.shape[-2:]
    edge_mask = load_stable_edge_mask(
        boundary_mask_dir,
        image_name,
        mask_cache,
        (height, width),
        rendered_image.device,
        rendered_image.dtype,
    )
    if edge_mask is None:
        return None, 0

    render_gray = rgb_to_grayscale(rendered_image)
    gt_gray = rgb_to_grayscale(gt_image)
    render_grad = sobel_gradient_magnitude(render_gray)
    gt_grad = sobel_gradient_magnitude(gt_gray)
    center_weight = get_center_weight_map(height, width, opt, rendered_image.device, rendered_image.dtype, center_weight_cache)
    gt_edge_mask = (gt_grad > getattr(opt, "boundary_gt_grad_thresh", 0.08)).to(dtype=rendered_image.dtype)
    final_weight = edge_mask * center_weight * gt_edge_mask
    active_pixels = int((final_weight > 0).sum().item())

    if active_pixels < int(getattr(opt, "boundary_min_pixels", 256)):
        return None, active_pixels

    grad_diff = torch.sqrt((render_grad - gt_grad).pow(2) + 1e-6)
    loss = (grad_diff * final_weight).sum() / final_weight.sum().clamp_min(1e-6)
    return loss, active_pixels


def get_triangle_supervision_start_iter(opt):
    explicit_start = int(getattr(opt, "triangle_init_start_iter", -1))
    default_start = max(
        opt.update_until + 1,
        get_semantic_supervision_start_iter(opt),
        get_boundary_supervision_start_iter(opt),
    )
    return explicit_start if explicit_start >= 0 else default_start


def get_triangle_joint_start_iter(opt):
    return get_triangle_supervision_start_iter(opt) + max(0, int(getattr(opt, "triangle_only_iters", 1000)))


def get_triangle_loss_scale(iteration, opt):
    start_iter = get_triangle_supervision_start_iter(opt)
    if iteration < start_iter:
        return 0.0
    ramp_iters = int(getattr(opt, "triangle_ramp_iters", 1000))
    if ramp_iters <= 0:
        return 1.0
    progress = (iteration - start_iter + 1) / float(ramp_iters)
    return min(max(progress, 0.0), 1.0)


def sanitize_boundary_candidates(boundary_candidates, anchor_count, device="cuda"):
    if not boundary_candidates:
        return None

    parent_anchor_indices = boundary_candidates.get("parent_anchor_indices")
    if parent_anchor_indices is None:
        return None

    parent_anchor_indices = parent_anchor_indices.to(device=device, dtype=torch.long)
    if parent_anchor_indices.numel() == 0:
        semantic_features = boundary_candidates.get("semantic_features")
        semantic_dim = semantic_features.shape[1] if semantic_features is not None and semantic_features.dim() == 2 else 128
        return {
            "parent_anchor_indices": torch.empty((0,), device=device, dtype=torch.long),
            "positions": torch.empty((0, 3), device=device, dtype=torch.float32),
            "semantic_features": torch.empty((0, semantic_dim), device=device, dtype=torch.float32),
            "confidence": torch.empty((0,), device=device, dtype=torch.float32),
            "view_counts": torch.empty((0,), device=device, dtype=torch.float32),
            "angle_stability": torch.empty((0,), device=device, dtype=torch.float32),
            "depth_stability": torch.empty((0,), device=device, dtype=torch.float32),
            "center_weight": torch.empty((0,), device=device, dtype=torch.float32),
        }

    valid_mask = (parent_anchor_indices >= 0) & (parent_anchor_indices < anchor_count)
    if not valid_mask.any():
        return None

    sanitized = {"parent_anchor_indices": parent_anchor_indices[valid_mask]}
    for key in ["positions", "semantic_features", "confidence", "view_counts", "angle_stability", "depth_stability", "center_weight"]:
        value = boundary_candidates.get(key)
        if value is None:
            continue
        value = value.to(device=device)
        if value.shape[0] == valid_mask.shape[0]:
            sanitized[key] = value[valid_mask]

    if "positions" not in sanitized:
        sanitized["positions"] = torch.empty((sanitized["parent_anchor_indices"].shape[0], 3), device=device, dtype=torch.float32)
    if "semantic_features" not in sanitized:
        sanitized["semantic_features"] = torch.empty((sanitized["parent_anchor_indices"].shape[0], 128), device=device, dtype=torch.float32)
    for key in ["confidence", "view_counts", "angle_stability", "depth_stability", "center_weight"]:
        if key not in sanitized:
            sanitized[key] = torch.zeros((sanitized["parent_anchor_indices"].shape[0],), device=device, dtype=torch.float32)

    sanitized["positions"] = sanitized["positions"].to(dtype=torch.float32)
    sanitized["semantic_features"] = sanitized["semantic_features"].to(dtype=torch.float32)
    for key in ["confidence", "view_counts", "angle_stability", "depth_stability", "center_weight"]:
        sanitized[key] = sanitized[key].to(dtype=torch.float32)
    return sanitized


def project_points_to_image(viewpoint_camera, points):
    if points.shape[0] == 0:
        return torch.empty((0, 2), device=points.device, dtype=points.dtype), torch.empty((0,), device=points.device, dtype=points.dtype)

    R = torch.tensor(viewpoint_camera.R, device=points.device, dtype=points.dtype)
    T = torch.tensor(viewpoint_camera.T, device=points.device, dtype=points.dtype)
    pts_cam = (points @ R.t()) + T
    depth = pts_cam[:, 2]

    fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx * 0.5)
    fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy * 0.5)
    cx = 0.5 * viewpoint_camera.image_width
    cy = 0.5 * viewpoint_camera.image_height

    uv = torch.empty((points.shape[0], 2), device=points.device, dtype=points.dtype)
    uv[:, 0] = fx * (pts_cam[:, 0] / depth.clamp_min(1e-6)) + cx
    uv[:, 1] = fy * (pts_cam[:, 1] / depth.clamp_min(1e-6)) + cy
    return uv, depth


def point_to_segment_distance(grid, a, b):
    segment = b - a
    segment_norm = (segment * segment).sum().clamp_min(1e-6)
    projection = ((grid - a) * segment).sum(dim=-1) / segment_norm
    projection = projection.clamp(0.0, 1.0)
    closest = a + projection.unsqueeze(-1) * segment
    return torch.norm(grid - closest, dim=-1)


def render_boundary_triangles(viewpoint_camera, gaussians, visible_mask, opt):
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    device = gaussians.get_anchor.device
    dtype = gaussians.get_anchor.dtype

    zero_alpha = torch.zeros((1, height, width), device=device, dtype=dtype)
    zero_rgb = torch.zeros((3, height, width), device=device, dtype=dtype)
    empty_result = {
        "triangle_alpha": zero_alpha,
        "triangle_rgb_residual": zero_rgb,
        "triangle_weight_sum": zero_alpha.clone(),
        "triangle_support_map": zero_alpha.clone(),
        "triangle_count": 0,
        "triangle_visible_count": 0,
    }

    triangle_state = gaussians.get_active_triangle_state(visible_mask=visible_mask)
    if triangle_state is None:
        return empty_result

    confidence = triangle_state["confidence"]
    render_max = int(getattr(opt, "triangle_render_max_per_view", 256))
    if render_max > 0 and confidence.numel() > render_max:
        topk = torch.topk(confidence, k=render_max, largest=True).indices
        for key, value in list(triangle_state.items()):
            if torch.is_tensor(value) and value.shape[0] == confidence.shape[0]:
                triangle_state[key] = value[topk]
        confidence = triangle_state["confidence"]

    ctrl = triangle_state["ctrl"]
    flat_uv, flat_depth = project_points_to_image(viewpoint_camera, ctrl.reshape(-1, 3))
    uv = flat_uv.view(ctrl.shape[0], 3, 2)
    depth = flat_depth.view(ctrl.shape[0], 3)
    valid_depth_mask = (depth > 0.05).all(dim=1)
    if not valid_depth_mask.any():
        return empty_result

    for key, value in list(triangle_state.items()):
        if torch.is_tensor(value) and value.shape[0] == valid_depth_mask.shape[0]:
            triangle_state[key] = value[valid_depth_mask]
    ctrl = triangle_state["ctrl"]
    uv = uv[valid_depth_mask]
    depth = depth[valid_depth_mask]

    alpha_accum = torch.zeros((1, height, width), device=device, dtype=dtype)
    support_accum = torch.zeros((1, height, width), device=device, dtype=dtype)
    weight_accum = torch.zeros((1, height, width), device=device, dtype=dtype)
    rgb_accum = torch.zeros((3, height, width), device=device, dtype=dtype)
    sharpness = float(getattr(opt, "triangle_render_sharpness", 6.0))
    depth_temperature = float(getattr(opt, "triangle_depth_temperature", 0.25))
    support_band_scale = float(getattr(opt, "triangle_support_band_scale", 2.5))
    support_band_bias = float(getattr(opt, "triangle_support_band_bias", 2.0))
    support_sharpness_scale = float(getattr(opt, "triangle_support_sharpness_scale", 0.5))

    visible_count = 0
    for tri_idx in range(ctrl.shape[0]):
        verts = uv[tri_idx]
        tri_depth = depth[tri_idx].mean()
        thickness = triangle_state["thickness"][tri_idx, 0]
        alpha = triangle_state["alpha"][tri_idx, 0]
        rgb_residual = triangle_state["rgb_residual"][tri_idx]
        confidence_weight = triangle_state["confidence"][tri_idx].clamp(0.0, 1.0)
        center_quality = triangle_state["center_weight"][tri_idx].clamp(0.0, 1.0)
        support_thickness = thickness * support_band_scale + support_band_bias
        support_pad = max(float(thickness.item()), float(support_thickness.item())) + 2.0
        support_sharpness = max(1.0, sharpness * support_sharpness_scale)

        x0 = max(int(torch.floor(verts[:, 0].min() - support_pad).item()), 0)
        x1 = min(int(torch.ceil(verts[:, 0].max() + support_pad).item()), width - 1)
        y0 = max(int(torch.floor(verts[:, 1].min() - support_pad).item()), 0)
        y1 = min(int(torch.ceil(verts[:, 1].max() + support_pad).item()), height - 1)
        if x1 <= x0 or y1 <= y0:
            continue

        ys = torch.arange(y0, y1 + 1, device=device, dtype=dtype)
        xs = torch.arange(x0, x1 + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)

        dist01 = point_to_segment_distance(grid, verts[0], verts[1])
        dist12 = point_to_segment_distance(grid, verts[1], verts[2])
        dist20 = point_to_segment_distance(grid, verts[2], verts[0])
        edge_dist = torch.minimum(dist01, torch.minimum(dist12, dist20))

        support_strength = confidence_weight * (0.5 + 0.5 * center_quality)
        support_band = torch.sigmoid((support_thickness - edge_dist) * support_sharpness) * support_strength
        if float(support_band.max().item()) <= 1e-6:
            continue

        depth_weight = torch.exp(-tri_depth * depth_temperature)
        support_accum[:, y0:y1 + 1, x0:x1 + 1] += (support_band * depth_weight).unsqueeze(0)

        band_alpha = torch.sigmoid((thickness - edge_dist) * sharpness) * alpha
        if float(band_alpha.max().item()) <= 1e-6:
            continue

        contribution = band_alpha * depth_weight

        alpha_accum[:, y0:y1 + 1, x0:x1 + 1] += contribution.unsqueeze(0)
        weight_accum[:, y0:y1 + 1, x0:x1 + 1] += contribution.unsqueeze(0)
        rgb_accum[:, y0:y1 + 1, x0:x1 + 1] += contribution.unsqueeze(0) * rgb_residual.view(3, 1, 1)
        visible_count += 1

    if visible_count == 0:
        return empty_result

    triangle_support_map = 1.0 - torch.exp(-support_accum)
    triangle_alpha = 1.0 - torch.exp(-alpha_accum)
    triangle_rgb_residual = rgb_accum / weight_accum.clamp_min(1e-6)
    return {
        "triangle_alpha": triangle_alpha,
        "triangle_rgb_residual": triangle_rgb_residual,
        "triangle_weight_sum": weight_accum,
        "triangle_support_map": triangle_support_map,
        "triangle_count": int(ctrl.shape[0]),
        "triangle_visible_count": int(visible_count),
    }


def hybrid_render(viewpoint_camera, gaussians, pipe, bg_color, opt, scaling_modifier=1.0, visible_mask=None, retain_grad=False):
    base_pkg = render(
        viewpoint_camera,
        gaussians,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        visible_mask=visible_mask,
        retain_grad=retain_grad,
    )
    triangle_pkg = render_boundary_triangles(viewpoint_camera, gaussians, visible_mask, opt)
    fused_render = torch.clamp(
        base_pkg["render"] + triangle_pkg["triangle_alpha"] * triangle_pkg["triangle_rgb_residual"],
        0.0,
        1.0,
    )
    render_pkg = dict(base_pkg)
    render_pkg["render_base"] = base_pkg["render"]
    render_pkg["render"] = fused_render
    render_pkg.update(triangle_pkg)
    return render_pkg


def compute_triangle_branch_losses(render_pkg, gt_image, image_name, boundary_mask_dir, opt, mask_cache, center_weight_cache, gaussians):
    stats = {
        "support_active_pixels": 0,
        "refined_core_pixels": 0,
        "supervision_active_pixels": 0,
    }
    if render_pkg.get("triangle_visible_count", 0) <= 0:
        return {}, stats

    triangle_alpha = render_pkg["triangle_alpha"]
    edge_mask = load_stable_edge_mask(
        boundary_mask_dir,
        image_name,
        mask_cache,
        triangle_alpha.shape[-2:],
        triangle_alpha.device,
        triangle_alpha.dtype,
    )
    if edge_mask is None:
        return {}, stats

    center_weight = get_center_weight_map(
        triangle_alpha.shape[-2],
        triangle_alpha.shape[-1],
        opt,
        triangle_alpha.device,
        triangle_alpha.dtype,
        center_weight_cache,
    )
    losses = {}
    gt_gray = rgb_to_grayscale(gt_image)
    gt_grad = sobel_gradient_magnitude(gt_gray)
    low_thresh = float(getattr(opt, "boundary_gt_grad_thresh", 0.08))
    high_thresh = max(
        low_thresh + 1e-3,
        float(getattr(opt, "triangle_gt_grad_high_thresh", max(low_thresh * 2.0, low_thresh + 0.08))),
    )
    gt_grad_strength = ((gt_grad - low_thresh) / max(high_thresh - low_thresh, 1e-6)).clamp(0.0, 1.0)
    gt_edge_mask = (gt_grad_strength > 0).to(dtype=triangle_alpha.dtype)
    triangle_support = render_pkg.get("triangle_support_map", None)
    if triangle_support is None or float(triangle_support.max().item()) <= 1e-6:
        triangle_support = torch.ones_like(edge_mask)
    else:
        triangle_support = triangle_support.to(device=triangle_alpha.device, dtype=triangle_alpha.dtype)

    edge_core = edge_mask.clamp(0.0, 1.0)
    support_min_weight = float(getattr(opt, "triangle_support_min_weight", 0.10))
    support_mask = (triangle_support >= support_min_weight).to(dtype=triangle_alpha.dtype)
    stats["support_active_pixels"] = int((support_mask > 0).sum().item())
    base_core = edge_core * gt_edge_mask * support_mask

    peak_kernel = max(1, int(getattr(opt, "triangle_gt_peak_kernel", 3)))
    if peak_kernel % 2 == 0:
        peak_kernel += 1
    if peak_kernel > 1:
        local_max = F.max_pool2d(gt_grad, kernel_size=peak_kernel, stride=1, padding=peak_kernel // 2)
        gt_peak_mask = (gt_grad >= (local_max - 1e-6)).to(dtype=triangle_alpha.dtype)
    else:
        gt_peak_mask = torch.ones_like(edge_core)

    refined_core = base_core * gt_peak_mask
    if int((refined_core > 0).sum().item()) < int(getattr(opt, "triangle_peak_min_pixels", 64)):
        refined_core = base_core
    stats["refined_core_pixels"] = int((refined_core > 0).sum().item())

    band_kernel = max(1, int(getattr(opt, "triangle_mask_band_kernel", 3)))
    if band_kernel % 2 == 0:
        band_kernel += 1
    if band_kernel > 1:
        edge_band = F.max_pool2d(base_core, kernel_size=band_kernel, stride=1, padding=band_kernel // 2)
    else:
        edge_band = base_core
    edge_ring = (edge_band - refined_core).clamp(0.0, 1.0)
    soft_target = torch.clamp(
        refined_core * (0.5 + 0.5 * gt_grad_strength) +
        float(getattr(opt, "triangle_mask_ring_weight", 0.35)) * edge_ring * gt_grad_strength,
        0.0,
        1.0,
    )
    supervision_weight = edge_band * center_weight * gt_grad_strength * triangle_support
    active_pixels = int((supervision_weight > support_min_weight).sum().item())
    stats["supervision_active_pixels"] = active_pixels

    if active_pixels >= int(getattr(opt, "triangle_mask_min_pixels", getattr(opt, "boundary_min_pixels", 256))):
        # Restrict supervision to edge regions that visible triangles can plausibly explain in this view.
        mask_residual = torch.abs(triangle_alpha - soft_target) * supervision_weight
        losses["mask"] = mask_residual.sum() / supervision_weight.sum().clamp_min(1e-6)

        rgb_weight = supervision_weight * torch.clamp(triangle_alpha + soft_target, 0.0, 1.0)
        rgb_residual = torch.abs(render_pkg["render"] - gt_image) * rgb_weight
        losses["edge_rgb"] = rgb_residual.sum() / (rgb_weight.sum().clamp_min(1e-6) * gt_image.shape[0])

    triangle_state = gaussians.get_active_triangle_state()
    if triangle_state is None:
        return losses, stats

    if triangle_state["semantic_feat"].shape[0] > 0 and triangle_state["semantic_target"].shape == triangle_state["semantic_feat"].shape:
        triangle_feat = F.normalize(triangle_state["semantic_feat"], p=2, dim=-1)
        triangle_target = F.normalize(triangle_state["semantic_target"].detach(), p=2, dim=-1)
        confidence = triangle_state["confidence"].clamp_min(1e-3)
        losses["semantic"] = ((1.0 - F.cosine_similarity(triangle_feat, triangle_target, dim=-1)) * confidence).sum() / confidence.sum().clamp_min(1e-6)

    centers = triangle_state["ctrl"].mean(dim=1)
    if centers.shape[0] > 0:
        mv_weight = triangle_state["confidence"] * triangle_state["angle_stability"].clamp(0.0, 1.0)
        mv_disp = torch.norm(centers - triangle_state["target_center"], dim=-1)
        losses["mv_cons"] = (mv_disp * mv_weight).sum() / mv_weight.sum().clamp_min(1e-6)

        depth_weight = triangle_state["confidence"] * triangle_state["depth_stability"].clamp(0.0, 1.0)
        ctrl_disp = torch.norm(triangle_state["ctrl"] - triangle_state["target_center"].unsqueeze(1), dim=-1).mean(dim=1)
        losses["depth_gate"] = (ctrl_disp * depth_weight).sum() / depth_weight.sum().clamp_min(1e-6)

        tri_vec_a = triangle_state["ctrl"][:, 1] - triangle_state["ctrl"][:, 0]
        tri_vec_b = triangle_state["ctrl"][:, 2] - triangle_state["ctrl"][:, 0]
        tri_area = 0.5 * torch.norm(torch.cross(tri_vec_a, tri_vec_b, dim=-1), dim=-1)
        ctrl_shift = torch.norm(triangle_state["ctrl"] - triangle_state["init_ctrl"], dim=-1).mean(dim=1)
        thickness = triangle_state["thickness"].squeeze(-1)
        rgb_sparse = triangle_state["rgb_residual"].abs().mean(dim=-1)
        losses["reg"] = (
            ctrl_shift.mean() +
            0.1 * tri_area.mean() +
            0.05 * thickness.mean() +
            0.1 * rgb_sparse.mean()
        )

    return losses, stats


def get_triangle_stage_code(stage_name):
    if stage_name == "triangle_only":
        return 1
    if stage_name == "joint":
        return 2
    return 0


def build_triangle_log_message(iteration, stage_name, triangle_total_count, triangle_active_count, triangle_view_count, triangle_visible_count, triangle_scale, triangle_loss_values, triangle_weights, triangle_support_active_pixels=0, triangle_refined_core_pixels=0, triangle_supervision_active_pixels=0):
    if triangle_loss_values:
        loss_terms = ", ".join(
            f"{name}={triangle_loss_values[name]:.6f}(w={triangle_weights.get(name, 0.0):.6f})"
            for name in sorted(triangle_loss_values.keys())
        )
    else:
        loss_terms = "none"
    return (
        f"[TRIANGLE][Iter {iteration}] stage={stage_name} "
        f"total={triangle_total_count} active={triangle_active_count} "
        f"view={triangle_view_count} visible={triangle_visible_count} "
        f"support_active_pixels={triangle_support_active_pixels} "
        f"refined_core_pixels={triangle_refined_core_pixels} "
        f"supervision_active_pixels={triangle_supervision_active_pixels} "
        f"scale={triangle_scale:.4f} losses={loss_terms}"
    )


def initialize_semantic_routing(gaussians, dataset, opt, logger=None):
    num_anchors = gaussians.get_anchor.shape[0]
    cluster_ids = torch.full((num_anchors,), -1, device="cuda", dtype=torch.long)
    semantic_features = gaussians.semantic_features
    has_valid_semantics = (
        semantic_features.dim() == 2 and
        semantic_features.shape[0] == num_anchors and
        semantic_features.shape[1] > 0
    )
    if gaussians.semantic_valid_mask.dim() == 1 and gaussians.semantic_valid_mask.shape[0] == num_anchors:
        valid_mask = gaussians.semantic_valid_mask
    elif has_valid_semantics:
        valid_mask = semantic_features.abs().sum(dim=1) > 0
    else:
        valid_mask = torch.zeros((num_anchors,), device="cuda", dtype=torch.bool)
    gaussians.set_semantic_routing(cluster_ids, valid_mask, initialized=True)

    if logger:
        valid_count = int(valid_mask.sum().item())
        logger.info(
            f"[SEMANTIC] Shared color head enabled. Expert routing disabled; "
            f"{valid_count} anchors keep semantic validity flags only."
        )


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sam_checkpoint, clip_model_path, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    boundary_mask_dir = os.path.join(dataset.source_path, "stable_edge_masks")
    boundary_mask_cache = {}
    center_weight_cache = {}
    # 1. 初始化高斯模型
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.semantic_num_experts)
    # 2. 加载场景
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)

    # ========================== [核心：语义提升逻辑 (已修改为支持断点加载)] ==========================
    # 定义保存路径
    sem_save_path = os.path.join(dataset.model_path, "anchor_semantic_features.pt")
    sem_conf_save_path = os.path.join(dataset.model_path, "anchor_semantic_confidence.pt")
    boundary_candidate_save_path = os.path.join(dataset.model_path, "boundary_candidates.pt")
    TARGET_SEMANTIC_DIM = 128
    boundary_candidates = None
    
    if os.path.exists(sem_save_path):
        logger.info(f"[SEMANTIC] Loading cached semantic features from {sem_save_path}")
        loaded_features = torch.load(sem_save_path, map_location="cuda")
        
        if loaded_features.shape[0] != gaussians.get_anchor.shape[0]:
            logger.warning(
                f"[SEMANTIC] Feature count mismatch: cache={loaded_features.shape[0]}, "
                f"anchors={gaussians.get_anchor.shape[0]}. Aligning to current anchor count."
            )
            aligned_features = torch.zeros((gaussians.get_anchor.shape[0], loaded_features.shape[1]), device="cuda")
            min_len = min(loaded_features.shape[0], gaussians.get_anchor.shape[0])
            aligned_features[:min_len] = loaded_features[:min_len]
            gaussians.semantic_features = aligned_features
        else:
            gaussians.semantic_features = loaded_features

        if os.path.exists(sem_conf_save_path):
            loaded_confidence = torch.load(sem_conf_save_path, map_location="cuda")
            if loaded_confidence.shape[0] != gaussians.get_anchor.shape[0]:
                aligned_confidence = torch.zeros((gaussians.get_anchor.shape[0],), device="cuda")
                min_len = min(loaded_confidence.shape[0], gaussians.get_anchor.shape[0])
                aligned_confidence[:min_len] = loaded_confidence[:min_len]
                gaussians.semantic_confidence = aligned_confidence
            else:
                gaussians.semantic_confidence = loaded_confidence
        else:
            logger.warning("[SEMANTIC] Missing confidence cache. Falling back to nonzero-feature confidence.")
            gaussians.semantic_confidence = (gaussians.semantic_features.abs().sum(dim=1) > 0).float()

        gaussians.semantic_valid_mask = gaussians.semantic_confidence > 0
        logger.info(f"[SEMANTIC] Loaded semantic features for {gaussians.semantic_features.shape[0]} anchors.")
        if os.path.exists(boundary_candidate_save_path):
            boundary_candidates = sanitize_boundary_candidates(
                torch.load(boundary_candidate_save_path, map_location="cuda"),
                gaussians.get_anchor.shape[0],
                device="cuda",
            )
            if boundary_candidates is not None:
                logger.info(
                    f"[TRIANGLE] Loaded cached boundary candidates from {boundary_candidate_save_path} "
                    f"({boundary_candidates['parent_anchor_indices'].shape[0]} entries)."
                )
        torch.cuda.empty_cache()

    if boundary_candidates is None and sam_checkpoint and os.path.exists(sam_checkpoint) and (
        not os.path.exists(sem_save_path) or not os.path.exists(boundary_candidate_save_path)
    ):
        if os.path.exists(sem_save_path):
            logger.info("[TRIANGLE] Semantic cache exists but boundary candidate cache is missing. Re-running semantic lifting for boundary candidates.")
        else:
            logger.info("[SEMANTIC] No cache found. Starting 2D-3D semantic lifting.")
        real_anchors = gaussians.get_anchor.detach()
        logger.info(f"[SEMANTIC] Lifting semantics for {real_anchors.shape[0]} anchors.")
        
        voter = SemanticVoter(
            dataset.source_path,
            sam_checkpoint,
            clip_model_name=clip_model_path,
            device="cuda",
            boundary_kernel=getattr(opt, "semantic_boundary_kernel", 5),
            min_interior_area=getattr(opt, "semantic_min_interior_pixels", 32),
            min_views=getattr(opt, "semantic_min_views", 2),
            boundary_mask_score_thresh=getattr(opt, "boundary_mask_score_thresh", 0.90),
            boundary_mask_min_area_ratio=getattr(opt, "boundary_mask_min_area_ratio", 0.005),
            boundary_mask_max_area_ratio=getattr(opt, "boundary_mask_max_area_ratio", 0.20),
            boundary_border_ignore_ratio=getattr(opt, "boundary_border_ignore_ratio", 0.10),
            boundary_center_inner_ratio=getattr(opt, "boundary_center_inner_ratio", 0.25),
            boundary_center_outer_ratio=getattr(opt, "boundary_center_outer_ratio", 0.15),
            boundary_center_min_overlap_ratio=getattr(opt, "boundary_center_min_overlap_ratio", 0.30),
            boundary_candidate_confidence_threshold=getattr(opt, "triangle_confidence_threshold", 0.55),
            boundary_candidate_max_count=getattr(opt, "triangle_max_candidates", 1024),
            target_feature_dim=getattr(opt, "triangle_feature_dim", 128),
        )
        semantic_result = voter.run(real_anchors)
        
        gaussians.semantic_features = semantic_result["features"].to("cuda")
        gaussians.semantic_confidence = semantic_result["confidence"].to("cuda")
        gaussians.semantic_valid_mask = semantic_result["valid_mask"].to("cuda", dtype=torch.bool)
        boundary_candidates = sanitize_boundary_candidates(
            semantic_result.get("boundary_candidates"),
            gaussians.get_anchor.shape[0],
            device="cuda",
        )
        
        torch.save(gaussians.semantic_features, sem_save_path)
        torch.save(gaussians.semantic_confidence, sem_conf_save_path)
        if boundary_candidates is not None:
            torch.save(boundary_candidates, boundary_candidate_save_path)
        logger.info(f"[SEMANTIC] Saved semantic features to {sem_save_path}")
        if boundary_candidates is not None:
            logger.info(
                f"[TRIANGLE] Saved {boundary_candidates['parent_anchor_indices'].shape[0]} boundary candidates to {boundary_candidate_save_path}"
            )

        del voter
        del semantic_result
        if 'real_anchors' in locals():
            del real_anchors
        torch.cuda.empty_cache()
        logger.info("[SEMANTIC] Released SAM/CLIP memory and continued to training.")

    elif not os.path.exists(sem_save_path):
        logger.info("[SEMANTIC] No SAM checkpoint and no cache found. Semantic initialization is skipped.")
        gaussians.semantic_features = torch.zeros((gaussians.get_anchor.shape[0], TARGET_SEMANTIC_DIM), device="cuda")
        gaussians.semantic_confidence = torch.zeros((gaussians.get_anchor.shape[0],), device="cuda")
        gaussians.semantic_valid_mask = torch.zeros((gaussians.get_anchor.shape[0],), device="cuda", dtype=torch.bool)
    if boundary_candidates is None and os.path.exists(boundary_candidate_save_path):
        boundary_candidates = sanitize_boundary_candidates(
            torch.load(boundary_candidate_save_path, map_location="cuda"),
            gaussians.get_anchor.shape[0],
            device="cuda",
        )
    # ===========================================================================================

    boundary_masks_ready = has_stable_edge_masks(boundary_mask_dir)
    if getattr(opt, "boundary_loss_weight", 0.0) > 0 and not boundary_masks_ready:
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            logger.info("[BOUNDARY] No stable edge masks found. Exporting SAM-based boundary masks for sharpening supervision.")
            boundary_voter = SemanticVoter(
                dataset.source_path,
                sam_checkpoint,
                clip_model_name=clip_model_path,
                device="cuda",
                boundary_kernel=getattr(opt, "semantic_boundary_kernel", 5),
                min_interior_area=getattr(opt, "semantic_min_interior_pixels", 32),
                min_views=getattr(opt, "semantic_min_views", 2),
                enable_clip=False,
                boundary_mask_score_thresh=getattr(opt, "boundary_mask_score_thresh", 0.90),
                boundary_mask_min_area_ratio=getattr(opt, "boundary_mask_min_area_ratio", 0.005),
                boundary_mask_max_area_ratio=getattr(opt, "boundary_mask_max_area_ratio", 0.20),
                boundary_border_ignore_ratio=getattr(opt, "boundary_border_ignore_ratio", 0.10),
                boundary_center_inner_ratio=getattr(opt, "boundary_center_inner_ratio", 0.25),
                boundary_center_outer_ratio=getattr(opt, "boundary_center_outer_ratio", 0.15),
                boundary_center_min_overlap_ratio=getattr(opt, "boundary_center_min_overlap_ratio", 0.30),
            )
            boundary_voter.export_stable_edge_masks()
            del boundary_voter
            torch.cuda.empty_cache()
            boundary_masks_ready = has_stable_edge_masks(boundary_mask_dir)
        else:
            logger.warning("[BOUNDARY] Stable edge masks are missing and SAM checkpoint is unavailable. Boundary sharpening supervision will be skipped.")

    # [新增] 在获取完语义特征后，立刻执行 PCA 降维并导出可供外部软件查看的点云文件
    if gaussians.semantic_features.shape[0] > 0 and gaussians.semantic_features.shape[1] >= 3:
        sem_ply_path = os.path.join(dataset.model_path, "anchor_semantic_pca_vis.ply")
        logger.info("\n[SEMANTIC] 正在提取并生成语义特征的 PCA 降维 3D 可视化点云...")
        export_semantic_point_cloud(gaussians.get_anchor, gaussians.semantic_features, sem_ply_path, logger)

    # 3. 设置优化器和学习率策略
    if gaussians.semantic_features.dim() == 2 and gaussians.semantic_features.shape[1] > 0:
        gaussians.init_semantic_adapter(gaussians.semantic_features.shape[1])

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if not gaussians.semantic_routing_initialized:
        initialize_semantic_routing(gaussians, dataset, opt, logger)

    def render_fn(viewpoint_camera, gaussian_model, pipeline, bg_color, **kwargs):
        return hybrid_render(viewpoint_camera, gaussian_model, pipeline, bg_color, opt, **kwargs)

    triangle_start_iter = get_triangle_supervision_start_iter(opt)
    triangle_joint_iter = get_triangle_joint_start_iter(opt)
    current_train_stage = "main"
    if logger:
        logger.info(
            f"[SEMANTIC] Supervision starts at iter {get_semantic_supervision_start_iter(opt)} "
            f"with confidence threshold {getattr(opt, 'semantic_confidence_threshold', 0.6):.2f}."
        )
        logger.info(
            f"[BOUNDARY] Supervision starts at iter {get_boundary_supervision_start_iter(opt)} "
            f"with weight {getattr(opt, 'boundary_loss_weight', 0.01):.4f}."
        )
        if boundary_masks_ready:
            logger.info(f"[BOUNDARY] Stable edge masks ready at {boundary_mask_dir}.")
        else:
            logger.info("[BOUNDARY] Stable edge masks not found; auxiliary boundary loss stays inactive.")
        if boundary_candidates is not None:
            logger.info(
                f"[TRIANGLE] Cached {boundary_candidates['parent_anchor_indices'].shape[0]} boundary candidates. "
                f"Triangle init starts at iter {triangle_start_iter}, joint finetune starts at iter {triangle_joint_iter}."
            )
        else:
            logger.info("[TRIANGLE] No boundary candidate cache available. Hybrid boundary branch will stay disabled.")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # [新增] 历史指标记录器，用于绘制训练过程的 PSNR 曲线
    iter_history = []
    psnr_history = []

    for iteration in range(first_iter, opt.iterations + 1):
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if iteration >= triangle_start_iter and not gaussians.triangle_initialized:
            initialized_triangles = gaussians.initialize_boundary_triangles(boundary_candidates, opt, logger)
            if initialized_triangles > 0:
                gaussians.add_triangle_optimizer_groups(opt)

        desired_stage = "main"
        if gaussians.has_boundary_triangles and iteration >= triangle_start_iter:
            desired_stage = "triangle_only" if iteration < triangle_joint_iter else "joint"
        if desired_stage != current_train_stage:
            if desired_stage == "main":
                gaussians.set_main_branch_trainable(True)
                gaussians.set_triangle_branch_trainable(False)
            elif desired_stage == "triangle_only":
                gaussians.set_main_branch_trainable(False)
                gaussians.set_triangle_branch_trainable(True)
            else:
                gaussians.set_main_branch_trainable(True)
                gaussians.set_triangle_branch_trainable(True)
            current_train_stage = desired_stage
            if logger is not None:
                logger.info(f"[TRIANGLE] Training stage switched to {current_train_stage} at iter {iteration}.")

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_fn(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        offset_selection_mask = render_pkg["selection_mask"]
        radii = render_pkg["radii"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.001 * scaling_reg

        semantic_loss = None
        semantic_weight = 0.0
        semantic_scale = get_semantic_loss_scale(iteration, opt)
        if opt.semantic_loss_weight > 0 and semantic_scale > 0 and current_train_stage != "triangle_only":
            semantic_loss = compute_semantic_loss(gaussians, voxel_visible_mask, opt)
            if semantic_loss is not None:
                semantic_weight = semantic_scale * opt.semantic_loss_weight
                loss = loss + semantic_weight * semantic_loss

        triangle_branch_active = gaussians.has_boundary_triangles and iteration >= triangle_start_iter
        boundary_loss = None
        boundary_weight = 0.0
        boundary_active_pixels = 0
        boundary_scale = get_boundary_loss_scale(iteration, opt)
        if (
            not triangle_branch_active and
            boundary_masks_ready and
            getattr(opt, "boundary_loss_weight", 0.0) > 0 and
            boundary_scale > 0
        ):
            boundary_loss, boundary_active_pixels = compute_boundary_sharpen_loss(
                image,
                gt_image,
                viewpoint_cam.image_name,
                boundary_mask_dir,
                opt,
                boundary_mask_cache,
                center_weight_cache,
            )
            if boundary_loss is not None:
                boundary_weight = boundary_scale * opt.boundary_loss_weight
                loss = loss + boundary_weight * boundary_loss

        triangle_scale = get_triangle_loss_scale(iteration, opt)
        triangle_losses = {}
        triangle_stats = {
            "support_active_pixels": 0,
            "refined_core_pixels": 0,
            "supervision_active_pixels": 0,
        }
        triangle_weights = {}
        if triangle_branch_active and triangle_scale > 0:
            triangle_losses, triangle_stats = compute_triangle_branch_losses(
                render_pkg,
                gt_image,
                viewpoint_cam.image_name,
                boundary_mask_dir,
                opt,
                boundary_mask_cache,
                center_weight_cache,
                gaussians,
            )
            triangle_weight_map = {
                "mask": getattr(opt, "triangle_loss_weight_mask", 0.02),
                "edge_rgb": getattr(opt, "triangle_loss_weight_edge_rgb", 0.01),
                "semantic": getattr(opt, "triangle_loss_weight_sem", 0.005),
                "mv_cons": getattr(opt, "triangle_loss_weight_mv", 0.002),
                "depth_gate": getattr(opt, "triangle_loss_weight_depth", 0.002),
                "reg": getattr(opt, "triangle_loss_weight_reg", 0.001),
            }
            for name, term in triangle_losses.items():
                base_weight = float(triangle_weight_map.get(name, 0.0))
                if term is None or base_weight <= 0:
                    continue
                applied_weight = triangle_scale * base_weight
                triangle_weights[name] = applied_weight
                loss = loss + applied_weight * term

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            semantic_loss_value = semantic_loss.item() if semantic_loss is not None else None
            boundary_loss_value = boundary_loss.item() if boundary_loss is not None else None
            triangle_loss_values = {name: term.item() for name, term in triangle_losses.items()}
            triangle_total_count = int(gaussians._triangle_ctrl.shape[0]) if gaussians._triangle_ctrl.dim() == 3 else 0
            triangle_active_count = int(gaussians._triangle_active_mask.sum().item()) if gaussians._triangle_active_mask.dim() == 1 else 0
            triangle_view_count = int(render_pkg.get("triangle_count", 0))
            triangle_visible_count = int(render_pkg.get("triangle_visible_count", 0))
            triangle_support_active_pixels = int(triangle_stats.get("support_active_pixels", 0))
            triangle_refined_core_pixels = int(triangle_stats.get("refined_core_pixels", 0))
            triangle_supervision_active_pixels = int(triangle_stats.get("supervision_active_pixels", 0))
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Stage": current_train_stage,
                    "TriVis": triangle_visible_count,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer and semantic_loss_value is not None:
                tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/semantic_loss', semantic_loss_value, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/semantic_weight', semantic_weight, iteration)
            if wandb is not None and semantic_loss_value is not None:
                wandb.log({"train_semantic_loss": semantic_loss_value, "train_semantic_weight": semantic_weight})

            if tb_writer and boundary_loss_value is not None:
                tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/boundary_loss', boundary_loss_value, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/boundary_weight', boundary_weight, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/boundary_active_pixels', boundary_active_pixels, iteration)
            if wandb is not None and boundary_loss_value is not None:
                wandb.log({
                    "train_boundary_loss": boundary_loss_value,
                    "train_boundary_weight": boundary_weight,
                    "train_boundary_active_pixels": boundary_active_pixels,
                })

            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/scale', triangle_scale, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/stage_code', get_triangle_stage_code(current_train_stage), iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/branch_active', float(triangle_branch_active), iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/total_count', triangle_total_count, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/active_count', triangle_active_count, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/view_count', triangle_view_count, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/visible_count', triangle_visible_count, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/support_active_pixels', triangle_support_active_pixels, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/refined_core_pixels', triangle_refined_core_pixels, iteration)
                tb_writer.add_scalar(f'{dataset_name}/train_triangle/supervision_active_pixels', triangle_supervision_active_pixels, iteration)
            if wandb is not None:
                wandb.log({
                    "train_triangle_scale": triangle_scale,
                    "train_triangle_stage_code": get_triangle_stage_code(current_train_stage),
                    "train_triangle_branch_active": float(triangle_branch_active),
                    "train_triangle_total_count": triangle_total_count,
                    "train_triangle_active_count": triangle_active_count,
                    "train_triangle_view_count": triangle_view_count,
                    "train_triangle_visible_count": triangle_visible_count,
                    "train_triangle_support_active_pixels": triangle_support_active_pixels,
                    "train_triangle_refined_core_pixels": triangle_refined_core_pixels,
                    "train_triangle_supervision_active_pixels": triangle_supervision_active_pixels,
                })

            for loss_name, loss_value in triangle_loss_values.items():
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/triangle_{loss_name}', loss_value, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/triangle_{loss_name}_weight', triangle_weights.get(loss_name, 0.0), iteration)
                if wandb is not None:
                    wandb.log({
                        f"train_triangle_{loss_name}": loss_value,
                        f"train_triangle_{loss_name}_weight": triangle_weights.get(loss_name, 0.0),
                    })

            triangle_log_interval = max(1, int(getattr(opt, "triangle_log_interval", 50)))
            should_log_triangle = (
                iteration == triangle_start_iter or
                iteration == triangle_joint_iter or
                (triangle_branch_active and iteration % triangle_log_interval == 0) or
                (gaussians.triangle_initialized and triangle_visible_count == 0 and iteration % 10 == 0)
            )
            if logger is not None and should_log_triangle:
                logger.info(
                    build_triangle_log_message(
                        iteration,
                        current_train_stage,
                        triangle_total_count,
                        triangle_active_count,
                        triangle_view_count,
                        triangle_visible_count,
                        triangle_scale,
                        triangle_loss_values,
                        triangle_weights,
                        triangle_support_active_pixels,
                        triangle_refined_core_pixels,
                        triangle_supervision_active_pixels,
                    )
                )

            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fn, (pipe, background), wandb, logger, iter_history, psnr_history)

            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in checkpoint_iterations:
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if MATPLOTLIB_FOUND and len(iter_history) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(iter_history, psnr_history, marker='o', linestyle='-', color='r')
        plt.title('Test PSNR over Training Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plot_path = os.path.join(dataset.model_path, 'training_psnr_curve.png')
        plt.savefig(plot_path)
        plt.close()
        if logger: logger.info(f"[PLOT] Training PSNR curve saved to {plot_path}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, iter_history=None, psnr_history=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # [保留] 训练过程中的详细图像对比记录（前 30 张）
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                if wandb is not None:
                    gt_image_list, render_image_list, errormap_list = [], [], []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # [新增] 把训练阶段获取的 PSNR 记录下来，方便画图
                if config['name'] == 'test':
                    if iter_history is not None: iter_history.append(iteration)
                    val = psnr_test.item() if hasattr(psnr_test, 'item') else psnr_test
                    if psnr_history is not None: psnr_history.append(val)

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()
        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, opt):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True); makedirs(error_path, exist_ok=True); makedirs(gts_path, exist_ok=True)
    
    t_list, visible_count_list, per_view_dict = [], [], {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = hybrid_render(view, gaussians, pipeline, background, opt, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()
        t_list.append(t_end - t_start)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)
        gt = view.original_image[0:3, :, :]
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    return t_list, visible_count_list


def log_fps_stats(name, timings, logger):
    if logger is None:
        return
    if not timings:
        logger.warning(f"[EVAL] No rendered views for {name}; skipping FPS calculation.")
        return
    valid_timings = timings[5:] if len(timings) > 5 else timings
    if len(valid_timings) == 0:
        logger.warning(f"[EVAL] Not enough rendered views for {name}; skipping FPS calculation.")
        return
    mean_time = torch.tensor(valid_timings).mean().item()
    if mean_time <= 0:
        logger.warning(f"[EVAL] Invalid mean render time for {name}; skipping FPS calculation.")
        return
    logger.info(f'{name} FPS: {1.0 / mean_time:.5f}')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt=None, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.semantic_num_experts)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if opt is None:
            opt = Namespace()
        
        if not skip_train:
            t_train_list, _ = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, opt)
            log_fps_stats("Train", t_train_list, logger)

        visible_count = 0
        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, opt)
            log_fps_stats("Test", t_test_list, logger)
    return visible_count

def readImages(renders_dir, gt_dir):
    renders, gts, image_names = [], [], []
    if not renders_dir.exists() or not gt_dir.exists():
        return renders, gts, image_names
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    # [修改 2] 在真正需要的时候才初始化 LPIPS，避免 GPU 冲突
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
    
    # [保留] 原始 train.py 的全量字典结构评估逻辑
    full_dict, per_view_dict = {}, {}
    scene_dir = model_paths
    full_dict[scene_dir], per_view_dict[scene_dir] = {}, {}
    test_dir = Path(scene_dir) / "test"

    if not test_dir.exists():
        if logger:
            logger.warning(f"[EVAL] Test directory not found: {test_dir}. Skipping evaluation.")
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        return

    method_names = [method for method in os.listdir(test_dir) if (test_dir / method).is_dir()]
    if len(method_names) == 0:
        if logger:
            logger.warning(f"[EVAL] No test render methods found under: {test_dir}. Skipping evaluation.")
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        return

    for method in method_names:
        full_dict[scene_dir][method], per_view_dict[scene_dir][method] = {}, {}
        method_dir = test_dir / method
        renders, gts, image_names = readImages(method_dir / "renders", method_dir / "gt")
        if len(image_names) == 0:
            if logger:
                logger.warning(f"[EVAL] No rendered images found for method {method} in {method_dir}. Skipping metric aggregation.")
            full_dict[scene_dir][method].update({"num_views": 0})
            per_view_dict[scene_dir][method].update({"SSIM": {}, "PSNR": {}})
            continue
        ssims, psnrs, lpipss = [], [], []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(), "PSNR": torch.tensor(psnrs).mean().item(), "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {n: s.item() for s, n in zip(ssims, image_names)}, "PSNR": {n: p.item() for p, n in zip(psnrs, image_names)}})

        # ==========================================================
        # [新增模块]：生成可视化曲线图并直接拷贝测试图片出来
        # ==========================================================
        if MATPLOTLIB_FOUND and len(psnrs) > 0:
            psnr_list = [p.item() for p in psnrs]
            plt.figure(figsize=(12, 5))
            plt.plot(range(len(psnr_list)), psnr_list, marker='.', linestyle='-', color='b')
            plt.title(f'Per-View Test PSNR ({method})')
            plt.xlabel('View Index')
            plt.ylabel('PSNR (dB)')
            plt.grid(True)
            
            # 绘制均值辅助线
            mean_psnr = sum(psnr_list) / len(psnr_list)
            plt.axhline(y=mean_psnr, color='r', linestyle='--', label=f'Mean PSNR: {mean_psnr:.2f} dB')
            plt.legend()
            
            plot_path = os.path.join(scene_dir, f'per_view_psnr_curve_{method}.png')
            plt.savefig(plot_path)
            plt.close()
            if logger: logger.info(f"[PLOT] Per-view PSNR curve saved to {plot_path}")

        # 将生成的测试渲染图提取并拷贝到最外层目录，方便使用者查看
        easy_access_dir = Path(scene_dir) / f"test_renders_easy_access_{method}"
        os.makedirs(easy_access_dir, exist_ok=True)
        
        for fname in image_names:
            src_file = method_dir / "renders" / fname
            dst_file = easy_access_dir / fname
            if src_file.exists():
                shutil.copy(src_file, dst_file)
        
        if logger: logger.info(f"[EXPORT] All test render images have been copied to: {easy_access_dir}")
        # ==========================================================

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    logger.info("Evaluation complete.")

def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    controlshow = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter); controlshow.setFormatter(formatter)
    logger.addHandler(fileinfo); logger.addHandler(controlshow)
    return logger

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--enable_backup", action="store_true", default=False, help="Copy a source snapshot into <model_path>/backup before training.")
    # [新增] 参数指定 SAM 权重路径
    parser.add_argument("--sam_checkpoint", type=str, default="./weights/sam_hq_vit_base", help="Path to SAM-HQ weights")
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-base-patch32", help="Local path or HF id for CLIP image model")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # ??????
    if args.enable_backup:
        try:
            saveRuntimeCode(os.path.join(args.model_path, 'backup'))
        except Exception as e:
            logger.info(f'Save code failed: {e}')
    else:
        logger.info('Code backup disabled. Skipping backup snapshot.')

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 启动训练主程序
    training(lp.extract(args), op.extract(args), pp.extract(args), args.source_path.split('/')[-1],  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sam_checkpoint, args.clip_model_path, wandb=None, logger=logger)
    
    # 后处理：渲染训练/测试集并评估
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), op.extract(args), skip_train=False, logger=logger)
    evaluate(args.model_path, visible_count=visible_count, logger=logger)
