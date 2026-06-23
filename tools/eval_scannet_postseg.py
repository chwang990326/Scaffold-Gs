import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm
from transformers import CLIPTokenizer
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_init_main import SAMCLIPSegmentor


NYU40_NAMES = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refrigerator",
    28: "showercurtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
}

TARGET_IDS = {
    "19": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    "15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    "10": [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}

SCANNET_SEMSEG_NUM_CLASSES = len(TARGET_IDS["19"]) + 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_label_mesh(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    labels = np.asarray(vertex["label"], dtype=np.int32)
    faces = []
    for face in ply["face"].data["vertex_indices"]:
        if len(face) == 3:
            faces.append(face)
    if not faces:
        raise RuntimeError(f"No triangular faces found in {path}")
    return points, labels, np.asarray(faces, dtype=np.int32)


def load_transform_meta(scene_data_dir):
    path = scene_data_dir / "transforms_train.json"
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "width": int(data["w"]),
        "height": int(data["h"]),
        "fx": float(data["fl_x"]),
        "fy": float(data["fl_y"]),
        "cx": float(data.get("cx", (float(data["w"]) - 1.0) * 0.5)),
        "cy": float(data.get("cy", (float(data["h"]) - 1.0) * 0.5)),
    }


def sorted_render_files(render_dir):
    return sorted(render_dir.glob("*.png"), key=lambda p: int(p.stem))


def build_test_view_mapping(model_scene_dir, method_name):
    camera_path = model_scene_dir / "cameras.json"
    train_path = model_scene_dir / "sparse_train_views.json"
    render_dir = model_scene_dir / "test" / method_name / "renders"

    with open(camera_path, "r") as f:
        cameras = json.load(f)
    with open(train_path, "r") as f:
        train_names = set(str(name) for name in json.load(f)["image_names"])

    render_files = sorted_render_files(render_dir)
    test_cameras = [camera for camera in cameras if str(camera["img_name"]) not in train_names]
    if len(test_cameras) != len(render_files):
        # Scene writes test cameras before train cameras when cameras.json is first created.
        test_cameras = cameras[: len(render_files)]
    if len(test_cameras) != len(render_files):
        raise RuntimeError(
            f"Cannot map renders to cameras in {model_scene_dir}: "
            f"{len(render_files)} renders vs {len(test_cameras)} test cameras."
        )

    return [
        {
            "render_name": render_file.name,
            "render_path": str(render_file),
            "img_name": str(camera["img_name"]),
            "camera": camera,
        }
        for render_file, camera in zip(render_files, test_cameras)
    ]


def project_points(points, camera, intrinsics, out_width, out_height):
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    position = np.asarray(camera["position"], dtype=np.float32)
    camera_points = (points - position) @ rotation

    sx = float(out_width) / float(camera["width"])
    sy = float(out_height) / float(camera["height"])
    fx = intrinsics["fx"] * sx
    fy = intrinsics["fy"] * sy
    cx = intrinsics["cx"] * sx
    cy = intrinsics["cy"] * sy

    z = camera_points[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * (camera_points[:, 0] / safe_z) + cx
    v = fy * (camera_points[:, 1] / safe_z) + cy
    return u, v, z


def majority_label(labels):
    labels = labels[labels > 0]
    if labels.size == 0:
        return 0
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def render_gt_semantic(points, labels, faces, camera, intrinsics, width, height):
    u, v, z = project_points(points, camera, intrinsics, width, height)
    projected = np.stack([u, v], axis=1)
    face_vertices = faces
    face_z = z[face_vertices].mean(axis=1)

    z_valid = np.all(z[face_vertices] > 1e-5, axis=1)
    xy = projected[face_vertices]
    x_min = xy[:, :, 0].min(axis=1)
    x_max = xy[:, :, 0].max(axis=1)
    y_min = xy[:, :, 1].min(axis=1)
    y_max = xy[:, :, 1].max(axis=1)
    in_frame = (x_max >= 0) & (x_min < width) & (y_max >= 0) & (y_min < height)
    valid_face_indices = np.where(z_valid & in_frame)[0]

    label_map = np.zeros((height, width), dtype=np.uint16)
    for face_idx in valid_face_indices[np.argsort(face_z[valid_face_indices])[::-1]]:
        tri = np.rint(projected[face_vertices[face_idx]]).astype(np.int32)
        if np.any(~np.isfinite(tri)):
            continue
        label = majority_label(labels[face_vertices[face_idx]])
        if label <= 0:
            continue
        cv2.fillConvexPoly(label_map, tri, int(label), lineType=cv2.LINE_8)
    return label_map


def get_or_create_gt_mask(args, scene, view, mesh_cache, intrinsics):
    render_path = Path(view["render_path"])
    with Image.open(render_path) as image:
        width, height = image.size

    gt_dir = Path(args.output_root) / "gt_semantic" / scene
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"{Path(view['render_name']).stem}_{view['img_name']}.npy"
    if gt_path.exists() and not args.refresh_gt:
        return np.load(gt_path)

    points, labels, faces = mesh_cache
    gt = render_gt_semantic(points, labels, faces, view["camera"], intrinsics, width, height)
    np.save(gt_path, gt)
    vis = colorize_label_map(gt)
    cv2.imwrite(str(gt_path.with_suffix(".png")), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return gt


def remap_labels(label_map, target_ids):
    remapped = np.zeros(label_map.shape, dtype=np.int32)
    for new_id, original_id in enumerate(target_ids, start=1):
        remapped[label_map == original_id] = new_id
    return remapped


def remap_pred19_to_target(pred19, target_ids):
    if target_ids == TARGET_IDS["19"]:
        return pred19.astype(np.int32, copy=False)
    original_to_target = {original_id: idx for idx, original_id in enumerate(target_ids, start=1)}
    remapped = np.zeros(pred19.shape, dtype=np.int32)
    for pred_id, original_id in enumerate(TARGET_IDS["19"], start=1):
        target_id = original_to_target.get(original_id)
        if target_id is not None:
            remapped[pred19 == pred_id] = target_id
    return remapped


def update_confusion(confusion, gt, pred, num_classes):
    valid = gt > 0
    if not np.any(valid):
        return
    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)
    pred_valid[(pred_valid < 0) | (pred_valid > num_classes)] = 0
    idx = gt_valid * (num_classes + 1) + pred_valid
    counts = np.bincount(idx, minlength=(num_classes + 1) * (num_classes + 1))
    confusion += counts.reshape(num_classes + 1, num_classes + 1)


def metrics_from_confusion(confusion):
    num_classes = confusion.shape[0] - 1
    ious = []
    accs = []
    for cls in range(1, num_classes + 1):
        tp = float(confusion[cls, cls])
        gt_count = float(confusion[cls, :].sum())
        pred_count = float(confusion[:, cls].sum())
        union = gt_count + pred_count - tp
        if gt_count > 0:
            accs.append(tp / gt_count)
        if union > 0:
            ious.append(tp / union)

    valid_pixels = float(confusion[1:, :].sum())
    correct_pixels = float(np.trace(confusion[1:, 1:]))
    return {
        "mIoU": 100.0 * float(np.mean(ious)) if ious else float("nan"),
        "mAcc": 100.0 * float(np.mean(accs)) if accs else float("nan"),
        "pixelAcc": 100.0 * correct_pixels / valid_pixels if valid_pixels > 0 else float("nan"),
        "validPixels": int(valid_pixels),
    }


def color_for_label(label):
    if label == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    rng = np.random.default_rng(int(label) * 1009)
    return rng.integers(32, 255, size=3, dtype=np.uint8)


def colorize_label_map(label_map):
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label in np.unique(label_map):
        if label == 0:
            continue
        color[label_map == label] = color_for_label(int(label))
    return color


def make_overlay(image_rgb, label_map, alpha=0.55):
    color = colorize_label_map(label_map)
    mask = label_map > 0
    overlay = image_rgb.copy()
    overlay[mask] = (
        image_rgb[mask].astype(np.float32) * (1.0 - alpha)
        + color[mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def add_title(image_rgb, title):
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 280), 30), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_overlay(args, method, scene, render_name, gt_original, pred_remapped, target_key, target_ids, image_path):
    if target_key != args.overlay_target:
        return
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gt_remapped = remap_labels(gt_original, target_ids)
    render_panel = add_title(image_rgb, "render")
    gt_panel = add_title(make_overlay(image_rgb, gt_remapped), f"GT {target_key}cls")
    pred_panel = add_title(make_overlay(image_rgb, pred_remapped), f"pred {target_key}cls")
    panel = np.concatenate([render_panel, gt_panel, pred_panel], axis=1)
    out_dir = Path(args.output_root) / "overlays" / method / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / render_name
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def build_scannet_semseg_model(num_classes=SCANNET_SEMSEG_NUM_CLASSES):
    return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)


def load_scannet_semseg_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        num_classes = int(checkpoint.get("num_classes", SCANNET_SEMSEG_NUM_CLASSES))
    else:
        state_dict = checkpoint
        num_classes = SCANNET_SEMSEG_NUM_CLASSES
        checkpoint = {}
    model = build_scannet_semseg_model(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


class SamClipPostSegmenter:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.segmentor = SAMCLIPSegmentor(
            args.sam_model_path,
            clip_model_name=args.clip_model_path,
            device=args.device,
            enable_clip=True,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path, local_files_only=True)
        self.text_feature_cache = {}

    @torch.no_grad()
    def text_features(self, target_key):
        if target_key in self.text_feature_cache:
            return self.text_feature_cache[target_key]
        target_names = [NYU40_NAMES[idx] for idx in TARGET_IDS[target_key]]
        prompts = [self.args.prompt_template.format(name=name) for name in target_names]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        feats = self.segmentor.clip_model.get_text_features(**inputs)
        feats = F.normalize(feats, p=2, dim=1)
        self.text_feature_cache[target_key] = feats
        return feats

    def cache_path(self, method, scene, render_name):
        return Path(self.args.output_root) / "sam_clip_cache" / method / scene / f"{Path(render_name).stem}.pt"

    def segment(self, method, scene, render_name, image_path):
        cache_path = self.cache_path(method, scene, render_name)
        if cache_path.exists() and not self.args.refresh_postseg:
            cached = torch.load(cache_path, map_location="cpu")
            return cached["id_map"].numpy(), cached["features"], cached.get("scores", {})

        id_map, _, id_to_feat, id_to_score, _ = self.segmentor.process_image(
            str(image_path),
            points_per_side=self.args.points_per_side,
            iou_thresh=self.args.sam_iou_thresh,
            area_thresh=self.args.sam_area_thresh,
            boundary_kernel=self.args.boundary_kernel,
            min_interior_area=self.args.min_interior_area,
            compute_clip_features=True,
        )
        features = {int(k): v.detach().cpu() for k, v in id_to_feat.items()}
        id_map_cpu = id_map.detach().cpu().to(torch.int32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"id_map": id_map_cpu, "features": features, "scores": id_to_score}, cache_path)
        return id_map_cpu.numpy(), features, id_to_score

    @torch.no_grad()
    def classify(self, id_map, features, target_key):
        pred = np.zeros(id_map.shape, dtype=np.int32)
        if not features:
            return pred
        mask_ids = sorted(features)
        image_features = torch.stack([features[mid] for mid in mask_ids], dim=0).to(self.device)
        image_features = F.normalize(image_features, p=2, dim=1)
        scores = image_features @ self.text_features(target_key).t()
        conf, cls = torch.max(scores, dim=1)
        cls = cls.detach().cpu().numpy().astype(np.int32) + 1
        conf = conf.detach().cpu().numpy()
        for mid, mapped_class, score in zip(mask_ids, cls, conf):
            if score < self.args.clip_confidence_threshold:
                continue
            pred[id_map == mid] = int(mapped_class)
        return pred


class ScannetSemSegPostSegmenter:
    def __init__(self, args):
        if not args.semseg_checkpoint:
            raise ValueError("--semseg_checkpoint is required for --postseg_backend scannet_semseg")
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.checkpoint = load_scannet_semseg_checkpoint(args.semseg_checkpoint, self.device)
        self.input_width = int(args.semseg_input_width or self.checkpoint.get("input_width", 512))
        self.input_height = int(args.semseg_input_height or self.checkpoint.get("input_height", 384))

    def cache_path(self, method, scene, render_name):
        stem = Path(render_name).stem
        return Path(self.args.output_root) / "scannet_semseg_cache" / method / scene / f"{stem}.npy"

    @torch.no_grad()
    def segment(self, method, scene, render_name, image_path):
        cache_path = self.cache_path(method, scene, render_name)
        if cache_path.exists() and not self.args.refresh_postseg:
            return np.load(cache_path), None, None

        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        resized = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(self.device)
        logits = self.model(tensor)["out"]
        pred_small = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred_small, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pred)
        return pred, None, None

    def classify(self, pred19, features, target_key):
        return remap_pred19_to_target(pred19, TARGET_IDS[target_key])


def make_post_segmenter(args):
    if args.gt_only:
        return None
    if args.postseg_backend == "sam_clip":
        return SamClipPostSegmenter(args)
    if args.postseg_backend == "scannet_semseg":
        return ScannetSemSegPostSegmenter(args)
    raise ValueError(f"Unknown postseg_backend={args.postseg_backend}")


def collect_scenes(baseline_root, ours_root, requested_scenes):
    if requested_scenes:
        return requested_scenes
    baseline_scenes = {
        path.name
        for path in Path(baseline_root).iterdir()
        if path.is_dir() and (path / "cameras.json").exists()
    }
    ours_scenes = {
        path.name
        for path in Path(ours_root).iterdir()
        if path.is_dir() and (path / "cameras.json").exists()
    }
    return sorted(baseline_scenes & ours_scenes)


def mean_or_nan(values):
    values = [value for value in values if not math.isnan(float(value))]
    return float(np.mean(values)) if values else float("nan")


def evaluate(args):
    scenes = collect_scenes(args.baseline_root, args.ours_root, args.scenes)
    if not scenes:
        raise RuntimeError("No ScanNet scenes found.")

    methods = {
        "baseline": Path(args.baseline_root),
        "ours": Path(args.ours_root),
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    post_segmenter = make_post_segmenter(args)
    summary_confusions = {
        method: {
            target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
            for target_key in args.targets
        }
        for method in methods
    }
    results = {
        "metadata": {
            "data_root": args.data_root,
            "baseline_root": args.baseline_root,
            "ours_root": args.ours_root,
            "method_name": args.method_name,
            "postseg_backend": "gt_only" if args.gt_only else args.postseg_backend,
            "semseg_checkpoint": args.semseg_checkpoint if args.postseg_backend == "scannet_semseg" else None,
            "targets": {key: [NYU40_NAMES[idx] for idx in TARGET_IDS[key]] for key in args.targets},
            "device": args.device,
            "points_per_side": args.points_per_side,
            "gt_only": args.gt_only,
        },
        "scenes": {},
    }

    for scene in tqdm(scenes, desc="Scenes"):
        scene_data_dir = Path(args.data_root) / scene
        label_mesh_path = scene_data_dir / f"{scene}_vh_clean_2.labels.ply"
        mesh_cache = read_label_mesh(label_mesh_path)
        intrinsics = load_transform_meta(scene_data_dir)
        baseline_mapping = build_test_view_mapping(methods["baseline"] / scene, args.method_name)
        scene_result = {"view_mapping": [], "methods": {}}
        for view in baseline_mapping:
            scene_result["view_mapping"].append(
                {
                    "render_name": view["render_name"],
                    "img_name": view["img_name"],
                }
            )

        for method, method_root in methods.items():
            mapping = build_test_view_mapping(method_root / scene, args.method_name)
            method_confusions = {
                target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
                for target_key in args.targets
            }
            method_views = {}
            max_views = len(mapping) if args.max_views_per_scene <= 0 else min(args.max_views_per_scene, len(mapping))
            iterator = tqdm(mapping[:max_views], desc=f"{scene} {method}", leave=False)
            for view in iterator:
                image_path = Path(view["render_path"])
                gt_original = get_or_create_gt_mask(args, scene, view, mesh_cache, intrinsics)
                if args.gt_only:
                    id_map, features = np.zeros(gt_original.shape, dtype=np.int32), {}
                else:
                    id_map, features, _ = post_segmenter.segment(method, scene, view["render_name"], image_path)

                view_metrics = {}
                for target_key in args.targets:
                    target_ids = TARGET_IDS[target_key]
                    gt = remap_labels(gt_original, target_ids)
                    pred = np.zeros_like(gt) if args.gt_only else post_segmenter.classify(id_map, features, target_key)
                    confusion = np.zeros((len(target_ids) + 1, len(target_ids) + 1), dtype=np.int64)
                    update_confusion(confusion, gt, pred, len(target_ids))
                    update_confusion(method_confusions[target_key], gt, pred, len(target_ids))
                    update_confusion(summary_confusions[method][target_key], gt, pred, len(target_ids))
                    view_metrics[target_key] = metrics_from_confusion(confusion)
                    if (not args.disable_overlay) and (
                        args.overlay_limit_per_scene <= 0
                        or len(list((Path(args.output_root) / "overlays" / method / scene).glob("*.png")))
                        < args.overlay_limit_per_scene
                    ):
                        save_overlay(
                            args,
                            method,
                            scene,
                            view["render_name"],
                            gt_original,
                            pred,
                            target_key,
                            target_ids,
                            image_path,
                        )
                method_views[view["render_name"]] = {
                    "img_name": view["img_name"],
                    "metrics": view_metrics,
                }

            scene_result["methods"][method] = {
                "metrics": {
                    target_key: metrics_from_confusion(method_confusions[target_key])
                    for target_key in args.targets
                },
                "views": method_views,
            }
        results["scenes"][scene] = scene_result

    results["summary"] = {
        method: {
            target_key: metrics_from_confusion(summary_confusions[method][target_key])
            for target_key in args.targets
        }
        for method in methods
    }
    results["summary"]["delta_ours_minus_baseline"] = {}
    for target_key in args.targets:
        results["summary"]["delta_ours_minus_baseline"][target_key] = {
            metric: results["summary"]["ours"][target_key][metric]
            - results["summary"]["baseline"][target_key][metric]
            for metric in ["mIoU", "mAcc", "pixelAcc"]
        }

    json_path = output_root / "semantic_postseg_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    write_markdown_table(output_root / "semantic_postseg_table.md", results, args.targets)
    return results


def format_float(value):
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.2f}"


def write_markdown_table(path, results, targets):
    scene_target = "19" if "19" in targets else targets[0]
    backend = results.get("metadata", {}).get("postseg_backend", "postseg")
    lines = [
        f"# ScanNet Full Test-View {backend} Post-Segmentation",
        "",
        "| Method | Target | mIoU | mAcc | Pixel Acc | Valid Pixels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["baseline", "ours"]:
        for target_key in targets:
            metrics = results["summary"][method][target_key]
            lines.append(
                f"| {method} | {target_key} | {format_float(metrics['mIoU'])} | "
                f"{format_float(metrics['mAcc'])} | {format_float(metrics['pixelAcc'])} | "
                f"{metrics['validPixels']} |"
            )
    lines.extend(
        [
            "",
            "| Delta | Target | mIoU | mAcc | Pixel Acc |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target_key in targets:
        metrics = results["summary"]["delta_ours_minus_baseline"][target_key]
        lines.append(
            f"| ours - baseline | {target_key} | {format_float(metrics['mIoU'])} | "
            f"{format_float(metrics['mAcc'])} | {format_float(metrics['pixelAcc'])} |"
        )
    lines.append("")
    lines.append(f"## Per-Scene {scene_target}-Class Summary")
    lines.append("")
    lines.append("| Scene | Baseline mIoU | Ours mIoU | Delta | Baseline Pixel Acc | Ours Pixel Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for scene, scene_result in results["scenes"].items():
        b = scene_result["methods"]["baseline"]["metrics"][scene_target]
        o = scene_result["methods"]["ours"]["metrics"][scene_target]
        lines.append(
            f"| {scene} | {format_float(b['mIoU'])} | {format_float(o['mIoU'])} | "
            f"{format_float(o['mIoU'] - b['mIoU'])} | {format_float(b['pixelAcc'])} | "
            f"{format_float(o['pixelAcc'])} |"
        )
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ScanNet full-view renders with projected GT semantics and 2D post-segmentation."
    )
    parser.add_argument("--data_root", default="/home/chwang/data/InstanceGS_data/scannet_wym")
    parser.add_argument("--baseline_root", default="/home/chwang/output/scannet10_full_scaffoldgs")
    parser.add_argument("--ours_root", default="/home/chwang/output/scannet10_full_ours_expert8_b10")
    parser.add_argument("--output_root", default="/home/chwang/output/semantic_eval/scannet10_full_postseg")
    parser.add_argument("--method_name", default="ours_30000")
    parser.add_argument("--postseg_backend", default="sam_clip", choices=["sam_clip", "scannet_semseg"])
    parser.add_argument("--sam_model_path", default="./weights/sam_hq_vit_base")
    parser.add_argument("--clip_model_path", default="./weights/clip-vit-base-patch32")
    parser.add_argument("--semseg_checkpoint", default=None)
    parser.add_argument("--semseg_input_width", type=int, default=0)
    parser.add_argument("--semseg_input_height", type=int, default=0)
    parser.add_argument("--prompt_template", default="a photo of a {name}")
    parser.add_argument("--targets", nargs="+", default=["19", "15", "10"], choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--points_per_side", type=int, default=16)
    parser.add_argument("--sam_iou_thresh", type=float, default=0.88)
    parser.add_argument("--sam_area_thresh", type=int, default=100)
    parser.add_argument("--boundary_kernel", type=int, default=5)
    parser.add_argument("--min_interior_area", type=int, default=32)
    parser.add_argument("--clip_confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--overlay_target", default="19", choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--overlay_limit_per_scene", type=int, default=3)
    parser.add_argument("--max_views_per_scene", type=int, default=0)
    parser.add_argument("--refresh_gt", action="store_true")
    parser.add_argument("--refresh_postseg", action="store_true")
    parser.add_argument("--disable_overlay", action="store_true")
    parser.add_argument("--gt_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
