import argparse
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer, SamHQModel, SamHQProcessor, SamModel, SamProcessor

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


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
    28: "shower curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
}

TARGET_IDS = {
    "19": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    "15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    "10": [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}

PALETTE = np.array(
    [
        [0, 0, 0],
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [178, 76, 76],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
    ],
    dtype=np.uint8,
)


def fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(float(fov) / 2.0))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_labels_from_ply(path):
    if PlyData is not None:
        ply = PlyData.read(path)
        vertex = ply["vertex"].data
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
        labels = np.asarray(vertex["label"]).astype(np.int64)
        return points, labels

    with open(path, "rb") as f:
        line = f.readline().decode("ascii", errors="ignore").strip()
        if line != "ply":
            raise RuntimeError(f"{path} is not a PLY file and plyfile is not installed.")
        ply_format = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            line = line.decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                ply_format = line.split()[1]
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                parts = line.split()
                if parts[1] == "list":
                    raise RuntimeError(f"List properties in vertex element are not supported without plyfile: {path}")
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        prop_names = [name for _, name in properties]
        if vertex_count is None or "x" not in prop_names or "y" not in prop_names or "z" not in prop_names or "label" not in prop_names:
            raise RuntimeError(f"PLY must contain vertex x/y/z/label properties: {path}")

        if ply_format == "ascii":
            name_to_idx = {name: idx for idx, (_, name) in enumerate(properties)}
            x_idx, y_idx, z_idx, label_idx = [name_to_idx[name] for name in ("x", "y", "z", "label")]
            points = np.empty((vertex_count, 3), dtype=np.float32)
            labels = np.empty((vertex_count,), dtype=np.int64)
            for idx in range(vertex_count):
                vals = f.readline().decode("ascii", errors="ignore").split()
                points[idx] = [float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])]
                labels[idx] = int(float(vals[label_idx]))
            return points, labels

        if ply_format not in ("binary_little_endian", "binary_big_endian"):
            raise RuntimeError(f"Unsupported PLY format {ply_format!r}: {path}")

        endian = "<" if ply_format == "binary_little_endian" else ">"
        dtype_map = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "i2",
            "int16": "i2",
            "ushort": "u2",
            "uint16": "u2",
            "int": "i4",
            "int32": "i4",
            "uint": "u4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }
        np_dtype = []
        for prop_type, name in properties:
            if prop_type not in dtype_map:
                raise RuntimeError(f"Unsupported PLY property type {prop_type!r}: {path}")
            code = dtype_map[prop_type]
            np_dtype.append((name, code if code.endswith("1") else endian + code))
        dtype = np.dtype(np_dtype)
        raw = f.read(dtype.itemsize * vertex_count)
        if len(raw) < dtype.itemsize * vertex_count:
            raise RuntimeError(f"Unexpected EOF in PLY vertex data: {path}")
        vertex = np.frombuffer(raw, dtype=dtype, count=vertex_count)
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
        labels = np.asarray(vertex["label"]).astype(np.int64)
    return points, labels


def find_label_ply(scene_data_dir, scene_name):
    candidates = [
        scene_data_dir / f"{scene_name}_vh_clean_2.labels.ply",
        scene_data_dir / f"{scene_name}_vh_clean.labels.ply",
    ]
    candidates.extend(sorted(scene_data_dir.glob("*labels.ply")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ScanNet labels ply found in {scene_data_dir}")


def get_scene_names(root):
    return sorted([p.name for p in Path(root).iterdir() if p.is_dir()])


def render_dir(scene_root, split="test", iteration=30000):
    path = Path(scene_root) / split / f"ours_{iteration}" / "renders"
    if not path.exists():
        raise FileNotFoundError(f"Missing render dir: {path}")
    return path


def build_camera_view(camera_json, out_width, out_height):
    src_width = int(camera_json["width"])
    src_height = int(camera_json["height"])
    fx = float(camera_json["fx"]) * (float(out_width) / float(src_width))
    fy = float(camera_json["fy"]) * (float(out_height) / float(src_height))
    cx = float(out_width) * 0.5
    cy = float(out_height) * 0.5

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(camera_json["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(camera_json["position"], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return w2c, fx, fy, cx, cy


def project_labeled_points(points, labels, camera_json, out_width, out_height, depth_epsilon=0.05):
    w2c, fx, fy, cx, cy = build_camera_view(camera_json, out_width, out_height)
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]

    points_cam = points @ rot.T + trans[None, :]
    z = points_cam[:, 2]
    valid = z > 0.05
    if not np.any(valid):
        return np.zeros((out_height, out_width), dtype=np.int32)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    u = np.rint((x * fx / z) + cx).astype(np.int32)
    v = np.rint((y * fy / z) + cy).astype(np.int32)
    valid &= (u >= 0) & (u < out_width) & (v >= 0) & (v < out_height) & (labels > 0)

    if not np.any(valid):
        return np.zeros((out_height, out_width), dtype=np.int32)

    u = u[valid]
    v = v[valid]
    z = z[valid]
    lbl = labels[valid]
    flat = v * out_width + u

    order = np.lexsort((z, flat))
    flat = flat[order]
    z = z[order]
    lbl = lbl[order]
    unique_flat, first_idx = np.unique(flat, return_index=True)
    min_z = z[first_idx]

    keep = z <= (np.repeat(min_z, np.diff(np.r_[first_idx, len(z)])) + depth_epsilon)
    flat_kept = flat[keep]
    lbl_kept = lbl[keep]

    gt = np.zeros(out_height * out_width, dtype=np.int32)
    if flat_kept.size == 0:
        return gt.reshape(out_height, out_width)

    packed = np.stack([flat_kept, lbl_kept], axis=1)
    order = np.lexsort((packed[:, 1], packed[:, 0]))
    packed = packed[order]
    start = 0
    while start < packed.shape[0]:
        end = start + 1
        while end < packed.shape[0] and packed[end, 0] == packed[start, 0]:
            end += 1
        vals, counts = np.unique(packed[start:end, 1], return_counts=True)
        gt[packed[start, 0]] = vals[np.argmax(counts)]
        start = end
    return gt.reshape(out_height, out_width)


def remap_to_target(label_map, target_ids):
    remap = np.zeros_like(label_map, dtype=np.int64)
    for new_id, old_id in enumerate(target_ids, start=1):
        remap[label_map == old_id] = new_id
    return remap


class MetricAccumulator:
    def __init__(self, num_classes):
        self.num_classes = int(num_classes)
        self.intersection = np.zeros(self.num_classes + 1, dtype=np.float64)
        self.union = np.zeros(self.num_classes + 1, dtype=np.float64)
        self.gt_total = np.zeros(self.num_classes + 1, dtype=np.float64)
        self.correct = np.zeros(self.num_classes + 1, dtype=np.float64)
        self.valid_pixels = 0

    def update(self, gt, pred):
        valid = gt > 0
        if not np.any(valid):
            return
        gt = gt[valid]
        pred = pred[valid].copy()
        pred[(pred < 1) | (pred > self.num_classes)] = 0
        self.valid_pixels += int(gt.size)
        for cls in range(1, self.num_classes + 1):
            gt_cls = gt == cls
            pred_cls = pred == cls
            inter = np.logical_and(gt_cls, pred_cls).sum()
            union = np.logical_or(gt_cls, pred_cls).sum()
            self.intersection[cls] += inter
            self.union[cls] += union
            self.gt_total[cls] += gt_cls.sum()
            self.correct[cls] += inter

    def summary(self):
        present = self.gt_total[1:] > 0
        if not np.any(present):
            return {"mIoU": float("nan"), "mAcc": float("nan"), "pixel_acc": float("nan"), "valid_pixels": 0}
        iou = np.divide(
            self.intersection[1:],
            np.maximum(self.union[1:], 1.0),
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.union[1:] > 0,
        )
        acc = np.divide(
            self.correct[1:],
            np.maximum(self.gt_total[1:], 1.0),
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.gt_total[1:] > 0,
        )
        pixel_acc = float(self.correct[1:].sum() / max(self.gt_total[1:].sum(), 1.0))
        return {
            "mIoU": float(iou[present].mean()),
            "mAcc": float(acc[present].mean()),
            "pixel_acc": pixel_acc,
            "valid_pixels": int(self.valid_pixels),
        }


class SAMCLIPPostSegmentor:
    def __init__(
        self,
        sam_model_dir,
        clip_model_path,
        target_ids_by_name,
        prompt_template,
        device,
        points_per_side=24,
        sam_iou_thresh=0.88,
        area_thresh=100,
        batch_size=32,
        target_long_side=1024,
    ):
        self.device = torch.device(device)
        self.points_per_side = int(points_per_side)
        self.sam_iou_thresh = float(sam_iou_thresh)
        self.area_thresh = int(area_thresh)
        self.batch_size = int(batch_size)
        self.target_long_side = int(target_long_side)

        use_sam_hq = False
        config_path = Path(sam_model_dir) / "config.json"
        if config_path.exists():
            try:
                use_sam_hq = load_json(config_path).get("model_type") == "sam_hq"
            except Exception:
                use_sam_hq = False
        if "sam-hq" in str(sam_model_dir).lower() or "sam_hq" in str(sam_model_dir).lower():
            use_sam_hq = True

        if use_sam_hq:
            self.sam_processor = SamHQProcessor.from_pretrained(sam_model_dir)
            self.sam_model = SamHQModel.from_pretrained(sam_model_dir).to(self.device).eval()
        else:
            self.sam_processor = SamProcessor.from_pretrained(sam_model_dir)
            self.sam_model = SamModel.from_pretrained(sam_model_dir).to(self.device).eval()

        self.clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=os.path.isdir(clip_model_path)).to(self.device).eval()
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_path, local_files_only=os.path.isdir(clip_model_path))
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=os.path.isdir(clip_model_path))

        self.text_features = {}
        for target_name, ids in target_ids_by_name.items():
            class_names = [NYU40_NAMES[i] for i in ids]
            prompts = [prompt_template.format(name=name) for name in class_names]
            inputs = self.clip_tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
            self.text_features[target_name] = F.normalize(feats, dim=1, p=2)

    @torch.no_grad()
    def _sam_masks(self, image_rgb):
        h, w, _ = image_rgb.shape
        scale = self.target_long_side / float(max(h, w))
        if scale < 1.0:
            input_image = cv2.resize(image_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            input_image = image_rgb

        inputs = self.sam_processor(input_image, return_tensors="pt").to(self.device)
        if hasattr(self.sam_model, "get_image_embeddings"):
            image_embeddings = self.sam_model.get_image_embeddings(inputs.pixel_values)
            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[0]
        else:
            image_embeddings = self.sam_model.vision_encoder(inputs.pixel_values).last_hidden_state

        xs = np.linspace(0, input_image.shape[1] - 1, self.points_per_side, dtype=np.float32)
        ys = np.linspace(0, input_image.shape[0] - 1, self.points_per_side, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        points = torch.from_numpy(np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)).to(self.device)
        points = points.float().reshape(-1, 1, 1, 2)

        final_id_map = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        best_score_map = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        mask_scores = {}

        for start in range(0, points.shape[0], self.batch_size):
            b_points = points[start : start + self.batch_size]
            curr_bs = b_points.shape[0]
            outputs = self.sam_model(
                image_embeddings=image_embeddings.repeat(curr_bs, 1, 1, 1),
                input_points=b_points,
                input_labels=torch.ones((curr_bs, 1, 1), dtype=torch.long, device=self.device),
            )
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks,
                [[h, w]] * curr_bs,
                [[input_image.shape[0], input_image.shape[1]]] * curr_bs,
            )
            masks = torch.stack(masks).squeeze(1)
            scores = outputs.iou_scores.squeeze(1)
            best_idx = torch.argmax(scores, dim=1)
            rows = torch.arange(curr_bs, device=self.device)
            masks = masks[rows, best_idx]
            scores = scores[rows, best_idx]

            for j in range(curr_bs):
                score = float(scores[j].item())
                if score <= self.sam_iou_thresh:
                    continue
                mask = masks[j] > 0
                if int(mask.sum().item()) <= self.area_thresh:
                    continue
                mask_id = start + j + 1
                update = mask & (scores[j] > best_score_map)
                final_id_map[update] = mask_id
                best_score_map[update] = scores[j]
                mask_scores[mask_id] = score

            del outputs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return final_id_map.cpu().numpy().astype(np.int32), mask_scores

    @torch.no_grad()
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image)
        id_map, mask_scores = self._sam_masks(image_rgb)
        unique_ids = [int(x) for x in np.unique(id_map) if x > 0]

        if not unique_ids:
            h, w = id_map.shape
            return {name: np.zeros((h, w), dtype=np.int64) for name in TARGET_IDS}

        mask_features = {}
        for mask_id in unique_ids:
            mask = id_map == mask_id
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            crop = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
            if crop.shape[0] <= 10 or crop.shape[1] <= 10:
                continue
            masked_crop = np.zeros_like(crop)
            masked_crop[crop_mask] = crop[crop_mask]
            pil = Image.fromarray(masked_crop).convert("RGB")
            clip_inputs = self.clip_image_processor(images=pil, return_tensors="pt").to(self.device)
            feat = self.clip_model.get_image_features(**clip_inputs)
            mask_features[mask_id] = F.normalize(feat, dim=1, p=2).squeeze(0)

        predictions = {}
        h, w = id_map.shape
        for target_name, text_features in self.text_features.items():
            pred = np.zeros((h, w), dtype=np.int64)
            if not mask_features:
                predictions[target_name] = pred
                continue
            for mask_id, feat in mask_features.items():
                scores = feat.unsqueeze(0) @ text_features.t()
                label = int(torch.argmax(scores, dim=1).item()) + 1
                pred[id_map == mask_id] = label
            predictions[target_name] = pred
        return predictions


def colorize_labels(label_map):
    labels = np.clip(label_map.astype(np.int64), 0, len(PALETTE) - 1)
    return PALETTE[labels]


def save_overlay(image_path, pred, gt, out_path, alpha=0.45):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    pred_color = colorize_labels(pred)
    gt_color = colorize_labels(gt)
    pred_overlay = (image * (1 - alpha) + pred_color * alpha).astype(np.uint8)
    gt_overlay = (image * (1 - alpha) + gt_color * alpha).astype(np.uint8)
    canvas = np.concatenate([image, pred_overlay, gt_overlay], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def load_or_predict(segmentor, render_path, cache_dir, method, scene, view_stem, targets, reuse_preds):
    target_cache = {
        target: cache_dir / method / scene / f"{view_stem}_{target}.npy"
        for target in targets
    }
    if reuse_preds and all(path.exists() for path in target_cache.values()):
        return {target: np.load(path) for target, path in target_cache.items()}

    pred_by_target = segmentor.predict(render_path)
    for target, path in target_cache.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pred_by_target[target].astype(np.int16))
    return pred_by_target


def format_percent(value):
    if value != value:
        return "nan"
    return f"{value * 100.0:.2f}"


def build_markdown(results):
    lines = []
    lines.append("# ScanNet Full Test-View SAM+CLIP Post-Segmentation")
    lines.append("")
    lines.append("| Method | Target | mIoU | mAcc | Pixel Acc | Valid Pixels |")
    lines.append("| ------ | -----: | ---: | ---: | --------: | -----------: |")
    for method in ["baseline", "ours"]:
        for target in ["19", "15", "10"]:
            metrics = results["summary"][method][target]
            lines.append(
                f"| {method} | {target} | {format_percent(metrics['mIoU'])} | "
                f"{format_percent(metrics['mAcc'])} | {format_percent(metrics['pixel_acc'])} | "
                f"{metrics['valid_pixels']} |"
            )
    lines.append("")
    lines.append("| Delta | Target | mIoU | mAcc | Pixel Acc |")
    lines.append("| ----- | -----: | ---: | ---: | --------: |")
    for target in ["19", "15", "10"]:
        base = results["summary"]["baseline"][target]
        ours = results["summary"]["ours"][target]
        lines.append(
            f"| ours - baseline | {target} | "
            f"{(ours['mIoU'] - base['mIoU']) * 100.0:+.2f} | "
            f"{(ours['mAcc'] - base['mAcc']) * 100.0:+.2f} | "
            f"{(ours['pixel_acc'] - base['pixel_acc']) * 100.0:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "baseline": Path(args.baseline_root),
        "ours": Path(args.ours_root),
    }
    scenes = args.scenes or sorted(set(get_scene_names(args.baseline_root)) & set(get_scene_names(args.ours_root)))
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise RuntimeError("No common scenes found.")

    target_ids = {k: TARGET_IDS[k] for k in args.targets}
    segmentor = SAMCLIPPostSegmentor(
        args.sam_model_dir,
        args.clip_model_path,
        target_ids,
        args.prompt_template,
        args.device,
        points_per_side=args.points_per_side,
        sam_iou_thresh=args.sam_iou_thresh,
        area_thresh=args.area_thresh,
        batch_size=args.sam_batch_size,
        target_long_side=args.sam_long_side,
    )

    accumulators = {
        method: {target: MetricAccumulator(len(ids)) for target, ids in target_ids.items()}
        for method in methods
    }
    per_scene = defaultdict(lambda: defaultdict(dict))
    per_view = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    gt_cache = {}

    for scene in tqdm(scenes, desc="Scenes"):
        scene_data_dir = Path(args.data_root) / scene
        scene_camera_json = load_json(methods["baseline"] / scene / "cameras.json")
        label_ply = find_label_ply(scene_data_dir, scene)
        points, labels = read_labels_from_ply(label_ply)

        scene_accumulators = {
            method: {target: MetricAccumulator(len(ids)) for target, ids in target_ids.items()}
            for method in methods
        }
        render_files = sorted(render_dir(methods["baseline"] / scene, args.split, args.iteration).glob("*.png"))
        if args.max_views > 0:
            render_files = render_files[: args.max_views]

        for render_path_base in tqdm(render_files, desc=f"{scene} views", leave=False):
            view_idx = int(render_path_base.stem)
            if view_idx >= len(scene_camera_json):
                raise IndexError(f"{render_path_base.name} maps to cameras.json[{view_idx}], but only {len(scene_camera_json)} cameras exist.")
            camera = scene_camera_json[view_idx]

            sample_image = Image.open(render_path_base)
            width, height = sample_image.size
            gt_full = gt_cache.get((scene, view_idx, width, height))
            if gt_full is None:
                gt_full = project_labeled_points(points, labels, camera, width, height, args.gt_depth_epsilon)
                gt_cache[(scene, view_idx, width, height)] = gt_full

            gt_by_target = {target: remap_to_target(gt_full, ids) for target, ids in target_ids.items()}
            for method, root in methods.items():
                render_path = root / scene / args.split / f"ours_{args.iteration}" / "renders" / render_path_base.name
                if not render_path.exists():
                    raise FileNotFoundError(f"Missing matching render: {render_path}")
                pred_by_target = load_or_predict(
                    segmentor,
                    render_path,
                    output_dir / "pred_cache",
                    method,
                    scene,
                    render_path.stem,
                    target_ids.keys(),
                    args.reuse_preds,
                )

                for target in target_ids:
                    gt = gt_by_target[target]
                    pred = pred_by_target[target]
                    accumulators[method][target].update(gt, pred)
                    scene_accumulators[method][target].update(gt, pred)

                    if args.save_overlays and target == args.overlay_target:
                        overlay_dir = output_dir / "overlays" / method / scene
                        if args.max_overlays_per_scene < 0 or len(list(overlay_dir.glob("*.png"))) < args.max_overlays_per_scene:
                            save_overlay(render_path, pred, gt, overlay_dir / render_path.name)

                    view_acc = MetricAccumulator(len(target_ids[target]))
                    view_acc.update(gt, pred)
                    per_view[scene][method][render_path.name][target] = view_acc.summary()

        for method in methods:
            for target in target_ids:
                per_scene[scene][method][target] = scene_accumulators[method][target].summary()

    results = {
        "config": {
            "baseline_root": str(Path(args.baseline_root).resolve()),
            "ours_root": str(Path(args.ours_root).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "scenes": scenes,
            "split": args.split,
            "iteration": args.iteration,
            "targets": args.targets,
            "sam_model_dir": args.sam_model_dir,
            "clip_model_path": args.clip_model_path,
            "points_per_side": args.points_per_side,
            "sam_iou_thresh": args.sam_iou_thresh,
            "area_thresh": args.area_thresh,
            "reuse_preds": args.reuse_preds,
        },
        "summary": {
            method: {target: accumulators[method][target].summary() for target in target_ids}
            for method in methods
        },
        "per_scene": per_scene,
        "per_view": per_view,
    }
    with open(output_dir / "semantic_postseg_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    markdown = build_markdown(results)
    with open(output_dir / "semantic_postseg_results.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM+CLIP post-segmentation on rendered ScanNet test views.")
    parser.add_argument("--baseline_root", default=r"D:\program\seg_scafflod_gs\output\scannet10_full_scaffoldgs")
    parser.add_argument("--ours_root", default=r"D:\program\seg_scafflod_gs\output\scannet10_full_ours_expert8_b10")
    parser.add_argument("--data_root", default=r"D:\program\seg_scafflod_gs\data\InstanceGS_data\scannet_wym")
    parser.add_argument("--output_dir", default=r"D:\program\seg_scafflod_gs\output\scannet10_full_semantic_postseg")
    parser.add_argument("--sam_model_dir", default="./weights/sam_hq_vit_base")
    parser.add_argument("--clip_model_path", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="test")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--targets", nargs="+", default=["19", "15", "10"], choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--points_per_side", type=int, default=24)
    parser.add_argument("--sam_batch_size", type=int, default=32)
    parser.add_argument("--sam_long_side", type=int, default=1024)
    parser.add_argument("--sam_iou_thresh", type=float, default=0.88)
    parser.add_argument("--area_thresh", type=int, default=100)
    parser.add_argument("--gt_depth_epsilon", type=float, default=0.05)
    parser.add_argument("--prompt_template", default="a photo of a {name} in a room")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--reuse_preds", action="store_true")
    parser.add_argument("--overlay_target", default="19", choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--max_overlays_per_scene", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
