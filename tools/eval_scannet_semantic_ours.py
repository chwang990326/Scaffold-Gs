import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPModel, CLIPTokenizer


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


def read_labels_from_ply(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    labels = np.asarray(vertex["label"]).astype(np.int64)
    return points, labels


def load_json_text_features(text_features_path, target_names, device):
    with open(text_features_path, "r") as f:
        loaded = json.load(f)

    missing = [name for name in target_names if name not in loaded]
    if missing:
        raise KeyError(
            f"Missing text features for {missing} in {text_features_path}. "
            "Use the same assets/text_features.json protocol as InstanceGaussian."
        )

    feats = np.stack([np.asarray(loaded[name], dtype=np.float32) for name in target_names], axis=0)
    feats = torch.from_numpy(feats).to(device)
    return F.normalize(feats, dim=1, p=2)


def load_hf_clip_text_model(clip_model_path, device):
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def build_hf_clip_text_features(tokenizer, model, target_names, prompt_template, device):
    prompts = [prompt_template.format(name=name) for name in target_names]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    feats = model.get_text_features(**inputs)
    return F.normalize(feats, dim=1, p=2)


def calculate_metrics(gt, pred, total_classes, device):
    gt = gt.to(device)
    pred = pred.to(device)
    pred = pred.clone()
    pred[gt == 0] = 0

    ious = torch.zeros(total_classes, device=device)
    correct = torch.zeros(total_classes, device=device)
    total = torch.zeros(total_classes, device=device)
    union = torch.zeros(total_classes, device=device)

    for cls in range(1, total_classes):
        gt_cls = gt == cls
        pred_cls = pred == cls
        correct[cls] = torch.sum(gt_cls & pred_cls)
        total[cls] = torch.sum(gt_cls)
        union[cls] = torch.sum(gt_cls | pred_cls)
        if union[cls] > 0:
            ious[cls] = correct[cls] / union[cls]

    valid_gt_classes = torch.unique(gt)
    valid_gt_classes = valid_gt_classes[valid_gt_classes != 0]
    if valid_gt_classes.numel() == 0:
        return float("nan"), float("nan"), float("nan")

    class_accuracy = correct / total.clamp_min(1.0)
    mean_iou = ious[valid_gt_classes].mean().item()
    mean_acc = class_accuracy[valid_gt_classes].mean().item()
    valid_mask = gt != 0
    overall_acc = torch.sum((gt == pred) & valid_mask).float() / valid_mask.sum().clamp_min(1)
    return mean_iou, mean_acc, overall_acc.item()


def predict_vertex_labels(anchor_xyz, anchor_features, vertex_xyz, text_features, k, chunk_size, score_chunk_size, device):
    anchor_xyz = anchor_xyz.to(dtype=torch.float32)
    anchor_features = F.normalize(anchor_features.to(device=device, dtype=torch.float32), dim=1, p=2)
    text_features = text_features.to(device=device, dtype=torch.float32)

    valid_anchor_mask = (anchor_features.abs().sum(dim=1) > 0).cpu()
    anchor_xyz = anchor_xyz[valid_anchor_mask]
    anchor_features = anchor_features[valid_anchor_mask.to(device)]
    if anchor_xyz.numel() == 0:
        raise RuntimeError("No valid anchor semantic features found.")
    if anchor_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"Feature dimension mismatch: anchor={anchor_features.shape[1]}, text={text_features.shape[1]}. "
            "For ScanNet mIoU/mAcc, re-run seg_train after the raw512 patch so "
            "anchor_semantic_features_raw512.pt exists."
        )

    labels = []
    confs = []
    for start in range(0, anchor_features.shape[0], score_chunk_size):
        scores = anchor_features[start:start + score_chunk_size] @ text_features.t()
        conf, label = torch.max(scores, dim=1)
        labels.append((label.cpu().numpy() + 1).astype(np.int64))
        confs.append(conf.cpu().numpy().astype(np.float32))
    anchor_label = np.concatenate(labels, axis=0)
    anchor_conf = np.concatenate(confs, axis=0)

    anchor_xyz_np = anchor_xyz.cpu().numpy().astype(np.float32)
    vertex_xyz = vertex_xyz.astype(np.float32)
    k = max(1, min(int(k), anchor_xyz.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(anchor_xyz_np)

    predictions = []
    num_classes = text_features.shape[0]
    for start in range(0, vertex_xyz.shape[0], chunk_size):
        query = vertex_xyz[start:start + chunk_size]
        nn_dist, nn_idx = nn.kneighbors(query, return_distance=True)
        nn_labels = anchor_label[nn_idx]
        nn_weights = anchor_conf[nn_idx] / np.maximum(nn_dist, 1e-4)

        scores = np.zeros((query.shape[0], num_classes + 1), dtype=np.float32)
        row_idx = np.repeat(np.arange(query.shape[0])[:, None], k, axis=1)
        np.add.at(scores, (row_idx.reshape(-1), nn_labels.reshape(-1)), nn_weights.reshape(-1))
        predictions.append(torch.from_numpy(np.argmax(scores[:, 1:], axis=1).astype(np.int64) + 1))

    return torch.cat(predictions, dim=0).long()


def evaluate_scene(args, scene):
    device = torch.device(args.device)
    model_dir = Path(args.model_root) / scene
    data_dir = Path(args.data_root) / scene

    anchor_path = model_dir / "anchor_positions_for_semantic.pt"
    feature_path = model_dir / "anchor_semantic_features_raw512.pt"
    if not feature_path.exists():
        feature_path = model_dir / "anchor_semantic_features.pt"

    if not anchor_path.exists():
        ply_candidates = sorted((model_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
        raise FileNotFoundError(
            f"Missing {anchor_path}. Re-run seg_train after the raw512 patch. "
            f"Found trained point clouds: {[str(p) for p in ply_candidates[-3:]]}"
        )
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing semantic features under {model_dir}")

    gt_ply = data_dir / f"{scene}_vh_clean_2.labels.ply"
    vertex_xyz, labels = read_labels_from_ply(gt_ply)
    anchor_xyz = torch.load(anchor_path, map_location="cpu")
    anchor_features = torch.load(feature_path, map_location="cpu")
    hf_tokenizer = None
    hf_text_model = None
    if args.text_feature_source == "hf_clip":
        hf_tokenizer, hf_text_model = load_hf_clip_text_model(args.clip_model_path, device)

    scene_results = {}
    for target_name, target_ids in TARGET_IDS.items():
        target_names = [NYU40_NAMES[idx] for idx in target_ids]
        if args.text_feature_source == "hf_clip":
            text_features = build_hf_clip_text_features(
                hf_tokenizer,
                hf_text_model,
                target_names,
                args.prompt_template,
                device,
            )
        else:
            text_features = load_json_text_features(args.text_features, target_names, device)

        remapped = np.zeros_like(labels, dtype=np.int64)
        for new_id, original_id in enumerate(target_ids, start=1):
            remapped[labels == original_id] = new_id

        pred = predict_vertex_labels(
            anchor_xyz,
            anchor_features,
            vertex_xyz,
            text_features,
            args.knn,
            args.chunk_size,
            args.score_chunk_size,
            device,
        )
        gt = torch.from_numpy(remapped.astype(np.int64))
        miou, macc, acc = calculate_metrics(gt, pred, total_classes=len(target_ids) + 1, device=device)
        scene_results[target_name] = {
            "mIoU": miou * 100.0,
            "mAcc": macc * 100.0,
            "overallAcc": acc * 100.0,
        }

    return scene_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ours anchor CLIP features on ScanNet semantic mIoU/mAcc.")
    parser.add_argument("--data_root", default="/home/chwang/data/InstanceGS_data/scannet_wym")
    parser.add_argument("--model_root", required=True)
    parser.add_argument(
        "--text_feature_source",
        choices=["hf_clip", "json"],
        default="hf_clip",
        help="Use hf_clip for Ours features; use json only for matching InstanceGaussian assets.",
    )
    parser.add_argument("--clip_model_path", default="./weights/clip-vit-base-patch32")
    parser.add_argument("--prompt_template", default="a photo of a {name}")
    parser.add_argument("--text_features", default="/home/chwang/InstanceGaussian/assets/text_features.json")
    parser.add_argument("--scenes", nargs="+", default=["scene0000_00"])
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--score_chunk_size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = {}
    metadata = {
        "text_feature_source": args.text_feature_source,
        "clip_model_path": args.clip_model_path if args.text_feature_source == "hf_clip" else None,
        "prompt_template": args.prompt_template if args.text_feature_source == "hf_clip" else None,
        "text_features": args.text_features if args.text_feature_source == "json" else None,
    }

    for scene in args.scenes:
        result = evaluate_scene(args, scene)
        all_results[scene] = result
        parts = []
        for target in ["19", "15", "10"]:
            metrics = result[target]
            parts.append(f"{target}cls mIoU {metrics['mIoU']:.2f} mAcc {metrics['mAcc']:.2f}")
        print(f"{scene}: " + " | ".join(parts))

    mean_results = {}
    for target in ["19", "15", "10"]:
        mean_results[target] = {
            key: float(np.mean([all_results[scene][target][key] for scene in all_results]))
            for key in ["mIoU", "mAcc", "overallAcc"]
        }
    all_results["mean"] = mean_results
    all_results["metadata"] = metadata
    print("mean:", json.dumps(mean_results, indent=2))

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
