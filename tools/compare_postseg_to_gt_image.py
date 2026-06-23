import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_scannet_postseg import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_IDS,
    ScannetSemSegPostSegmenter,
    add_title,
    colorize_label_map,
    format_float,
    make_overlay,
    metrics_from_confusion,
    remap_pred19_to_target,
    update_confusion,
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path(path, repo_root=REPO_ROOT):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def find_gt_image(metadata, scene, render_name, img_name):
    method_name = metadata["method_name"]
    candidates = [
        Path(metadata["baseline_root"]) / scene / "test" / method_name / "gt" / render_name,
        Path(metadata["ours_root"]) / scene / "test" / method_name / "gt" / render_name,
        Path(metadata["data_root"]) / scene / "color" / f"{img_name}.jpg",
        Path(metadata["data_root"]) / scene / "color" / f"{img_name}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No GT RGB image found for {scene} {render_name} img_name={img_name}")


def render_path(metadata, method, scene, render_name):
    root = Path(metadata[f"{method}_root"])
    return root / scene / "test" / metadata["method_name"] / "renders" / render_name


def pred_cache_path(eval_root, method, scene, render_name):
    return Path(eval_root) / "scannet_semseg_cache" / method / scene / f"{Path(render_name).stem}.npy"


def load_render_pred(eval_root, method, scene, render_name):
    path = pred_cache_path(eval_root, method, scene, render_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached {method} prediction: {path}")
    return np.load(path)


def resize_like(label_map, shape_hw):
    if tuple(label_map.shape) == tuple(shape_hw):
        return label_map
    return cv2.resize(
        label_map.astype(np.int32),
        (int(shape_hw[1]), int(shape_hw[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


def confusion_metrics(ref_pred19, render_pred19, target_key):
    target_ids = TARGET_IDS[target_key]
    ref = remap_pred19_to_target(ref_pred19, target_ids)
    pred = remap_pred19_to_target(render_pred19, target_ids)
    pred = resize_like(pred, ref.shape)
    confusion = np.zeros((len(target_ids) + 1, len(target_ids) + 1), dtype=np.int64)
    update_confusion(confusion, ref, pred, len(target_ids))
    return confusion, metrics_from_confusion(confusion)


def make_panel(gt_image_path, baseline_image_path, ours_image_path, gt_pred, baseline_pred, ours_pred, target_key):
    gt_rgb = np.asarray(Image.open(gt_image_path).convert("RGB"))
    baseline_rgb = np.asarray(Image.open(baseline_image_path).convert("RGB"))
    ours_rgb = np.asarray(Image.open(ours_image_path).convert("RGB"))

    target_ids = TARGET_IDS[target_key]
    gt_target = remap_pred19_to_target(gt_pred, target_ids)
    baseline_target = resize_like(remap_pred19_to_target(baseline_pred, target_ids), gt_target.shape)
    ours_target = resize_like(remap_pred19_to_target(ours_pred, target_ids), gt_target.shape)

    baseline_rgb = cv2.resize(baseline_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    ours_rgb = cv2.resize(ours_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    panels = [
        add_title(gt_rgb, "GT RGB"),
        add_title(colorize_label_map(gt_target), f"GT pred {target_key}"),
        add_title(make_overlay(baseline_rgb, baseline_target), "baseline pred"),
        add_title(make_overlay(ours_rgb, ours_target), "ours pred"),
    ]
    return np.concatenate(panels, axis=1)


def format_signed(value):
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):+.2f}"


def write_markdown(path, summary, top_rows, largest, args):
    lines = [
        "# GT-RGB-Pred Reference Comparison",
        "",
        "Reference is the ScanNet supervised 2D model prediction on the real GT RGB test image.",
        "",
        "| Method | Target | mIoU | mAcc | Pixel Acc | Valid Pixels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["baseline", "ours"]:
        for target_key in args.targets:
            metrics = summary[method][target_key]
            lines.append(
                f"| {method} | {target_key} | {format_float(metrics['mIoU'])} | "
                f"{format_float(metrics['mAcc'])} | {format_float(metrics['pixelAcc'])} | "
                f"{metrics['validPixels']} |"
            )

    lines.extend(["", "| Delta | Target | mIoU | mAcc | Pixel Acc |", "|---|---:|---:|---:|---:|"])
    for target_key in args.targets:
        delta = summary["delta_ours_minus_baseline"][target_key]
        lines.append(
            f"| ours - baseline | {target_key} | {format_signed(delta['mIoU'])} | "
            f"{format_signed(delta['mAcc'])} | {format_signed(delta['pixelAcc'])} |"
        )

    lines.extend(
        [
            "",
            f"## Largest Difference ({args.rank_target}-class mIoU)",
            "",
            f"- Scene: `{largest['scene']}`",
            f"- Render: `{largest['render_name']}`",
            f"- ScanNet image id: `{largest['img_name']}`",
            f"- Baseline mIoU: {format_float(largest['baseline_metrics']['mIoU'])}",
            f"- Ours mIoU: {format_float(largest['ours_metrics']['mIoU'])}",
            f"- Delta: {format_signed(largest['delta_mIoU'])}",
            f"- Abs delta: {format_float(abs(largest['delta_mIoU']))}",
            f"- Panel: `{largest['panel_path']}`",
            f"- GT RGB: `{largest['gt_image_path']}`",
            f"- Baseline render: `{largest['baseline_render_path']}`",
            f"- Ours render: `{largest['ours_render_path']}`",
            "",
            f"## Top {len(top_rows)} Absolute Differences",
            "",
            "| Rank | Scene | Render | Img | Baseline mIoU | Ours mIoU | Delta | Abs Delta |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            f"| {idx} | {row['scene']} | {row['render_name']} | {row['img_name']} | "
            f"{format_float(row['baseline_metrics']['mIoU'])} | "
            f"{format_float(row['ours_metrics']['mIoU'])} | "
            f"{format_signed(row['delta_mIoU'])} | {format_float(abs(row['delta_mIoU']))} |"
        )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline/ours semantic predictions against the 2D model prediction on GT RGB images."
    )
    parser.add_argument(
        "--eval_root",
        default="output/semantic_eval/scannet10_full_postseg_scannet_semseg_fulltrain_resume_3ep",
    )
    parser.add_argument("--results_json", default=None)
    parser.add_argument(
        "--output_root",
        default="output/semantic_eval/scannet10_full_gtimage_ref_scannet_semseg_fulltrain_resume_3ep",
    )
    parser.add_argument(
        "--semseg_checkpoint",
        default="output/scannet_semseg2d/scannet10_deeplabv3_resnet50_19cls_fulltrain_resume_3ep.pt",
    )
    parser.add_argument("--semseg_input_width", type=int, default=384)
    parser.add_argument("--semseg_input_height", type=int, default=288)
    parser.add_argument("--targets", nargs="+", default=["19", "15", "10"], choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--rank_target", default="19", choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refresh_gt_pred", action="store_true")
    args = parser.parse_args()

    eval_root = resolve_path(args.eval_root)
    output_root = resolve_path(args.output_root)
    results_json = resolve_path(args.results_json) if args.results_json else eval_root / "semantic_postseg_results.json"
    prior_results = load_json(results_json)
    metadata = prior_results["metadata"]
    metadata["semseg_checkpoint"] = str(resolve_path(args.semseg_checkpoint))

    output_root.mkdir(parents=True, exist_ok=True)
    segmenter_args = SimpleNamespace(
        semseg_checkpoint=str(resolve_path(args.semseg_checkpoint)),
        semseg_input_width=args.semseg_input_width,
        semseg_input_height=args.semseg_input_height,
        output_root=str(output_root),
        device=args.device,
        refresh_postseg=args.refresh_gt_pred,
    )
    segmenter = ScannetSemSegPostSegmenter(segmenter_args)

    summary_confusions = {
        method: {
            target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
            for target_key in args.targets
        }
        for method in ["baseline", "ours"]
    }
    view_rows = []
    results = {
        "metadata": {
            "eval_root": str(eval_root),
            "source_results_json": str(results_json),
            "output_root": str(output_root),
            "semseg_checkpoint": str(resolve_path(args.semseg_checkpoint)),
            "semseg_input_width": args.semseg_input_width,
            "semseg_input_height": args.semseg_input_height,
            "device": args.device,
            "reference": "GT RGB image prediction from the same scannet_semseg model",
            "targets": args.targets,
            "rank_target": args.rank_target,
            "baseline_root": metadata["baseline_root"],
            "ours_root": metadata["ours_root"],
            "data_root": metadata["data_root"],
            "method_name": metadata["method_name"],
        },
        "views": [],
    }

    for scene, scene_result in tqdm(prior_results["scenes"].items(), desc="Scenes"):
        for view in tqdm(scene_result["view_mapping"], desc=scene, leave=False):
            render_name = view["render_name"]
            img_name = str(view["img_name"])
            gt_image = find_gt_image(metadata, scene, render_name, img_name)
            baseline_image = render_path(metadata, "baseline", scene, render_name)
            ours_image = render_path(metadata, "ours", scene, render_name)

            gt_pred19, _, _ = segmenter.segment("gt_rgb", scene, render_name, gt_image)
            baseline_pred19 = load_render_pred(eval_root, "baseline", scene, render_name)
            ours_pred19 = load_render_pred(eval_root, "ours", scene, render_name)

            row = {
                "scene": scene,
                "render_name": render_name,
                "img_name": img_name,
                "gt_image_path": str(gt_image),
                "baseline_render_path": str(baseline_image),
                "ours_render_path": str(ours_image),
                "targets": {},
            }
            for target_key in args.targets:
                for method, pred19 in [("baseline", baseline_pred19), ("ours", ours_pred19)]:
                    confusion, metrics = confusion_metrics(gt_pred19, pred19, target_key)
                    row["targets"].setdefault(target_key, {})[method] = metrics
                b = row["targets"][target_key]["baseline"]
                o = row["targets"][target_key]["ours"]
                row["targets"][target_key]["delta_ours_minus_baseline"] = {
                    metric: o[metric] - b[metric] for metric in ["mIoU", "mAcc", "pixelAcc"]
                }
            view_rows.append(row)
            results["views"].append(row)

    # Rebuild summary confusions correctly from per-view predictions. The loop above keeps
    # per-view metrics in JSON; this pass accumulates all pixels without serializing matrices.
    summary_confusions = {
        method: {
            target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
            for target_key in args.targets
        }
        for method in ["baseline", "ours"]
    }
    for row in tqdm(view_rows, desc="Accumulate"):
        scene = row["scene"]
        render_name = row["render_name"]
        gt_pred19 = np.load(pred_cache_path(output_root, "gt_rgb", scene, render_name))
        for method in ["baseline", "ours"]:
            pred19 = load_render_pred(eval_root, method, scene, render_name)
            for target_key in args.targets:
                confusion, _ = confusion_metrics(gt_pred19, pred19, target_key)
                summary_confusions[method][target_key] += confusion

    summary = {
        method: {
            target_key: metrics_from_confusion(summary_confusions[method][target_key])
            for target_key in args.targets
        }
        for method in ["baseline", "ours"]
    }
    summary["delta_ours_minus_baseline"] = {}
    for target_key in args.targets:
        summary["delta_ours_minus_baseline"][target_key] = {
            metric: summary["ours"][target_key][metric] - summary["baseline"][target_key][metric]
            for metric in ["mIoU", "mAcc", "pixelAcc"]
        }
    results["summary"] = summary

    for row in view_rows:
        target_metrics = row["targets"][args.rank_target]
        row["baseline_metrics"] = target_metrics["baseline"]
        row["ours_metrics"] = target_metrics["ours"]
        row["delta_mIoU"] = (
            target_metrics["delta_ours_minus_baseline"]["mIoU"]
            if not math.isnan(target_metrics["delta_ours_minus_baseline"]["mIoU"])
            else 0.0
        )
        row["abs_delta_mIoU"] = abs(row["delta_mIoU"])

    ranked = sorted(view_rows, key=lambda item: item["abs_delta_mIoU"], reverse=True)
    top_rows = ranked[: args.top_k]
    largest = ranked[0]

    panel_dir = output_root / "largest_difference"
    panel_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(top_rows, start=1):
        scene = row["scene"]
        render_name = row["render_name"]
        gt_pred19 = np.load(pred_cache_path(output_root, "gt_rgb", scene, render_name))
        baseline_pred19 = load_render_pred(eval_root, "baseline", scene, render_name)
        ours_pred19 = load_render_pred(eval_root, "ours", scene, render_name)
        panel = make_panel(
            row["gt_image_path"],
            row["baseline_render_path"],
            row["ours_render_path"],
            gt_pred19,
            baseline_pred19,
            ours_pred19,
            args.rank_target,
        )
        sign = "plus" if row["delta_mIoU"] >= 0 else "minus"
        panel_path = (
            panel_dir
            / f"rank{rank:02d}_{scene}_{Path(render_name).stem}_{args.rank_target}cls_"
            f"{sign}{abs(row['delta_mIoU']):.2f}_panel.png"
        )
        cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        row["panel_path"] = str(panel_path)

    largest["panel_path"] = top_rows[0]["panel_path"]
    results["largest_difference"] = largest
    results["top_abs_differences"] = top_rows

    json_path = output_root / "gt_image_reference_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = output_root / "gt_image_reference_view_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "render_name",
                "img_name",
                "target",
                "baseline_mIoU",
                "ours_mIoU",
                "delta_mIoU",
                "baseline_mAcc",
                "ours_mAcc",
                "delta_mAcc",
                "gt_image_path",
                "baseline_render_path",
                "ours_render_path",
            ]
        )
        for row in view_rows:
            for target_key in args.targets:
                target = row["targets"][target_key]
                writer.writerow(
                    [
                        row["scene"],
                        row["render_name"],
                        row["img_name"],
                        target_key,
                        target["baseline"]["mIoU"],
                        target["ours"]["mIoU"],
                        target["delta_ours_minus_baseline"]["mIoU"],
                        target["baseline"]["mAcc"],
                        target["ours"]["mAcc"],
                        target["delta_ours_minus_baseline"]["mAcc"],
                        row["gt_image_path"],
                        row["baseline_render_path"],
                        row["ours_render_path"],
                    ]
                )

    write_markdown(output_root / "gt_image_reference_table.md", summary, top_rows, largest, args)
    print(f"[INFO] wrote {json_path}")
    print(f"[INFO] wrote {output_root / 'gt_image_reference_table.md'}")
    print(f"[INFO] wrote top {len(top_rows)} panels to {panel_dir}")
    print(f"[INFO] largest panel {largest['panel_path']}")


if __name__ == "__main__":
    main()
