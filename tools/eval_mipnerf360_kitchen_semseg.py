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
    TARGET_IDS,
    ScannetSemSegPostSegmenter,
    colorize_label_map,
    format_float,
    make_overlay,
    metrics_from_confusion,
    remap_pred19_to_target,
    update_confusion,
)


def sorted_pngs(path):
    return sorted(Path(path).glob("*.png"), key=lambda item: int(item.stem))


def resize_label_like(label_map, shape_hw):
    if tuple(label_map.shape) == tuple(shape_hw):
        return label_map
    return cv2.resize(
        label_map.astype(np.int32),
        (int(shape_hw[1]), int(shape_hw[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


def resize_image_to_height(image_rgb, height):
    if image_rgb.shape[0] == height:
        return image_rgb
    width = max(1, int(round(image_rgb.shape[1] * float(height) / float(image_rgb.shape[0]))))
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def title_panel(image_rgb, title):
    title_height = 42
    out = np.zeros((image_rgb.shape[0] + title_height, image_rgb.shape[1], 3), dtype=np.uint8)
    out[title_height:] = image_rgb
    cv2.putText(
        out,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def metric_for_target(reference_pred19, pred19, target_key):
    target_ids = TARGET_IDS[target_key]
    reference = remap_pred19_to_target(reference_pred19, target_ids)
    pred = remap_pred19_to_target(pred19, target_ids)
    pred = resize_label_like(pred, reference.shape)
    confusion = np.zeros((len(target_ids) + 1, len(target_ids) + 1), dtype=np.int64)
    update_confusion(confusion, reference, pred, len(target_ids))
    return confusion, metrics_from_confusion(confusion)


def make_semantic_panel(row, gt_pred19, baseline_pred19, ours_pred19, target_key):
    target_ids = TARGET_IDS[target_key]
    gt_rgb = np.asarray(Image.open(row["gt_path"]).convert("RGB"))
    baseline_rgb = np.asarray(Image.open(row["baseline_render_path"]).convert("RGB"))
    ours_rgb = np.asarray(Image.open(row["ours_render_path"]).convert("RGB"))

    gt_label = remap_pred19_to_target(gt_pred19, target_ids)
    baseline_label = remap_pred19_to_target(baseline_pred19, target_ids)
    ours_label = remap_pred19_to_target(ours_pred19, target_ids)

    baseline_label = resize_label_like(baseline_label, baseline_rgb.shape[:2])
    ours_label = resize_label_like(ours_label, ours_rgb.shape[:2])

    panels = [
        title_panel(make_overlay(gt_rgb, gt_label), "GT semantic"),
        title_panel(make_overlay(baseline_rgb, baseline_label), "baseline semantic"),
        title_panel(make_overlay(ours_rgb, ours_label), "ours semantic"),
    ]
    height = min(panel.shape[0] for panel in panels)
    panels = [resize_image_to_height(panel, height) for panel in panels]
    return np.concatenate(panels, axis=1)


def write_markdown(path, results, targets, largest, rank_target):
    lines = [
        "# MipNeRF360 Kitchen ScanNet-SemSeg Reference Evaluation",
        "",
        "Reference is the trained ScanNet semantic model prediction on the real GT RGB image.",
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

    lines.extend(["", "| Delta | Target | mIoU | mAcc | Pixel Acc |", "|---|---:|---:|---:|---:|"])
    for target_key in targets:
        metrics = results["summary"]["delta_ours_minus_baseline"][target_key]
        lines.append(
            f"| ours - baseline | {target_key} | {metrics['mIoU']:+.2f} | "
            f"{metrics['mAcc']:+.2f} | {metrics['pixelAcc']:+.2f} |"
        )

    lines.extend(
        [
            "",
            f"## Largest Difference ({rank_target}-class mIoU)",
            "",
            f"- Image: `{largest['image']}`",
            f"- Baseline mIoU: {largest['targets'][rank_target]['baseline']['mIoU']:.2f}",
            f"- Ours mIoU: {largest['targets'][rank_target]['ours']['mIoU']:.2f}",
            f"- Delta: {largest['targets'][rank_target]['delta_ours_minus_baseline']['mIoU']:+.2f}",
            f"- Abs delta: {largest['abs_delta_mIoU']:.2f}",
            f"- Panel: `{largest['panel_path']}`",
            f"- GT image: `{largest['gt_path']}`",
            f"- Baseline render: `{largest['baseline_render_path']}`",
            f"- Ours render: `{largest['ours_render_path']}`",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained ScanNet semantic model on MipNeRF360 kitchen GT/baseline/ours images."
    )
    parser.add_argument("--baseline_root", default="/home/chwang/output/mipnerf360_full_scaffoldgs/kitchen")
    parser.add_argument("--ours_root", default="/home/chwang/output/mipnerf360_full_ours_expert8_b10/kitchen")
    parser.add_argument("--method_name", default="ours_30000")
    parser.add_argument(
        "--semseg_checkpoint",
        default="output/scannet_semseg2d/scannet10_deeplabv3_resnet50_19cls_fulltrain_resume_3ep.pt",
    )
    parser.add_argument("--semseg_input_width", type=int, default=384)
    parser.add_argument("--semseg_input_height", type=int, default=288)
    parser.add_argument(
        "--output_root",
        default="output/semantic_eval/mipnerf360_kitchen_scannet_semseg_gtimage_ref",
    )
    parser.add_argument("--targets", nargs="+", default=["19", "15", "10"], choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--rank_target", default="19", choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refresh_pred", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.semseg_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    segmenter_args = SimpleNamespace(
        semseg_checkpoint=str(checkpoint_path),
        semseg_input_width=args.semseg_input_width,
        semseg_input_height=args.semseg_input_height,
        output_root=str(output_root),
        device=args.device,
        refresh_postseg=args.refresh_pred,
    )
    segmenter = ScannetSemSegPostSegmenter(segmenter_args)

    baseline_test = Path(args.baseline_root) / "test" / args.method_name
    ours_test = Path(args.ours_root) / "test" / args.method_name
    gt_dir = baseline_test / "gt"
    baseline_render_dir = baseline_test / "renders"
    ours_render_dir = ours_test / "renders"

    names = sorted(
        {
            path.name for path in sorted_pngs(gt_dir)
        }
        & {path.name for path in sorted_pngs(baseline_render_dir)}
        & {path.name for path in sorted_pngs(ours_render_dir)},
        key=lambda name: int(Path(name).stem),
    )
    if not names:
        raise RuntimeError("No aligned kitchen GT/baseline/ours images found.")

    summary_confusions = {
        method: {
            target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
            for target_key in args.targets
        }
        for method in ["baseline", "ours"]
    }
    rows = []

    for name in tqdm(names, desc="kitchen views"):
        gt_path = gt_dir / name
        baseline_path = baseline_render_dir / name
        ours_path = ours_render_dir / name

        gt_pred19, _, _ = segmenter.segment("gt", "kitchen", name, gt_path)
        baseline_pred19, _, _ = segmenter.segment("baseline", "kitchen", name, baseline_path)
        ours_pred19, _, _ = segmenter.segment("ours", "kitchen", name, ours_path)

        row = {
            "image": name,
            "gt_path": str(gt_path),
            "baseline_render_path": str(baseline_path),
            "ours_render_path": str(ours_path),
            "targets": {},
        }
        for target_key in args.targets:
            row["targets"][target_key] = {}
            for method, pred19 in [("baseline", baseline_pred19), ("ours", ours_pred19)]:
                confusion, metrics = metric_for_target(gt_pred19, pred19, target_key)
                summary_confusions[method][target_key] += confusion
                row["targets"][target_key][method] = metrics
            baseline_metrics = row["targets"][target_key]["baseline"]
            ours_metrics = row["targets"][target_key]["ours"]
            row["targets"][target_key]["delta_ours_minus_baseline"] = {
                metric: ours_metrics[metric] - baseline_metrics[metric]
                for metric in ["mIoU", "mAcc", "pixelAcc"]
            }
        rank_delta = row["targets"][args.rank_target]["delta_ours_minus_baseline"]["mIoU"]
        row["delta_mIoU"] = rank_delta if not math.isnan(float(rank_delta)) else 0.0
        row["abs_delta_mIoU"] = abs(row["delta_mIoU"])
        rows.append(row)

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

    largest = sorted(rows, key=lambda row: row["abs_delta_mIoU"], reverse=True)[0]
    gt_pred19, _, _ = segmenter.segment("gt", "kitchen", largest["image"], Path(largest["gt_path"]))
    baseline_pred19, _, _ = segmenter.segment("baseline", "kitchen", largest["image"], Path(largest["baseline_render_path"]))
    ours_pred19, _, _ = segmenter.segment("ours", "kitchen", largest["image"], Path(largest["ours_render_path"]))
    panel = make_semantic_panel(largest, gt_pred19, baseline_pred19, ours_pred19, args.rank_target)
    panel_dir = output_root / "largest_difference"
    panel_dir.mkdir(parents=True, exist_ok=True)
    sign = "plus" if largest["delta_mIoU"] >= 0 else "minus"
    panel_path = panel_dir / (
        f"kitchen_{Path(largest['image']).stem}_{args.rank_target}cls_{sign}{abs(largest['delta_mIoU']):.2f}.png"
    )
    cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    largest["panel_path"] = str(panel_path)

    results = {
        "metadata": {
            "baseline_root": args.baseline_root,
            "ours_root": args.ours_root,
            "method_name": args.method_name,
            "semseg_checkpoint": str(checkpoint_path),
            "semseg_input_width": args.semseg_input_width,
            "semseg_input_height": args.semseg_input_height,
            "device": args.device,
            "reference": "semantic prediction on GT RGB image",
            "targets": args.targets,
            "rank_target": args.rank_target,
        },
        "summary": summary,
        "views": rows,
        "largest_difference": largest,
    }
    with open(output_root / "kitchen_semseg_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_root / "kitchen_semseg_view_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "target",
                "baseline_mIoU",
                "ours_mIoU",
                "delta_mIoU",
                "baseline_mAcc",
                "ours_mAcc",
                "delta_mAcc",
                "baseline_pixelAcc",
                "ours_pixelAcc",
                "delta_pixelAcc",
            ]
        )
        for row in rows:
            for target_key in args.targets:
                metrics = row["targets"][target_key]
                writer.writerow(
                    [
                        row["image"],
                        target_key,
                        metrics["baseline"]["mIoU"],
                        metrics["ours"]["mIoU"],
                        metrics["delta_ours_minus_baseline"]["mIoU"],
                        metrics["baseline"]["mAcc"],
                        metrics["ours"]["mAcc"],
                        metrics["delta_ours_minus_baseline"]["mAcc"],
                        metrics["baseline"]["pixelAcc"],
                        metrics["ours"]["pixelAcc"],
                        metrics["delta_ours_minus_baseline"]["pixelAcc"],
                    ]
                )

    write_markdown(output_root / "kitchen_semseg_table.md", results, args.targets, largest, args.rank_target)
    print(f"[INFO] wrote {output_root / 'kitchen_semseg_table.md'}")
    print(f"[INFO] largest panel {panel_path}")


if __name__ == "__main__":
    main()
