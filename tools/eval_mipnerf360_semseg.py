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
    format_float,
    make_overlay,
    metrics_from_confusion,
    remap_pred19_to_target,
    update_confusion,
)


def sorted_pngs(path):
    return sorted(Path(path).glob("*.png"), key=lambda item: int(item.stem))


def collect_scenes(baseline_root, ours_root, requested_scenes):
    if requested_scenes:
        return requested_scenes
    baseline_root = Path(baseline_root)
    ours_root = Path(ours_root)
    baseline_scenes = {path.name for path in baseline_root.iterdir() if (path / "per_view.json").exists()}
    ours_scenes = {path.name for path in ours_root.iterdir() if (path / "per_view.json").exists()}
    return sorted(baseline_scenes & ours_scenes)


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
    baseline_label = resize_label_like(remap_pred19_to_target(baseline_pred19, target_ids), baseline_rgb.shape[:2])
    ours_label = resize_label_like(remap_pred19_to_target(ours_pred19, target_ids), ours_rgb.shape[:2])

    panels = [
        title_panel(make_overlay(gt_rgb, gt_label), "GT semantic"),
        title_panel(make_overlay(baseline_rgb, baseline_label), "baseline semantic"),
        title_panel(make_overlay(ours_rgb, ours_label), "ours semantic"),
    ]
    height = min(panel.shape[0] for panel in panels)
    panels = [resize_image_to_height(panel, height) for panel in panels]
    return np.concatenate(panels, axis=1)


def init_confusions(targets):
    return {
        target_key: np.zeros((len(TARGET_IDS[target_key]) + 1, len(TARGET_IDS[target_key]) + 1), dtype=np.int64)
        for target_key in targets
    }


def metrics_from_confusions(confusions, targets):
    return {target_key: metrics_from_confusion(confusions[target_key]) for target_key in targets}


def delta_metrics(baseline_metrics, ours_metrics, targets):
    return {
        target_key: {
            metric: ours_metrics[target_key][metric] - baseline_metrics[target_key][metric]
            for metric in ["mIoU", "mAcc", "pixelAcc"]
        }
        for target_key in targets
    }


def aligned_scene_images(baseline_scene_root, ours_scene_root, method_name):
    baseline_test = Path(baseline_scene_root) / "test" / method_name
    ours_test = Path(ours_scene_root) / "test" / method_name
    gt_dir = baseline_test / "gt"
    baseline_render_dir = baseline_test / "renders"
    ours_render_dir = ours_test / "renders"
    names = sorted(
        {path.name for path in sorted_pngs(gt_dir)}
        & {path.name for path in sorted_pngs(baseline_render_dir)}
        & {path.name for path in sorted_pngs(ours_render_dir)},
        key=lambda name: int(Path(name).stem),
    )
    return [
        {
            "image": name,
            "gt_path": str(gt_dir / name),
            "baseline_render_path": str(baseline_render_dir / name),
            "ours_render_path": str(ours_render_dir / name),
        }
        for name in names
    ]


def write_markdown(path, results, targets, rank_target, best_scene, top_rows):
    lines = [
        "# MipNeRF360 ScanNet-SemSeg Reference Evaluation",
        "",
        "Reference is the trained ScanNet semantic model prediction on each real GT RGB image.",
        "",
        "## Overall",
        "",
        "| Method | Target | mIoU | mAcc | Pixel Acc | Valid Pixels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["baseline", "ours"]:
        for target_key in targets:
            metrics = results["overall"][method][target_key]
            lines.append(
                f"| {method} | {target_key} | {format_float(metrics['mIoU'])} | "
                f"{format_float(metrics['mAcc'])} | {format_float(metrics['pixelAcc'])} | "
                f"{metrics['validPixels']} |"
            )

    lines.extend(["", "| Delta | Target | mIoU | mAcc | Pixel Acc |", "|---|---:|---:|---:|---:|"])
    for target_key in targets:
        metrics = results["overall"]["delta_ours_minus_baseline"][target_key]
        lines.append(
            f"| ours - baseline | {target_key} | {metrics['mIoU']:+.2f} | "
            f"{metrics['mAcc']:+.2f} | {metrics['pixelAcc']:+.2f} |"
        )

    lines.extend(
        [
            "",
            f"## Per-Scene {rank_target}-Class Summary",
            "",
            "| Scene | Baseline mIoU | Ours mIoU | Delta mIoU | Baseline mAcc | Ours mAcc | Delta mAcc |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, scene_result in results["scenes"].items():
        baseline = scene_result["summary"]["baseline"][rank_target]
        ours = scene_result["summary"]["ours"][rank_target]
        delta = scene_result["summary"]["delta_ours_minus_baseline"][rank_target]
        marker = " **selected**" if scene == best_scene else ""
        lines.append(
            f"| {scene}{marker} | {baseline['mIoU']:.2f} | {ours['mIoU']:.2f} | "
            f"{delta['mIoU']:+.2f} | {baseline['mAcc']:.2f} | {ours['mAcc']:.2f} | "
            f"{delta['mAcc']:+.2f} |"
        )

    lines.extend(
        [
            "",
            f"## Selected Scene: {best_scene}",
            "",
            f"Selected by highest `{rank_target}`-class ours mIoU.",
            "",
            "| Target | Baseline mIoU | Ours mIoU | Delta mIoU | Baseline mAcc | Ours mAcc | Delta mAcc |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selected_summary = results["scenes"][best_scene]["summary"]
    for target_key in targets:
        baseline = selected_summary["baseline"][target_key]
        ours = selected_summary["ours"][target_key]
        delta = selected_summary["delta_ours_minus_baseline"][target_key]
        lines.append(
            f"| {target_key} | {baseline['mIoU']:.2f} | {ours['mIoU']:.2f} | "
            f"{delta['mIoU']:+.2f} | {baseline['mAcc']:.2f} | {ours['mAcc']:.2f} | "
            f"{delta['mAcc']:+.2f} |"
        )

    lines.extend(
        [
            "",
            f"## Top {len(top_rows)} Ours Views In {best_scene}",
            "",
            f"Ranked by `{rank_target}`-class ours mIoU.",
            "",
            "| Rank | Image | Baseline mIoU | Ours mIoU | Delta mIoU | Baseline mAcc | Ours mAcc | Panel |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(top_rows, start=1):
        metrics = row["targets"][rank_target]
        lines.append(
            f"| {rank} | {row['image']} | {metrics['baseline']['mIoU']:.2f} | "
            f"{metrics['ours']['mIoU']:.2f} | "
            f"{metrics['delta_ours_minus_baseline']['mIoU']:+.2f} | "
            f"{metrics['baseline']['mAcc']:.2f} | {metrics['ours']['mAcc']:.2f} | "
            f"`{row.get('panel_path', '')}` |"
        )

    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained ScanNet semantic model on all MipNeRF360 scenes."
    )
    parser.add_argument("--baseline_root", default="/home/chwang/output/mipnerf360_full_scaffoldgs")
    parser.add_argument("--ours_root", default="/home/chwang/output/mipnerf360_full_ours_expert8_b10")
    parser.add_argument("--method_name", default="ours_30000")
    parser.add_argument(
        "--semseg_checkpoint",
        default="output/scannet_semseg2d/scannet10_deeplabv3_resnet50_19cls_fulltrain_resume_3ep.pt",
    )
    parser.add_argument("--semseg_input_width", type=int, default=384)
    parser.add_argument("--semseg_input_height", type=int, default=288)
    parser.add_argument(
        "--output_root",
        default="output/semantic_eval/mipnerf360_all_scannet_semseg_gtimage_ref",
    )
    parser.add_argument("--targets", nargs="+", default=["19", "15", "10"], choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--rank_target", default="19", choices=sorted(TARGET_IDS.keys()))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--scenes", nargs="*", default=None)
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

    scenes = collect_scenes(args.baseline_root, args.ours_root, args.scenes)
    if not scenes:
        raise RuntimeError("No MipNeRF360 scenes found.")

    overall_confusions = {"baseline": init_confusions(args.targets), "ours": init_confusions(args.targets)}
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
            "top_k": args.top_k,
        },
        "scenes": {},
    }

    for scene in tqdm(scenes, desc="Scenes"):
        scene_rows = aligned_scene_images(
            Path(args.baseline_root) / scene,
            Path(args.ours_root) / scene,
            args.method_name,
        )
        if not scene_rows:
            continue
        scene_confusions = {"baseline": init_confusions(args.targets), "ours": init_confusions(args.targets)}
        view_results = []

        for row in tqdm(scene_rows, desc=scene, leave=False):
            gt_pred19, _, _ = segmenter.segment("gt", scene, row["image"], Path(row["gt_path"]))
            baseline_pred19, _, _ = segmenter.segment("baseline", scene, row["image"], Path(row["baseline_render_path"]))
            ours_pred19, _, _ = segmenter.segment("ours", scene, row["image"], Path(row["ours_render_path"]))

            row_result = dict(row)
            row_result["targets"] = {}
            for target_key in args.targets:
                row_result["targets"][target_key] = {}
                for method, pred19 in [("baseline", baseline_pred19), ("ours", ours_pred19)]:
                    confusion, metrics = metric_for_target(gt_pred19, pred19, target_key)
                    scene_confusions[method][target_key] += confusion
                    overall_confusions[method][target_key] += confusion
                    row_result["targets"][target_key][method] = metrics
                baseline_metrics = row_result["targets"][target_key]["baseline"]
                ours_metrics = row_result["targets"][target_key]["ours"]
                row_result["targets"][target_key]["delta_ours_minus_baseline"] = {
                    metric: ours_metrics[metric] - baseline_metrics[metric]
                    for metric in ["mIoU", "mAcc", "pixelAcc"]
                }

            rank_metrics = row_result["targets"][args.rank_target]
            row_result["rank_ours_mIoU"] = rank_metrics["ours"]["mIoU"]
            row_result["rank_delta_mIoU"] = rank_metrics["delta_ours_minus_baseline"]["mIoU"]
            view_results.append(row_result)

        baseline_summary = metrics_from_confusions(scene_confusions["baseline"], args.targets)
        ours_summary = metrics_from_confusions(scene_confusions["ours"], args.targets)
        results["scenes"][scene] = {
            "summary": {
                "baseline": baseline_summary,
                "ours": ours_summary,
                "delta_ours_minus_baseline": delta_metrics(baseline_summary, ours_summary, args.targets),
            },
            "views": view_results,
        }

    baseline_overall = metrics_from_confusions(overall_confusions["baseline"], args.targets)
    ours_overall = metrics_from_confusions(overall_confusions["ours"], args.targets)
    results["overall"] = {
        "baseline": baseline_overall,
        "ours": ours_overall,
        "delta_ours_minus_baseline": delta_metrics(baseline_overall, ours_overall, args.targets),
    }

    best_scene = max(
        results["scenes"],
        key=lambda scene: results["scenes"][scene]["summary"]["ours"][args.rank_target]["mIoU"],
    )
    selected_rows = sorted(
        results["scenes"][best_scene]["views"],
        key=lambda row: -float(row["rank_ours_mIoU"]) if not math.isnan(float(row["rank_ours_mIoU"])) else float("inf"),
    )[: args.top_k]

    panel_dir = output_root / "best_scene_top_ours" / best_scene
    panel_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(selected_rows, start=1):
        gt_pred19, _, _ = segmenter.segment("gt", best_scene, row["image"], Path(row["gt_path"]))
        baseline_pred19, _, _ = segmenter.segment("baseline", best_scene, row["image"], Path(row["baseline_render_path"]))
        ours_pred19, _, _ = segmenter.segment("ours", best_scene, row["image"], Path(row["ours_render_path"]))
        panel = make_semantic_panel(row, gt_pred19, baseline_pred19, ours_pred19, args.rank_target)
        panel_path = panel_dir / (
            f"rank{rank:02d}_{Path(row['image']).stem}_{args.rank_target}cls_ours{row['rank_ours_mIoU']:.2f}.png"
        )
        cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        row["panel_path"] = str(panel_path)

    results["selected_scene"] = {
        "scene": best_scene,
        "criterion": f"highest {args.rank_target}-class ours mIoU",
        "top_ours_views": selected_rows,
    }

    with open(output_root / "mipnerf360_semseg_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_root / "mipnerf360_semseg_view_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
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
                "gt_path",
                "baseline_render_path",
                "ours_render_path",
            ]
        )
        for scene, scene_result in results["scenes"].items():
            for row in scene_result["views"]:
                for target_key in args.targets:
                    metrics = row["targets"][target_key]
                    writer.writerow(
                        [
                            scene,
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
                            row["gt_path"],
                            row["baseline_render_path"],
                            row["ours_render_path"],
                        ]
                    )

    write_markdown(
        output_root / "mipnerf360_semseg_table.md",
        results,
        args.targets,
        args.rank_target,
        best_scene,
        selected_rows,
    )
    print(f"[INFO] wrote {output_root / 'mipnerf360_semseg_table.md'}")
    print(f"[INFO] selected_scene={best_scene}")
    print(f"[INFO] wrote top {len(selected_rows)} panels to {panel_dir}")


if __name__ == "__main__":
    main()
