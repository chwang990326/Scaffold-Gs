import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DATASETS = {
    "mipnerf360": {
        "baseline_root": "/home/chwang/output/mipnerf360_full_scaffoldgs",
        "ours_root": "/home/chwang/output/mipnerf360_full_ours_expert8_b10",
    },
    "scannet": {
        "baseline_root": "/home/chwang/output/scannet10_full_scaffoldgs",
        "ours_root": "/home/chwang/output/scannet10_full_ours_expert8_b10",
    },
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def choose_method(per_view, requested):
    if requested in per_view:
        return requested
    if len(per_view) == 1:
        return next(iter(per_view))
    raise KeyError(f"Method {requested!r} not found. Available methods: {sorted(per_view)}")


def collect_scene_names(root):
    root = Path(root)
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "per_view.json").exists()
    }


def collect_rows(dataset_name, baseline_root, ours_root, method_name):
    baseline_root = Path(baseline_root)
    ours_root = Path(ours_root)
    scenes = sorted(collect_scene_names(baseline_root) & collect_scene_names(ours_root))
    rows = []
    missing = []

    for scene in scenes:
        baseline_per_view = load_json(baseline_root / scene / "per_view.json")
        ours_per_view = load_json(ours_root / scene / "per_view.json")
        baseline_method = choose_method(baseline_per_view, method_name)
        ours_method = choose_method(ours_per_view, method_name)
        baseline_psnr = baseline_per_view[baseline_method]["PSNR"]
        ours_psnr = ours_per_view[ours_method]["PSNR"]

        for image_name in sorted(set(baseline_psnr) & set(ours_psnr)):
            gt_path = baseline_root / scene / "test" / baseline_method / "gt" / image_name
            if not gt_path.exists():
                gt_path = ours_root / scene / "test" / ours_method / "gt" / image_name
            baseline_render_path = baseline_root / scene / "test" / baseline_method / "renders" / image_name
            ours_render_path = ours_root / scene / "test" / ours_method / "renders" / image_name

            if not (gt_path.exists() and baseline_render_path.exists() and ours_render_path.exists()):
                missing.append(
                    {
                        "dataset": dataset_name,
                        "scene": scene,
                        "image": image_name,
                        "gt_path": str(gt_path),
                        "baseline_render_path": str(baseline_render_path),
                        "ours_render_path": str(ours_render_path),
                    }
                )
                continue

            baseline_value = float(baseline_psnr[image_name])
            ours_value = float(ours_psnr[image_name])
            delta = ours_value - baseline_value
            rows.append(
                {
                    "dataset": dataset_name,
                    "scene": scene,
                    "image": image_name,
                    "baseline_psnr": baseline_value,
                    "ours_psnr": ours_value,
                    "delta_psnr": delta,
                    "abs_delta_psnr": abs(delta),
                    "gt_path": str(gt_path),
                    "baseline_render_path": str(baseline_render_path),
                    "ours_render_path": str(ours_render_path),
                    "baseline_method": baseline_method,
                    "ours_method": ours_method,
                }
            )
    return sorted(rows, key=lambda row: row["abs_delta_psnr"], reverse=True), missing


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def resize_to_height(image, height):
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * float(height) / float(image.shape[0]))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def title_panel(image, title):
    title_height = 42
    out = np.zeros((image.shape[0] + title_height, image.shape[1], 3), dtype=np.uint8)
    out[title_height:] = image
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


def make_panel(row):
    gt = read_rgb(row["gt_path"])
    baseline = read_rgb(row["baseline_render_path"])
    ours = read_rgb(row["ours_render_path"])
    target_height = min(gt.shape[0], baseline.shape[0], ours.shape[0])
    gt = resize_to_height(gt, target_height)
    baseline = resize_to_height(baseline, target_height)
    ours = resize_to_height(ours, target_height)

    panels = [
        title_panel(gt, "GT"),
        title_panel(baseline, "baseline"),
        title_panel(ours, "ours"),
    ]
    return np.concatenate(panels, axis=1)


def write_dataset_outputs(dataset_name, rows, output_root, top_k):
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    top_rows = rows[:top_k]

    for rank, row in enumerate(top_rows, start=1):
        sign = "plus" if row["delta_psnr"] >= 0 else "minus"
        image_stem = Path(row["image"]).stem
        panel_path = (
            dataset_dir
            / f"rank{rank:02d}_{row['scene']}_{image_stem}_psnr_{sign}{abs(row['delta_psnr']):.2f}.png"
        )
        panel = make_panel(row)
        cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        row["panel_path"] = str(panel_path)

    csv_path = dataset_dir / "top20_psnr_gap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "dataset",
                "scene",
                "image",
                "baseline_psnr",
                "ours_psnr",
                "delta_psnr",
                "abs_delta_psnr",
                "panel_path",
                "gt_path",
                "baseline_render_path",
                "ours_render_path",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(top_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "dataset": row["dataset"],
                    "scene": row["scene"],
                    "image": row["image"],
                    "baseline_psnr": row["baseline_psnr"],
                    "ours_psnr": row["ours_psnr"],
                    "delta_psnr": row["delta_psnr"],
                    "abs_delta_psnr": row["abs_delta_psnr"],
                    "panel_path": row.get("panel_path", ""),
                    "gt_path": row["gt_path"],
                    "baseline_render_path": row["baseline_render_path"],
                    "ours_render_path": row["ours_render_path"],
                }
            )

    table_path = dataset_dir / "top20_psnr_gap.md"
    lines = [
        f"# {dataset_name} Top {len(top_rows)} PSNR Gaps",
        "",
        "| Rank | Scene | Image | Baseline PSNR | Ours PSNR | Delta | Abs Delta | Panel |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(top_rows, start=1):
        panel_path = row.get("panel_path", "")
        lines.append(
            f"| {rank} | {row['scene']} | {row['image']} | "
            f"{row['baseline_psnr']:.2f} | {row['ours_psnr']:.2f} | "
            f"{row['delta_psnr']:+.2f} | {row['abs_delta_psnr']:.2f} | `{panel_path}` |"
        )
    table_path.write_text("\n".join(lines))
    return top_rows, csv_path, table_path


def write_summary(output_root, dataset_results):
    lines = ["# Top PSNR Gap Panels", ""]
    for dataset_name, result in dataset_results.items():
        rows = result["top_rows"]
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- Directory: `{output_root / dataset_name}`",
                f"- Table: `{result['table_path']}`",
                f"- CSV: `{result['csv_path']}`",
                "",
                "| Rank | Scene | Image | Baseline PSNR | Ours PSNR | Delta | Abs Delta |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['scene']} | {row['image']} | "
                f"{row['baseline_psnr']:.2f} | {row['ours_psnr']:.2f} | "
                f"{row['delta_psnr']:+.2f} | {row['abs_delta_psnr']:.2f} |"
            )
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Export top PSNR-gap GT/baseline/ours panels.")
    parser.add_argument("--output_root", default="output/psnr_gap_top20")
    parser.add_argument("--method_name", default="ours_30000")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_results = {}
    all_missing = []

    for dataset_name, config in DATASETS.items():
        rows, missing = collect_rows(
            dataset_name,
            config["baseline_root"],
            config["ours_root"],
            args.method_name,
        )
        if not rows:
            raise RuntimeError(f"No aligned PSNR rows found for {dataset_name}")
        top_rows, csv_path, table_path = write_dataset_outputs(dataset_name, rows, output_root, args.top_k)
        dataset_results[dataset_name] = {
            "top_rows": top_rows,
            "csv_path": str(csv_path),
            "table_path": str(table_path),
            "num_aligned_views": len(rows),
        }
        all_missing.extend(missing)

    with open(output_root / "results.json", "w") as f:
        json.dump(
            {
                "output_root": str(output_root),
                "method_name": args.method_name,
                "top_k": args.top_k,
                "datasets": dataset_results,
                "missing": all_missing,
            },
            f,
            indent=2,
        )
    write_summary(output_root, dataset_results)

    print(f"[INFO] wrote {output_root / 'summary.md'}")
    for dataset_name, result in dataset_results.items():
        top = result["top_rows"][0]
        print(
            f"[INFO] {dataset_name}: {len(result['top_rows'])} panels, "
            f"top={top['scene']} {top['image']} delta={top['delta_psnr']:+.2f}"
        )
    if all_missing:
        print(f"[WARN] skipped {len(all_missing)} rows with missing image files")


if __name__ == "__main__":
    main()
