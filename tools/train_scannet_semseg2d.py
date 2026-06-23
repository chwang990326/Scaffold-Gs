import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_scannet_postseg import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NYU40_NAMES,
    SCANNET_SEMSEG_NUM_CLASSES,
    TARGET_IDS,
    load_transform_meta,
    read_label_mesh,
    remap_labels,
    render_gt_semantic,
)


def collect_scenes(data_root, model_root, requested_scenes):
    if requested_scenes:
        return requested_scenes
    return sorted(
        path.name
        for path in Path(model_root).iterdir()
        if path.is_dir()
        and (path / "cameras.json").exists()
        and (Path(data_root) / path.name / "transforms_train.json").exists()
    )


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def camera_map_for_scene(model_scene_dir):
    cameras = load_json(Path(model_scene_dir) / "cameras.json")
    return {str(camera["img_name"]): camera for camera in cameras}


def train_names_for_scene(model_scene_dir):
    sparse_path = Path(model_scene_dir) / "sparse_train_views.json"
    if sparse_path.exists():
        return [str(name) for name in load_json(sparse_path)["image_names"]]
    return sorted(camera_map_for_scene(model_scene_dir), key=lambda name: int(name))


def image_path_for_name(scene_data_dir, img_name):
    for suffix in (".jpg", ".png", ".jpeg"):
        path = Path(scene_data_dir) / "color" / f"{img_name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing RGB image for {scene_data_dir} image {img_name}")


class ProjectedLabelCache:
    def __init__(self, data_root, model_root, cache_root, refresh=False):
        self.data_root = Path(data_root)
        self.model_root = Path(model_root)
        self.cache_root = Path(cache_root)
        self.refresh = refresh
        self.scene_cache = {}

    def _load_scene(self, scene):
        if scene in self.scene_cache:
            return self.scene_cache[scene]
        scene_data_dir = self.data_root / scene
        label_mesh_path = scene_data_dir / f"{scene}_vh_clean_2.labels.ply"
        loaded = {
            "mesh": read_label_mesh(label_mesh_path),
            "intrinsics": load_transform_meta(scene_data_dir),
            "cameras": camera_map_for_scene(self.model_root / scene),
        }
        self.scene_cache[scene] = loaded
        return loaded

    def get(self, scene, img_name, width, height):
        cache_path = self.cache_root / scene / f"{img_name}.npy"
        if cache_path.exists() and not self.refresh:
            return np.load(cache_path)

        loaded = self._load_scene(scene)
        camera = loaded["cameras"][str(img_name)]
        points, labels, faces = loaded["mesh"]
        gt_original = render_gt_semantic(
            points,
            labels,
            faces,
            camera,
            loaded["intrinsics"],
            width,
            height,
        )
        gt19 = remap_labels(gt_original, TARGET_IDS["19"]).astype(np.uint8)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, gt19)
        return gt19


class ScanNetSemSeg2DDataset(Dataset):
    def __init__(self, samples, label_cache, input_width, input_height, ignore_background=True, augment=False):
        self.samples = samples
        self.label_cache = label_cache
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.ignore_background = bool(ignore_background)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        width, height = image.size
        label = self.label_cache.get(sample["scene"], sample["img_name"], width, height)

        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        label = cv2.resize(label, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.ascontiguousarray(label[:, ::-1])

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        label_tensor = torch.from_numpy(label.astype(np.int64))
        if self.ignore_background:
            label_tensor = label_tensor.clone()
            label_tensor[label_tensor == 0] = 255
        return image_tensor, label_tensor


def build_samples(args, scenes):
    samples = []
    for scene in scenes:
        scene_data_dir = Path(args.data_root) / scene
        names = train_names_for_scene(Path(args.model_root) / scene)
        if args.max_train_views_per_scene > 0:
            names = names[: args.max_train_views_per_scene]
        for img_name in names:
            image_path = image_path_for_name(scene_data_dir, img_name)
            samples.append({"scene": scene, "img_name": str(img_name), "image_path": str(image_path)})
    return samples


def build_model(args):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=SCANNET_SEMSEG_NUM_CLASSES,
    )
    if args.imagenet_backbone:
        backbone_path = Path(args.imagenet_backbone_path)
        if backbone_path.exists():
            state = torch.load(backbone_path, map_location="cpu")
            missing, unexpected = model.backbone.load_state_dict(state, strict=False)
            print(
                f"[INFO] loaded local ResNet50 backbone from {backbone_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(f"[WARN] Missing local ImageNet backbone: {backbone_path}; using random init.")
    return model


def freeze_batch_norm(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


def save_checkpoint(path, model, args, scenes, epoch, global_step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": SCANNET_SEMSEG_NUM_CLASSES,
            "target_ids_19": TARGET_IDS["19"],
            "target_names_19": [NYU40_NAMES[idx] for idx in TARGET_IDS["19"]],
            "input_width": int(args.input_width),
            "input_height": int(args.input_height),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "scenes": scenes,
            "ignore_background": bool(args.ignore_background),
            "model": "torchvision.deeplabv3_resnet50",
        },
        path,
    )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenes = collect_scenes(args.data_root, args.model_root, args.scenes)
    if not scenes:
        raise RuntimeError("No ScanNet scenes found for semseg training.")

    samples = build_samples(args, scenes)
    if not samples:
        raise RuntimeError("No training RGB views found.")

    label_cache = ProjectedLabelCache(args.data_root, args.model_root, args.label_cache_root, args.refresh_labels)
    dataset = ScanNetSemSeg2DDataset(
        samples,
        label_cache,
        args.input_width,
        args.input_height,
        ignore_background=args.ignore_background,
        augment=not args.no_augment,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(args).to(device)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[INFO] resumed model weights from {args.resume_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0

    print(f"[INFO] scenes={len(scenes)} samples={len(samples)} output={args.output}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            model.apply(freeze_batch_norm)
        loss_sum = 0.0
        seen = 0
        iterator = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)["out"]
                loss = F.cross_entropy(logits, labels, ignore_index=255)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            seen += batch_size
            global_step += 1
            iterator.set_postfix(loss=f"{loss_sum / max(seen, 1):.4f}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        save_checkpoint(args.output, model, args, scenes, epoch, global_step)
        print(f"[INFO] saved {args.output} after epoch={epoch} step={global_step}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight ScanNet 2D semantic segmentation backend from projected mesh labels."
    )
    parser.add_argument("--data_root", default="/home/chwang/data/InstanceGS_data/scannet_wym")
    parser.add_argument("--model_root", default="/home/chwang/output/scannet10_full_scaffoldgs")
    parser.add_argument("--label_cache_root", default="output/scannet_semseg2d/label_cache")
    parser.add_argument("--output", default="output/scannet_semseg2d/scannet_deeplabv3_resnet50_19cls.pt")
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--input_width", type=int, default=512)
    parser.add_argument("--input_height", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_train_views_per_scene", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh_labels", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--ignore_background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imagenet_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imagenet_backbone_path", default="/home/chwang/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
    parser.add_argument("--freeze_bn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
