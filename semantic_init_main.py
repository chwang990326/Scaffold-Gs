import os
import sys
import struct
import numpy as np
import torch
import cv2
import collections
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import SamModel, SamProcessor, CLIPModel, CLIPImageProcessor
import torch.nn.functional as F
from PIL import Image as PILImage

# [新增] 导入 PCA
from sklearn.decomposition import PCA

# ==========================================
# 1. COLMAP 数据读取模块 (保持不变)
# ==========================================

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            if model_id == 0: num_params = 3
            elif model_id == 1: num_params = 4
            elif model_id == 2: num_params = 5 
            elif model_id == 3: num_params = 8 
            else: num_params = 4 
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = Camera(id=camera_id, model=model_id, width=width, height=height, params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<IdddddddI", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24) 
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=None, point3D_ids=None)
    return images

# ==========================================
# 2. HQ-SAM + CLIP 混合模块
# ==========================================

class SAMCLIPSegmentor:
    def __init__(self, sam_model_dir, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        
        print(f"[INFO] Loading HQ-SAM from {sam_model_dir}...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_dir)
        self.sam_model = SamModel.from_pretrained(sam_model_dir).to(device)
        self.sam_model.eval()

        print(f"[INFO] Loading CLIP from {clip_model_name}...")
        self.clip_model, self.clip_image_processor = self._load_clip_components(clip_model_name)
        self.clip_model.eval()

    def _load_clip_components(self, clip_model_name):
        clip_candidates = []
        if clip_model_name:
            clip_candidates.append(clip_model_name)

        local_fallbacks = [
            "./weights/clip-vit-base-patch32",
            "./weights/openai/clip-vit-base-patch32",
            "./weights/openai_clip-vit-base-patch32",
        ]
        for candidate in local_fallbacks:
            if candidate not in clip_candidates:
                clip_candidates.append(candidate)

        last_error = None

        for candidate in clip_candidates:
            if not candidate:
                continue
            is_local_path = os.path.isdir(candidate)
            try:
                model = CLIPModel.from_pretrained(candidate, local_files_only=True).to(self.device)
                image_processor = CLIPImageProcessor.from_pretrained(candidate, local_files_only=True)
                print(f"[INFO] CLIP loaded successfully from {candidate}")
                return model, image_processor
            except Exception as exc:
                last_error = exc
                if is_local_path:
                    print(f"[WARN] Failed to load local CLIP from {candidate}: {exc}")

        try:
            model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
            print(f"[INFO] CLIP downloaded successfully from {clip_model_name}")
            return model, image_processor
        except Exception as exc:
            last_error = exc

        raise RuntimeError(
            "Failed to load CLIP image model. "
            "Pass a local HuggingFace CLIP directory with --clip_model_path, "
            "or pre-download openai/clip-vit-base-patch32 to ./weights/clip-vit-base-patch32."
        ) from last_error

    @torch.no_grad()
    def process_image(self, image_path, points_per_side=32, iou_thresh=0.88, area_thresh=100, boundary_kernel=5, min_interior_area=32):
        """
        ???????????Feature Map (??? CLIP ???)
        """
        image_cv = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        target_size = 1024
        scale = target_size / max(h, w)
        input_image = cv2.resize(image_rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else image_rgb
        inputs = self.sam_processor(input_image, return_tensors="pt").to(self.device)
        image_embeddings = self.sam_model.vision_encoder(inputs.pixel_values).last_hidden_state

        x = np.linspace(0, input_image.shape[1] - 1, points_per_side)
        y = np.linspace(0, input_image.shape[0] - 1, points_per_side)
        xv, yv = np.meshgrid(x, y)
        points_tensor = torch.tensor(np.stack([xv.flatten(), yv.flatten()], axis=1), device=self.device).float().reshape(-1, 1, 1, 2)

        final_id_map = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        global_best_scores = torch.zeros((h, w), dtype=torch.float32, device=self.device)

        batch_size = 32
        for i in range(0, points_tensor.shape[0], batch_size):
            b_points = points_tensor[i : i+batch_size]
            curr_bs = b_points.shape[0]
            b_embeddings = image_embeddings.repeat(curr_bs, 1, 1, 1)
            outputs = self.sam_model(
                image_embeddings=b_embeddings,
                input_points=b_points,
                input_labels=torch.ones((curr_bs, 1, 1), device=self.device).long(),
            )
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks,
                [[h, w]] * curr_bs,
                [[input_image.shape[0], input_image.shape[1]]] * curr_bs,
            )

            batch_masks = torch.stack(masks).squeeze(1)
            batch_scores = outputs.iou_scores.squeeze(1)
            best_idx = torch.argmax(batch_scores, dim=1)
            rows = torch.arange(curr_bs, device=self.device)
            final_masks = batch_masks[rows, best_idx]
            final_scores = batch_scores[rows, best_idx]

            for j in range(curr_bs):
                if final_scores[j] > iou_thresh and final_masks[j].sum() > area_thresh:
                    mask_bool = final_masks[j] > 0
                    update_mask = mask_bool & (final_scores[j] > global_best_scores)
                    final_id_map[update_mask] = i + j + 1
                    global_best_scores[update_mask] = final_scores[j]
            del outputs
            torch.cuda.empty_cache()

        unique_ids = torch.unique(final_id_map)
        unique_ids = unique_ids[unique_ids > 0]

        id_to_clip_feat = {}
        id_to_score = {}
        interior_id_map = torch.zeros_like(final_id_map)
        kernel = None
        if boundary_kernel > 1:
            kernel = np.ones((boundary_kernel, boundary_kernel), dtype=np.uint8)

        for mid in unique_ids:
            mask = (final_id_map == mid).cpu().numpy().astype(np.uint8)
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0:
                continue

            if kernel is not None:
                eroded_mask = cv2.erode(mask, kernel, iterations=1)
                if int(eroded_mask.sum()) >= int(min_interior_area):
                    interior_mask = eroded_mask
                else:
                    interior_mask = np.zeros_like(mask)
            else:
                interior_mask = mask

            if interior_mask.any():
                interior_id_map[torch.from_numpy(interior_mask.astype(bool)).to(self.device)] = mid

            y1, y2, x1, x2 = y_indices.min(), y_indices.max(), x_indices.min(), x_indices.max()
            crop_rgb = image_rgb[y1:y2+1, x1:x2+1]
            crop_mask = mask[y1:y2+1, x1:x2+1].astype(bool)
            masked_crop_rgb = np.zeros_like(crop_rgb)
            masked_crop_rgb[crop_mask] = crop_rgb[crop_mask]

            pil_img = PILImage.fromarray(masked_crop_rgb).convert("RGB")
            if pil_img.size[0] <= 10 or pil_img.size[1] <= 10:
                continue

            clip_inputs = self.clip_image_processor(images=pil_img, return_tensors="pt").to(self.device)
            clip_feat = self.clip_model.get_image_features(**clip_inputs)
            clip_feat = F.normalize(clip_feat, p=2, dim=-1)
            id_to_clip_feat[mid.item()] = clip_feat.squeeze(0)

            pixel_scores = global_best_scores[final_id_map == mid]
            id_to_score[mid.item()] = float(pixel_scores.mean().item()) if pixel_scores.numel() > 0 else 0.0

        return final_id_map, interior_id_map, id_to_clip_feat, id_to_score


# ==========================================
# 3. 3D 关联与特征聚合模块
# ==========================================

def get_intrinsic_matrix(camera):
    K = np.eye(3)
    params = camera.params
    if camera.model == 0: K[0, 0] = params[0]; K[1, 1] = params[0]; K[0, 2] = params[1]; K[1, 2] = params[2]
    elif camera.model == 1: K[0, 0] = params[0]; K[1, 1] = params[1]; K[0, 2] = params[2]; K[1, 2] = params[3]
    return torch.tensor(K, dtype=torch.float32)

class SemanticVoter:
    def __init__(self, scene_path, sam_model_path, clip_model_name="openai/clip-vit-base-patch32", device="cuda", boundary_kernel=5, min_interior_area=32, min_views=2):
        self.device = device
        self.scene_path = scene_path
        self.segmentor = SAMCLIPSegmentor(sam_model_path, clip_model_name=clip_model_name, device=device)
        self.boundary_kernel = max(1, int(boundary_kernel))
        self.min_interior_area = max(1, int(min_interior_area))
        self.min_views = max(1, int(min_views))
        
        # [???] ?????? 2D ?????????????????
        self.mask_out_dir = os.path.join(scene_path, "semantic_masks_2d")
        os.makedirs(self.mask_out_dir, exist_ok=True)
        print(f"[INFO] 2D Semantic masks will be saved to: {self.mask_out_dir}")
        
        sparse_dir = os.path.join(scene_path, "sparse", "0")
        if not os.path.exists(sparse_dir): sparse_dir = os.path.join(scene_path, "sparse")
             
        self.cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        self.images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        self.sorted_image_keys = sorted(self.images.keys(), key=lambda x: self.images[x].name)

    def run(self, anchors):
        num_anchors = anchors.shape[0]
        anchor_features = torch.zeros((num_anchors, 512), device=self.device)
        anchor_score_sums = torch.zeros(num_anchors, device=self.device)
        anchor_view_counts = torch.zeros(num_anchors, device=self.device)
        
        for img_key in tqdm(self.sorted_image_keys, desc="Lifting Semantics"):
            colmap_img = self.images[img_key]
            colmap_cam = self.cameras[colmap_img.camera_id]
            
            full_img_path = os.path.join(self.scene_path, "images_4", colmap_img.name)
            if not os.path.exists(full_img_path):
                full_img_path = os.path.join(self.scene_path, "images", colmap_img.name)
            if not os.path.exists(full_img_path):
                continue

            id_map, interior_id_map, id_to_feat, id_to_score = self.segmentor.process_image(
                full_img_path,
                boundary_kernel=self.boundary_kernel,
                min_interior_area=self.min_interior_area,
            )
            h, w = id_map.shape
            
            id_map_cpu = id_map.cpu().numpy()
            color_map = np.zeros((h, w, 3), dtype=np.uint8)
            unique_ids = np.unique(id_map_cpu)
            for uid in unique_ids:
                if uid == 0:
                    continue
                np.random.seed(int(uid) * 123)
                color = np.random.randint(0, 255, size=3)
                color_map[id_map_cpu == uid] = color

            mask_save_path = os.path.join(self.mask_out_dir, colmap_img.name)
            mask_save_path = os.path.splitext(mask_save_path)[0] + "_mask.png"
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            cv2.imwrite(mask_save_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
            
            K = get_intrinsic_matrix(colmap_cam).to(self.device)
            scale_x = w / colmap_cam.width
            scale_y = h / colmap_cam.height
            if scale_x != 1.0 or scale_y != 1.0:
                K[0, 0] *= scale_x
                K[1, 1] *= scale_y
                K[0, 2] *= scale_x
                K[1, 2] *= scale_y

            R = torch.tensor(qvec2rotmat(colmap_img.qvec), dtype=torch.float32, device=self.device)
            T = torch.tensor(colmap_img.tvec, dtype=torch.float32, device=self.device)
            
            pts_cam = (anchors @ R.t()) + T
            depth = pts_cam[:, 2]
            pts_2d = pts_cam[:, :2] / depth.unsqueeze(1)
            pts_2d = pts_2d @ K[:2, :2].t() + K[:2, 2]
            u, v = pts_2d[:, 0].long(), pts_2d[:, 1].long()
            
            valid_mask = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            
            sampled_ids = interior_id_map[v[valid_indices], u[valid_indices]]
            
            for mid, feat in id_to_feat.items():
                mask_in_img = (sampled_ids == mid)
                affected_anchors = valid_indices[mask_in_img]
                if affected_anchors.numel() == 0:
                    continue
                score = float(id_to_score.get(mid, 0.0))
                anchor_features[affected_anchors] += feat.unsqueeze(0) * score
                anchor_score_sums[affected_anchors] += score
                anchor_view_counts[affected_anchors] += 1
            
            del id_map, interior_id_map, id_to_feat, id_to_score
            torch.cuda.empty_cache()

        mask = anchor_score_sums > 0
        anchor_features[mask] /= anchor_score_sums[mask].unsqueeze(1)
        anchor_features[mask] = F.normalize(anchor_features[mask], p=2, dim=-1)

        anchor_confidence = torch.zeros(num_anchors, device=self.device)
        if mask.any():
            avg_score = torch.zeros_like(anchor_confidence)
            avg_score[mask] = anchor_score_sums[mask] / anchor_view_counts[mask].clamp_min(1.0)
            view_confidence = torch.clamp(anchor_view_counts / float(self.min_views), max=1.0)
            anchor_confidence = avg_score * view_confidence
            anchor_confidence[~mask] = 0.0
        
        print(f"[INFO] 3D Feature lifting done. Original shape: {anchor_features.shape}")

        TARGET_DIM = 128
        if anchor_features.shape[1] > TARGET_DIM:
            print(f"[INFO] Applying PCA to reduce dimensions from {anchor_features.shape[1]} to {TARGET_DIM}...")
            valid_mask_cpu = mask.cpu().numpy()
            features_np = anchor_features.cpu().numpy()
            valid_features = features_np[valid_mask_cpu]
            
            if valid_features.shape[0] > TARGET_DIM:
                pca = PCA(n_components=TARGET_DIM)
                reduced_valid_features = pca.fit_transform(valid_features)
                reduced_features = torch.zeros((num_anchors, TARGET_DIM), device=self.device, dtype=torch.float32)
                reduced_features[mask] = torch.tensor(reduced_valid_features, device=self.device, dtype=torch.float32)
                reduced_features[mask] = F.normalize(reduced_features[mask], p=2, dim=-1)
                anchor_features = reduced_features
                print(f"[INFO] PCA done. New shape: {anchor_features.shape}")
            else:
                print("[WARN] Not enough valid features to run PCA. Skipping.")

        return {
            "features": anchor_features,
            "confidence": anchor_confidence,
            "valid_mask": mask,
        }


if __name__ == "__main__":
    SCENE_PATH = "./data/garden" 
    SAM_MODEL_DIR = "./weights/sam_hq_vit_base" 
    OUTPUT_PATH = os.path.join(SCENE_PATH, "anchor_features.pt")
    
    dummy_anchors = torch.randn((50000, 3), device="cuda") 
    
    voter = SemanticVoter(SCENE_PATH, SAM_MODEL_DIR)
    final_result = voter.run(dummy_anchors)
    
    torch.save(final_result["features"], OUTPUT_PATH)
    print(f"Saved 3D CLIP features to {OUTPUT_PATH}")
