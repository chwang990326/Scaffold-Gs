import os
import sys
import struct
import numpy as np
import torch
import cv2
import collections
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import SamModel, SamProcessor
import torch.nn.functional as F

# ==========================================
# 0. 环境兼容性预处理
# ==========================================
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    test = torch.randn(1, device=device)
    print(f"[INFO] 成功识别显卡: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[WARNING] GPU 模式不可用: {e}")
    device = torch.device("cpu")

# ==========================================
# 1. COLMAP 数据读取模块
# ==========================================

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

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = camera_properties
            if model_id == 0: num_params = 3
            elif model_id == 1: num_params = 4
            elif model_id == 2: num_params = 5
            elif model_id == 3: num_params = 8
            else: num_params = 4 
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_id,
                                        width=width, height=height,
                                        params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_header = read_next_bytes(fid, 4, "I")
            image_id = binary_image_header[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"))
            tvec = np.array(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 4, "I")[0]
            image_name = ""
            while True:
                char = read_next_bytes(fid, 1, "c")[0]
                if char == b"\x00": break
                image_name += char.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24) 
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                    camera_id=camera_id, name=image_name,
                                    xys=None, point3D_ids=None)
    return images

# ==========================================
# 2. HQ-SAM 分割模块 (修复索引形状错误版)
# ==========================================

class SAMHQSegmentor:
    def __init__(self, model_dir, device="cpu"):
        self.device = device
        print(f"[INFO] Loading HQ-SAM from {model_dir} on {self.device}...")
        self.processor = SamProcessor.from_pretrained(model_dir)
        self.model = SamModel.from_pretrained(model_dir).to(self.device)
        
        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path, points_per_side=12, iou_thresh=0.85, area_thresh=100):
        image_cv = cv2.imread(image_path)
        if image_cv is None: return None
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_cv.shape[:2]

        target_size = 1024
        scale = target_size / max(orig_h, orig_w)
        input_image = cv2.resize(image_cv, (int(orig_w*scale), int(orig_h*scale))) if scale < 1.0 else image_cv
        new_h, new_w = input_image.shape[:2]

        x = np.linspace(0, new_w - 1, points_per_side)
        y = np.linspace(0, new_h - 1, points_per_side)
        xv, yv = np.meshgrid(x, y)
        input_points = np.stack([xv.flatten(), yv.flatten()], axis=1)

        input_points = input_points[None, :, None, :] 
        input_labels = np.ones(input_points.shape[:3], dtype=np.int64) 

        inputs = self.processor(input_image, return_tensors="pt").to(self.device)
        outputs = self.model(
            pixel_values=inputs.pixel_values,
            input_points=torch.tensor(input_points, device=self.device, dtype=torch.float32),
            input_labels=torch.tensor(input_labels, device=self.device, dtype=torch.int64),
            multimask_output=False 
        )
        
        # 核心修复：post_process_masks 返回的是在 1024 尺度下的 mask
        masks = self.processor.post_process_masks(
            outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes
        )[0] # [N, 1, new_h, new_w]
        
        masks = masks.squeeze(1) # [N, new_h, new_w]
        iou_scores = outputs.iou_scores[0].squeeze(-1) 

        valid_indices = torch.where(iou_scores > iou_thresh)[0]
        # 画布初始化为原始尺寸
        final_id_map = torch.zeros((orig_h, orig_w), dtype=torch.int32, device=self.device)
        
        if len(valid_indices) > 0:
            valid_masks = masks[valid_indices].float() # 转 float 以便 resize
            # 将 [N, new_h, new_w] 变为 [N, 1, new_h, new_w] 以进行插值
            valid_masks = F.interpolate(valid_masks.unsqueeze(1), size=(orig_h, orig_w), mode="nearest").squeeze(1)
            valid_masks = valid_masks > 0.5 # 转回 bool
            
            valid_scores = iou_scores[valid_indices]
            sort_idx = torch.argsort(valid_scores)
            for idx in sort_idx:
                mask = valid_masks[idx]
                if mask.sum() > area_thresh:
                    final_id_map[mask] = (valid_indices[idx] + 1).int()
        return final_id_map

# ==========================================
# 3. 3D 关联与投票模块
# ==========================================

def get_intrinsic_matrix(camera):
    K = np.eye(3)
    params = camera.params
    if camera.model == 0: # SIMPLE_PINHOLE
        K[0,0]=K[1,1]=params[0]; K[0,2]=params[1]; K[1,2]=params[2]
    elif camera.model == 1: # PINHOLE
        K[0,0]=params[0]; K[1,1]=params[1]; K[0,2]=params[2]; K[1,2]=params[3]
    return torch.tensor(K, dtype=torch.float32)

class SemanticVoter:
    def __init__(self, scene_path, model_path, device="cpu"):
        self.device = device
        self.scene_path = scene_path
        self.segmentor = SAMHQSegmentor(model_path, device=device)
        sparse_dir = os.path.join(scene_path, "sparse", "0")
        if not os.path.exists(sparse_dir): sparse_dir = os.path.join(scene_path, "sparse")
        self.cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        self.images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        self.sorted_image_keys = sorted(self.images.keys(), key=lambda x: self.images[x].name)

    def run(self, anchors):
        num_anchors = anchors.shape[0]
        all_votes_anchor_indices, all_votes_label_ids = [], []
        global_id_offset = 0
        anchors = anchors.to(self.device)
        
        for img_key in tqdm(self.sorted_image_keys, desc="Processing Images"):
            colmap_img = self.images[img_key]
            full_img_path = os.path.join(self.scene_path, "images", colmap_img.name)
            id_map = self.segmentor.process_image(full_img_path)
            if id_map is None: continue
            
            h, w = id_map.shape
            K = get_intrinsic_matrix(self.cameras[colmap_img.camera_id]).to(self.device)
            R = torch.tensor(qvec2rotmat(colmap_img.qvec), dtype=torch.float32, device=self.device)
            T = torch.tensor(colmap_img.tvec, dtype=torch.float32, device=self.device)
            
            pts_cam = (anchors @ R.t()) + T
            depth = pts_cam[:, 2]
            valid_mask = depth > 0.05
            pts_2d = pts_cam[:, :2] / depth.unsqueeze(1)
            pts_2d = pts_2d @ K[:2, :2].t() + K[:2, 2]
            u, v = pts_2d[:, 0].long(), pts_2d[:, 1].long()
            valid_mask &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            valid_idx = torch.where(valid_mask)[0]
            if len(valid_idx) == 0: continue
            sampled_ids = id_map[v[valid_idx], u[valid_idx]]
            has_sem = sampled_ids > 0
            
            all_votes_anchor_indices.append(valid_idx[has_sem].cpu().numpy())
            all_votes_label_ids.append((sampled_ids[has_sem] + global_id_offset).cpu().numpy())
            global_id_offset += (id_map.max().item() + 1)
            
        if not all_votes_anchor_indices: return torch.zeros(num_anchors, dtype=torch.int32)
        
        print("[INFO] Aggregating votes...")
        total_idx = np.concatenate(all_votes_anchor_indices)
        total_lbl = np.concatenate(all_votes_label_ids)
        packed = np.stack((total_idx, total_lbl), axis=1)
        packed = packed[packed[:, 0].argsort()]
        
        final_ids = np.zeros(num_anchors, dtype=np.int32)
        unq_anchors, split_idx = np.unique(packed[:, 0], return_index=True)
        groups = np.split(packed[:, 1], split_idx[1:])
        for i, a_idx in enumerate(unq_anchors):
            vals, counts = np.unique(groups[i], return_counts=True)
            final_ids[a_idx] = vals[np.argmax(counts)]
        return torch.from_numpy(final_ids)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    SCENE_PATH = "./data/garden" 
    MODEL_DIR = "./weights/sam_hq_vit_base" 
    OUTPUT_PATH = os.path.join(SCENE_PATH, "semantic_labels.pt")
    
    print(f"[INFO] Creating points on {device}...")
    dummy_anchors = torch.randn((100000, 3), device=device) 
    
    voter = SemanticVoter(SCENE_PATH, MODEL_DIR, device=device)
    semantic_ids = voter.run(dummy_anchors)
    
    torch.save(semantic_ids, OUTPUT_PATH)
    print(f"[SUCCESS] Saved to {OUTPUT_PATH}")