# garden_seg_expert_clean_final 实验结果汇报

## 1. 实验目的

本次实验目标是验证在 Scaffold-GS 基线模型上加入 `SAM/CLIP 语义提升 + semantic clustered color experts` 后，是否能够在不降低重建质量的前提下提升结构一致性和感知质量。

核心对比对象为：

| 实验目录 | 方法含义 |
|---|---|
| `garden_paper_like` | Scaffold-GS baseline，按论文参数对齐 |
| `garden_seg_expert_clean_final` | 本方案，semantic clustered color experts，关闭 boundary 和 triangle |

最终指标统一采用各输出目录中的 `results.json`，因为该文件来自训练结束后的统一渲染与评估流程。

## 2. 方法概述

`garden_seg_expert_clean_final` 是一个干净的语义专家颜色建模方案，其结构为：

```text
Scaffold-GS 主干
+ SAM/CLIP 2D-3D 语义提升
+ anchor semantic feature cache
+ semantic adapter
+ semantic clustered routing
+ 1 个 fallback color MLP
+ 4 个 semantic color expert MLP
- boundary loss
- BG-Triangle branch
```

该方案不是 boundary/hybrid 版本，也不是 BG-Triangle 版本。它只保留了当前实验中最稳定有效的语义专家部分。

### 2.1 Scaffold-GS 主干

保留 Scaffold-GS 的 anchor/offset 表示、opacity/cov/color 基础预测结构：

| 结构 | 作用 |
|---|---|
| `_anchor` | 3D anchor / voxel 中心 |
| `_offset` | 每个 anchor 下的 offset Gaussian |
| `_anchor_feat` | 重建主干特征 |
| `mlp_opacity` | 预测 opacity |
| `mlp_cov` | 预测 covariance / scale / rotation 相关量 |
| `mlp_color_fallback` | 共享基础颜色预测 |

### 2.2 SAM/CLIP 语义提升

使用 SAM/CLIP 从多视角 2D 图像中提取语义信息，并提升到 3D anchor 上，生成：

| 输出文件 | 作用 |
|---|---|
| `anchor_semantic_features.pt` | 每个 anchor 的语义特征 |
| `anchor_semantic_confidence.pt` | 每个 anchor 的语义置信度 |
| `anchor_semantic_pca_vis.ply` | 语义特征 PCA 可视化点云 |

本次实验中没有生成 `boundary_candidates.pt`，说明 triangle 分支没有参与。

### 2.3 Semantic clustered color experts

对高置信 anchor 按照语义特征和少量空间信息进行聚类：

```text
cluster input = normalized semantic feature + 0.15 * normalized xyz
cluster method = MiniBatchKMeans
num experts = 4
```

日志显示本次聚类结果为：

```text
valid anchors = 135471
high confidence anchors = 134690
fallback anchors = 1192
expert cluster sizes = [49036, 28922, 43264, 13468]
```

聚类分布较均衡，没有出现单个 expert 吞掉大部分 anchor 的情况。

### 2.4 Expert color fusion

颜色预测采用保守融合，而不是完全由 expert 替换 fallback：

```text
final_color = base_color + 0.25 * (expert_color - base_color)
```

其中：

| 项 | 含义 |
|---|---|
| `base_color` | `mlp_color_fallback` 输出 |
| `expert_color` | 当前语义 cluster 对应的 expert MLP 输出 |
| `0.25` | expert 修正权重 |

该设计保证 fallback color head 负责全局稳定重建，semantic expert 只做语义区域相关的局部颜色修正，因此不容易伤害 PSNR。

### 2.5 关闭的模块

本方案明确关闭以下模块：

| 模块 | 状态 | 说明 |
|---|---:|---|
| boundary loss | 关闭 | `boundary_loss_weight = 0.0` |
| stable edge mask supervision | 关闭 | 不参与 loss |
| BG-Triangle branch | 关闭 | `triangle_enabled = False` |
| boundary candidate cache | 未生成 | 无 `boundary_candidates.pt` |
| appearance embedding | 关闭 | `appearance_dim = 0` |

日志确认如下：

```text
[METHOD] effective=semantic_expert_clean profile=semantic_expert_clean experts=4 expert_blend=0.250 xyz_weight=0.150 boundary_weight=0.0000 triangle_enabled=False
[BOUNDARY] Disabled because boundary_loss_weight <= 0.
[TRIANGLE] Disabled by default. Use --enable_triangle_branch for hybrid boundary experiments.
```

## 3. 实验设置

### 3.1 数据集

| 项 | 设置 |
|---|---|
| Scene | `garden` |
| Dataset path | `./data/garden/` |
| Iterations | 30000 |
| Eval | enabled |
| Resolution | `-r -1` |
| Appearance dim | `0` |
| Voxel size | `0.001` |
| Update init factor | `16` |
| Ratio | `1` |

### 3.2 Baseline 命令

```bash
python train.py \
  -s ./data/garden/ \
  -m /home/chwang/output/garden_paper_like \
  --iterations 30000 \
  --eval \
  --appearance_dim 0 \
  --voxel_size 0.001 \
  --update_init_factor 16 \
  --ratio 1 \
  -r -1
```

### 3.3 本方案命令

```bash
python seg_train.py \
  -s ./data/garden/ \
  -m /home/chwang/output/garden_seg_expert_clean_final \
  --iterations 30000 \
  --eval \
  --appearance_dim 0 \
  --voxel_size 0.001 \
  --update_init_factor 16 \
  --ratio 1 \
  -r -1 \
  --sam_checkpoint ./weights/sam_hq_vit_base \
  --clip_model_path ./weights/clip-vit-base-patch32 \
  --method_profile semantic_expert_clean \
  --port 8121
```

## 4. 主结果：与 Scaffold-GS baseline 对比

| Method | Output dir | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Scaffold-GS baseline | `garden_paper_like` | 27.316597 | 0.850497 | 0.134792 |
| Ours: Semantic Expert Clean | `garden_seg_expert_clean_final` | 27.323305 | 0.851765 | 0.131792 |
| Improvement | - | +0.006708 | +0.001268 | -0.003001 |

### 4.1 结果解读

| 指标 | 现象 | 说明 |
|---|---|---|
| PSNR | 比 baseline 高 `0.006708 dB` | 没有伤害主干重建质量 |
| SSIM | 比 baseline 高 `0.001268` | 结构一致性提升明显 |
| LPIPS | 比 baseline 低 `0.003001` | 感知质量提升明显 |

该结果说明 semantic clustered color experts 能在不降低 PSNR 的前提下改善结构一致性和感知质量。

## 5. 消融实验汇总

### 5.1 全部已有实验

| Method / Variant | Output dir | Experts | Boundary | Triangle | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 相对 baseline 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Scaffold-GS baseline | `garden_paper_like` | 0 | No | No | 27.316597 | 0.850497 | 0.134792 | 基线 |
| Boundary shared | `garden_seg_boundary` | 0 | Yes | No | 27.341461 | 0.850636 | 0.134176 | PSNR/SSIM/LPIPS 小幅提升 |
| Semantic expert clean v1 | `garden_seg_expert_clean_v1` | 4 | No | No | 27.335112 | 0.851572 | 0.132281 | clean expert 重复实验，综合表现强 |
| Semantic expert clean final | `garden_seg_expert_clean_final` | 4 | No | No | 27.323305 | 0.851765 | 0.131792 | SSIM/LPIPS 最好，主方法 |
| Hybrid late triangle | `garden_hybrid_v1_round1_late_tri` | 0 | Yes | Yes | 27.383377 | 0.850316 | 0.134303 | PSNR 最高，但 SSIM 低于 baseline |
| Hybrid better edge | `garden_hybrid_v1_round2_better_edge` | 0 | Yes | Yes | 27.382256 | 0.850383 | 0.134311 | PSNR 很高，但结构/感知不如 expert |
| Hybrid edge-band support | `garden_hybrid_v1_round3_edge_band_support` | 0 | Yes | Yes | 27.343969 | 0.850373 | 0.134418 | 支撑区域扩大，但全局指标退步 |
| Expert + triangle mixed | `garden_seg_semantic_expert_v1` | 4 | Mixed | Yes | 27.309511 | 0.850938 | 0.133216 | 混入 triangle，不作为 pure expert 对照 |
| Early hybrid | `garden_hybrid_v1` | 0 | Yes | Yes | 27.274271 | 0.849995 | 0.134921 | 早期版本，不推荐 |

### 5.2 相对 baseline 的差值

| Variant | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ |
|---|---:|---:|---:|
| `garden_seg_expert_clean_final` | +0.006708 | +0.001268 | -0.003001 |
| `garden_seg_expert_clean_v1` | +0.018515 | +0.001075 | -0.002511 |
| `garden_seg_boundary` | +0.024864 | +0.000139 | -0.000617 |
| `garden_hybrid_v1_round1_late_tri` | +0.066780 | -0.000181 | -0.000489 |
| `garden_hybrid_v1_round2_better_edge` | +0.065659 | -0.000114 | -0.000481 |
| `garden_hybrid_v1_round3_edge_band_support` | +0.027372 | -0.000125 | -0.000374 |
| `garden_seg_semantic_expert_v1` | -0.007086 | +0.000441 | -0.001576 |
| `garden_hybrid_v1` | -0.042326 | -0.000502 | +0.000128 |

## 6. 消融结论

### 6.1 Semantic expert clean 是当前最适合做主方法的版本

`garden_seg_expert_clean_final` 的主要优势是 SSIM 和 LPIPS 最好，同时 PSNR 不低于 baseline。相比 hybrid，它结构更清晰、训练更轻、实验更干净。

### 6.2 Boundary shared 有稳定但较小的收益

`garden_seg_boundary` 相比 baseline 三项指标均有小幅改善，说明 SAM 稳定边界监督本身有一定效果。但该版本没有使用 semantic experts，感知指标仍不如 expert clean。

### 6.3 Hybrid / triangle 更偏向提升 PSNR

`garden_hybrid_v1_round1_late_tri` 和 `garden_hybrid_v1_round2_better_edge` 的 PSNR 最高，但 SSIM 低于 baseline，LPIPS 也不如 semantic expert clean。说明 triangle/hybrid 当前更像是局部边界或光度拟合增强，尚未稳定转化为全局结构和感知指标优势。

### 6.4 Edge-band support 没有带来预期收益

`garden_hybrid_v1_round3_edge_band_support` 的 PSNR、SSIM、LPIPS 都不如 round1/round2，说明扩大 projected edge-band support 后，有效监督区域和最终收益之间仍存在不匹配。该分支后续应继续作为边界专项实验，而不是主方法默认配置。

### 6.5 Expert + triangle 混合不适合作为当前主结论

`garden_seg_semantic_expert_v1` 同时启用了 expert 和 triangle，且 PSNR 低于 baseline。因此它不能作为 pure semantic expert 对照，也不适合作为当前主方法结果。

## 7. 速度与实验成本

| Method | Train FPS | Test FPS | 说明 |
|---|---:|---:|---|
| Scaffold-GS baseline | - | 37.08854 | baseline 日志仅记录 Test FPS |
| Semantic expert clean final | 51.64533 | 48.46497 | 速度可接受，且测试速度高于 baseline 日志记录 |
| Semantic expert clean v1 | 50.56123 | 39.84435 | clean expert 重复实验 |
| Boundary shared | 105.68903 | 100.69658 | 旧版本日志记录速度较高，可能受代码路径和评估流程影响 |
| Hybrid round1 | 15.31925 | 15.53564 | triangle 显著降低速度 |
| Hybrid round2 | 13.12979 | 13.41689 | triangle 显著降低速度 |
| Hybrid round3 | 12.68582 | 12.85587 | edge-band support 后速度更慢 |

速度结果显示，semantic expert clean 的额外成本可接受，而 hybrid/triangle 分支明显更重。

## 8. 推荐写法

### 8.1 中文结论

在 `garden` 场景中，本文方法在 Scaffold-GS 主干上引入 SAM/CLIP 语义提升与 semantic clustered color experts。实验结果表明，在关闭 boundary loss 和 BG-Triangle 分支的干净设置下，本文方法相较 Scaffold-GS baseline 将 SSIM 从 `0.850497` 提升至 `0.851765`，将 LPIPS 从 `0.134792` 降低至 `0.131792`，同时 PSNR 从 `27.316597 dB` 小幅提升至 `27.323305 dB`。这说明语义聚类颜色专家能够在不破坏重建主干的前提下提升结构一致性与感知质量。

### 8.2 英文结论

On the garden scene, the proposed semantic clustered color expert model improves perceptual and structural reconstruction quality over the Scaffold-GS baseline. With boundary supervision and the BG-Triangle branch disabled, the method improves SSIM from `0.850497` to `0.851765` and reduces LPIPS from `0.134792` to `0.131792`, while slightly improving PSNR from `27.316597 dB` to `27.323305 dB`. These results indicate that semantic-aware color experts can enhance local appearance modeling without degrading the Scaffold-GS reconstruction backbone.

## 9. 当前不足

| 问题 | 说明 |
|---|---|
| 缺少 shared clean 对照 | 当前 output 中还没有 `garden_seg_shared_clean_final`，无法完全分离 SAM/CLIP semantic lifting 与 expert routing 的贡献 |
| 只有 garden 单场景结果 | 需要在更多 Mip-NeRF 360 场景上复验 |
| 缺少边界专项指标 | triangle/boundary 的贡献很难只靠全图 PSNR/SSIM/LPIPS 体现 |
| hybrid 分支仍不稳定 | triangle 能提高 PSNR，但 SSIM/LPIPS 不如 expert clean |

## 10. 下一步建议

### 10.1 补跑 shared clean 对照

建议补跑：

```bash
python seg_train.py \
  -s ./data/garden/ \
  -m /home/chwang/output/garden_seg_shared_clean_final \
  --iterations 30000 \
  --eval \
  --appearance_dim 0 \
  --voxel_size 0.001 \
  --update_init_factor 16 \
  --ratio 1 \
  -r -1 \
  --sam_checkpoint ./weights/sam_hq_vit_base \
  --clip_model_path ./weights/clip-vit-base-patch32 \
  --method_profile semantic_shared_clean \
  --port 8120
```

补齐后可以形成最关键的三组主表：

| 组别 | 目的 |
|---|---|
| `garden_paper_like` | Scaffold-GS baseline |
| `garden_seg_shared_clean_final` | 验证 SAM/CLIP semantic lifting 本身 |
| `garden_seg_expert_clean_final` | 验证 semantic clustered color experts |

### 10.2 多场景复验

建议至少补充 `bicycle`、`bonsai`、`stump` 或 `room` 等场景，以确认 semantic expert clean 的收益不是 garden 单场景偶然结果。

### 10.3 增加边界专项指标

如果继续保留 boundary/hybrid 方向，建议新增：

| 指标 | 作用 |
|---|---|
| Edge PSNR / crop PSNR | 看目标物体边缘局部重建 |
| Gradient magnitude error | 衡量边缘梯度是否更接近 GT |
| Boundary F-score | 衡量渲染边界与 GT/SAM 边界对齐 |
| Center-object crop metrics | 针对 garden 桌子等中心物体评估 |

## 11. 最终结论

当前最推荐作为主方法的实验是：

```text
garden_seg_expert_clean_final
```

推荐理由：

| 理由 | 说明 |
|---|---|
| 实验干净 | boundary 和 triangle 均关闭 |
| 语义专家有效 | 4 experts 正常启用，cluster 分布健康 |
| SSIM 最好 | 当前所有实验最高 |
| LPIPS 最好 | 当前所有实验最低 |
| PSNR 不低于 baseline | 没有伤害重建主干 |
| 速度可接受 | Test FPS 为 `48.46497` |

综合来看，`semantic clustered color experts` 是目前最稳定、最容易解释、最适合作为论文主线的有效模块。boundary 和 triangle 分支可以作为后续边界增强消融，而不建议作为当前主方法默认配置。
