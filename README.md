# Scaffold-GS + SAM-HQ + CLIP 语义集成指南
# 本项目在 Scaffold-GS 的基础上集成了 SAM-HQ（高精度分割模型）和 CLIP（开放词汇语义模型），实现了将 2D 图像的语义特征提升 (Lifting) 至 3D 锚点 (Anchors) 的功能。

# 1. 环境准备
# 除了 Scaffold-GS 原有的依赖外，还需要安装处理 SAM 和 CLIP 所需的库
pip install transformers segment-anything-hq torch-scatter plyfile
# 基础训练命令
CUDA_VISIBLE_DEVICES=4 python seg_train.py     -s ./data/garden     -m ./output/garden_semantic     --sam_checkpoint ./weights/sam_hq_vit_base     --iterations 30000     --port 7000
