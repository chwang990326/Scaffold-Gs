# 1. 清理 conda 缓存
conda clean --all -y

# 2. 创建并激活环境
# Linux 下通常不需要强制指定 64 位，默认即是。
conda create -n scaffold_gs python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh # 确保脚本中能使用 conda activate
conda activate scaffold_gs

# 3. 安装依赖 (保持与你 Windows 脚本一致的 torch 版本)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja

# 4. 编译子模块
# Linux 不需要运行 VS 环境，但请确保已安装 build-essential (sudo apt install build-essential)

# 如果之前尝试编译失败，清理缓存
cd submodules/diff-gaussian-rasterization
rm -rf build dist *.egg-info
cd ../simple-knn
rm -rf build dist *.egg-info
cd ../..

# 设置环境变量并安装
export DISTUTILS_USE_SDK=1
# 如果你想手动指定架构可以写 export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
# 留空则让 PyTorch 自动检测当前显卡架构
export TORCH_CUDA_ARCH_LIST=""

# 安装第一个子模块
cd submodules/diff-gaussian-rasterization
python setup.py install

# 安装第二个子模块
cd ../simple-knn
python setup.py install

cd ../..
echo "安装完成！"