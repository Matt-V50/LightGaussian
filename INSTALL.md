#!/usr/bin/env fish
# ============================================================
# LightGaussian 环境配置脚本 (Fish Shell)
# 基于 3DGS + LightGaussian 压缩优化
# Paper: "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
# ============================================================

# 环境名称
set ENV_NAME "lightgs"

echo "=========================================="
echo "开始配置 LightGaussian 环境"
echo "=========================================="

# 创建conda环境
echo "创建 Python 3.10.12 环境..."
conda create -y -n $ENV_NAME python=3.10.12
conda activate $ENV_NAME

# ============================================================
# PyTorch + CUDA 12.1 (保持与你的版本一致)
# ============================================================
echo "安装 PyTorch 2.4.1 + CUDA 12.1..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia --yes

# 安装 CUDA toolkit
conda install cuda-toolkit -c nvidia/label/cuda-12.1.0 --yes

# 解决常见编译问题
echo "安装编译工具链..."
conda install mkl==2023.1.0 mkl-include -c conda-forge --yes
conda install cuda-cudart=12.1.55 -c nvidia/label/cuda-12.1.0 --yes

# ============================================================
# 3DGS 基础依赖
# ============================================================
echo "安装 3DGS 基础依赖..."
pip install plyfile==1.0.3 \
    tqdm \
    opencv-python==4.10.0.84 \
    pillow==11.0.0 \
    scikit-image \
    imageio \
    lpips

# ============================================================
# LightGaussian 特定依赖
# ============================================================
echo "安装 LightGaussian 额外依赖..."
pip install tensorboard \
    matplotlib \
    numpy==1.26.4 \
    icecream \
    scipy

set -x CUDA_HOME $CONDA_PREFIX
pip install --no-build-isolation \    
    submodules/compress-diff-gaussian-rasterization \
    submodules/simple-knn

echo "CUDA 扩展编译完成！"

# ============================================================
# 验证安装
# ============================================================
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "LightGaussian 环境配置完成!"
echo "=========================================="