#!/bin/bash

# 如果任何命令失败，立即退出脚本
set -e

# --- 环境设置 ---
ENV_NAME="inpo"
PYTHON_VERSION="3.10"

echo ">>> [步骤 1/10] 创建 Conda 环境: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ">>> [步骤 2/10] 安装 PyTorch, Torchvision, Torchaudio..."
conda run -n $ENV_NAME pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo ">>> [步骤 3/10] 安装 vllm..."
conda run -n $ENV_NAME pip install vllm==0.8.5

echo ">>> [步骤 4/10] 安装 transformers..."
conda run -n $ENV_NAME pip install transformers==4.53.1

echo ">>> [步骤 5/10] 安装 datasets..."
conda run -n $ENV_NAME pip install datasets==4.0.0

echo ">>> [步骤 6/10] 安装 flash-attention..."
conda run -n $ENV_NAME pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo ">>> [步骤 7/10] 安装 deepspeed..."
conda run -n $ENV_NAME pip install deepspeed==0.17.2

echo ">>> [步骤 8/10] 安装 huggingface-hub..."
conda run -n $ENV_NAME pip install huggingface-hub==0.33.2

echo ">>> [步骤 9/10] 安装 flashinfer-python..."
conda run -n $ENV_NAME pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

echo ">>> [步骤 10/10] 安装 more_itertools..."
conda run -n $ENV_NAME pip install more_itertools

echo ""
echo "✅ 全部安装成功！所有命令已按指定顺序串行执行完毕。"
echo "现在，请手动激活环境来开始使用:"
echo "conda activate $ENV_NAME"