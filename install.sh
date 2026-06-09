#!/usr/bin/env bash
set -eo pipefail
# command to install this environment: source init.sh

# install miniconda3 if not installed yet.
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

OS_NAME="$(uname -s)"
IS_MAC=0
if [ "${OS_NAME}" = "Darwin" ]; then
  IS_MAC=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

# Ensure conda activate/deactivate works in non-interactive shells.
eval "$(conda shell.bash hook)"

# CUDA / compiler module setup for cluster machines.
# Important:
# This script intentionally does NOT use system CUDA 13.0.
# PointNeXt/OpenPoints with torch 1.10.1 is tied to CUDA 11.3.
if [ "${IS_MAC}" -eq 0 ]; then
  # CUDA 11.3 does not know Ada Lovelace sm_89; PTX for 8.6 works on RTX 4090.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

  if command -v module >/dev/null 2>&1; then
    module purge

    # Only load these if your cluster actually provides them.
    # If not, the conda CUDA/compiler packages below are used.
    module load cuda/11.3 || true
    module load gcc/9.0 || true
  fi
fi

# Download/update OpenPoints submodules.
git submodule update --init --recursive

# Optional: update submodules to latest remote commit.
# Enable with: UPDATE_SUBMODULES_REMOTE=1 bash init.sh
if [ "${UPDATE_SUBMODULES_REMOTE:-0}" = "1" ]; then
  git submodule update --remote --merge
fi

# Recreate environment.
conda deactivate || true
conda env remove --name openpoints -y || true

if [ "${IS_MAC}" -eq 1 ]; then
  # macOS: old Linux pins are not suitable.
  conda create -n openpoints -y python=3.10 numpy numba -c conda-forge
else
  # Python 3.8 is required for newer wandb.
  # numpy=1.20 keeps compatibility with the older OpenPoints stack.
  conda create -n openpoints -y python=3.8 numpy=1.20 numba -c conda-forge
fi

conda activate openpoints

python -m pip install --upgrade pip setuptools wheel

# Install PyTorch.
if [ "${IS_MAC}" -eq 1 ]; then
  # macOS: CPU/MPS build, no CUDA toolkit.
  conda install -y pytorch torchvision -c pytorch
  echo "Skipping torch-scatter CUDA wheel on macOS."
else
  # IMPORTANT:
  # Do not install CUDA 13.0 here.
  # PyTorch 1.10.1 must use CUDA 11.3-compatible packages.
  conda install -y \
    cuda-nvcc=11.3 \
    -c nvidia/label/cuda-11.3.0 \
    -c nvidia \
    -c conda-forge

  # Torch 1.10.x can break with modern MKL.
  conda install -y "mkl<2024" pip -c conda-forge

  # Compiler stack for native C++/CUDA extensions.
  conda install -y \
    gcc_linux-64=9 \
    gxx_linux-64=9 \
    kernel-headers_linux-64=3.10.0 \
    sysroot_linux-64=2.17 \
    libxcrypt \
    -c conda-forge

  # CUDA compiler/dev components.
  # This replaces conda-forge cudatoolkit-dev, which failed in your install.
  conda install -y \
    cuda-nvcc=11.3 \
    cuda-cudart-dev=11.3 \
    cuda-libraries-dev=11.3 \
    -c nvidia \
    -c conda-forge

  # torch-scatter wheel for:
  # torch 1.10.x + CUDA 11.3 + Python 3.8
  pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl

  # Newer wandb needed for 86-character W&B API keys.
  pip install "wandb==0.22.3"
fi

# Install Python requirements.
if [ "${IS_MAC}" -eq 1 ]; then
  # h5py==3.6.0 often has no wheel for modern macOS arm64/Python and falls back to source build.
  TMP_REQ="$(mktemp)"
  sed 's/^h5py==3\.6\.0$/h5py>=3.10.0/' requirements.txt > "${TMP_REQ}"
  pip install -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
else
  pip install -r requirements.txt
fi

# Build native extensions.
if [ "${IS_MAC}" -eq 1 ]; then
  echo "Skipping CUDA-based C++/CUDA extensions on macOS: pointnet2_batch, pointops, chamfer_dist, emd."
else
  # Use conda CUDA/compiler, not /usr/local/cuda-13.0.
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

  echo "Python:"
  python --version

  echo "PyTorch / CUDA check:"
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

  echo "nvcc:"
  nvcc --version || true

  # pointnet++ library
  cd openpoints/cpp/pointnet2_batch
  python setup.py install
  cd ../

  # grid_subsampling library; necessary only for S3DIS_sphere
  cd subsampling
  python setup.py build_ext --inplace
  cd ..

  # point transformer library; necessary for Point Transformer and Stratified Transformer
  cd pointops
  python setup.py install
  cd ..

  # Optional reconstruction-task extensions: completion/chamfer/emd
  cd chamfer_dist
  python setup.py install --user
  cd ../emd
  python setup.py install --user
  cd ../../../
fi

echo "Installation finished."

echo "Final checks:"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

if command -v wandb >/dev/null 2>&1; then
  wandb --version
fi

echo ""
echo "To log in to W&B, run:"
echo "wandb login --relogin"