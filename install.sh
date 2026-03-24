#!/usr/bin/env bash
set -euo pipefail
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc


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

# The following lines are only for slurm/CUDA machines.
if [ "${IS_MAC}" -eq 0 ]; then
  export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
  if command -v module >/dev/null 2>&1; then
    module purge
    module load cuda/11.1.1
    module load gcc/7.5.0
  fi
fi

# download openpoints
# git submodule add git@github.com:guochengqian/openpoints.git
git submodule update --init --recursive

# Optional: update submodules to latest remote commit.
# Enable with: UPDATE_SUBMODULES_REMOTE=1 bash install.sh
if [ "${UPDATE_SUBMODULES_REMOTE:-0}" = "1" ]; then
  git submodule update --remote --merge
fi

# install PyTorch
conda deactivate || true
conda env remove --name openpoints -y || true
if [ "${IS_MAC}" -eq 1 ]; then
  # osx-arm64 no longer provides old python/numpy builds used by the original Linux setup.
  conda create -n openpoints -y python=3.10 numpy numba
else
  conda create -n openpoints -y python=3.7 numpy=1.20 numba
fi
conda activate openpoints

# please always double check installation for pytorch and torch-scatter from the official documentation
if [ "${IS_MAC}" -eq 1 ]; then
  # macOS: install CPU/MPS build (no CUDA toolkit)
  conda install -y pytorch torchvision -c pytorch
  echo "Skipping torch-scatter CUDA wheel on macOS."
else
  conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
  pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
fi

if [ "${IS_MAC}" -eq 1 ]; then
  # h5py==3.6.0 often has no wheel for modern macOS arm64/Python and falls back to source build.
  TMP_REQ="$(mktemp)"
  sed 's/^h5py==3\.6\.0$/h5py>=3.10.0/' requirements.txt > "${TMP_REQ}"
  pip install -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
else
  pip install -r requirements.txt
fi

if [ "${IS_MAC}" -eq 1 ]; then
  echo "Skipping CUDA-based C++/CUDA extensions on macOS (pointnet2_batch, pointops, chamfer_dist, emd)."
else
  # install cpp extensions, the pointnet++ library
  cd openpoints/cpp/pointnet2_batch
  python setup.py install
  cd ../

  # grid_subsampling library. necessary only if interested in S3DIS_sphere
  cd subsampling
  python setup.py build_ext --inplace
  cd ..

  # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
  cd pointops/
  python setup.py install
  cd ..

  # Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
  cd chamfer_dist
  python setup.py install --user
  cd ../emd
  python setup.py install --user
  cd ../../../
fi
