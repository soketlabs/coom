# COOM Training Framework

## Overview
A training framework for large-scale language models based on Megatron-Core, the COOM Training Framework is designed to efficiently handle extensive model training inspired by Deepseek's HAI-LLM optimizations. 

This framework is planned to support state-of-the-art innovations essential for achieving high-performance language modeling, particularly targeting efficiency and scalability for large models.

## Key Objectives
- Develop an optimized pretraining framework based on Deepseek's HAI-LLM.
- Enable efficient scaling of large multilingual language models.
- Incorporate cutting-edge optimizations to maximize training efficiency.

## Planned Features

| Feature                                |
|----------------------------------------|
| FP8 Pretraining                        |
| Mixture of Experts (MoE)               |
| Multi-Head Latent Attention (MLA)      |
| Multi-Token Prediction                 |
| Kernel Fusion                          |
| KV-Cache Optimisation                  |
| Load Balancing Experts                 |
| Progressive Model Expansion            |
| Energy-Efficient Training              |

## Future Directions
- Continuous improvements and feedback loops for better alignment and model robustness.
- Expansion into multilingual and multimodal capabilities.
- Further optimization for deployment in edge computing scenarios.

## Collaboration and Contribution
We welcome researchers and developers to contribute to the ongoing development of COOM Training Framework. Regular updates, comprehensive documentation, and open-source contributions are encouraged to foster community-driven improvements.

To ensure consistency and maintain code quality, all code contributions must **strictly follow [PEP 8](https://peps.python.org/pep-0008/)**. 

For more details or to get involved, please contact our team.

---

**Note:** This framework is currently under active development. Performance metrics and benchmarks will be shared in upcoming releases.

## Dev Installation

### Deps
```sh
sudo apt-get update
sudo apt-get install cmake python3-dev
```

```sh
uv sync
```

Nemo has some breaking deps so install it with `uv pip ...`
```sh
uv pip install nemo_toolkit[all]
```

Cuda-Toolkit [12.8]
```sh
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --toolkitpath=/cm/shared/cuda/12.8
```

CuDNN
```sh
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
sudo mkdir -p /cm/shared/cuda_dnn/9.14_12
sudo mv cudnn-linux-x86_64-9.14.0.64_cuda12-archive /cm/shared/cuda_dnn/9.14_12
sudo chmod -R a+rX /cm/shared/cuda_dnn/9.14_12

```

NCCL Install
```sh
git clone https://github.com/NVIDIA/nccl.git
cd nccl
sudo make src.build BUILDDIR=/cm/shared/apps/nccl/
```

TE
```sh
# https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html
uv pip install --no-build-isolation transformer_engine[pytorch]
```

### Dataset Makefile Copy
Copy Makefile code from github to `megatron.core.datasets.Makefile`


### Deps
```
cuda-driver==12.2
cuda-toolkit==12.4
cudnn==9.13.1.26-1 (12.x)
```

### Creating a Modulefile in the HPC Cluster
```sh
sudo mkdir -p /cm/shared/modulefiles/cuda
sudo nano /cm/shared/modulefiles/cuda/12.8
```

```sh
#%Module1.0
proc ModulesHelp { } {
  puts stderr "Sets up environment for CUDA 12.8"
}
module-whatis "sets up environment for CUDA 12.8"

set version 12.8
set root /cm/shared/cuda/$version

setenv CUDA_HOME $root
prepend-path PATH $root/bin
prepend-path LD_LIBRARY_PATH $root/lib64
prepend-path LD_LIBRARY_PATH $root/extras/CUPTI/lib64

# The 'conflict' directive prevents multiple versions from being loaded simultaneously.
conflict cuda
```

If nccl, cuda, python-dev already installed then:
```
module load gcc
module load cuda
module load nccl
module load python/3.10
```

Install transformer engine:
```
uv pip install --no-build-isolation transformer_engine[pytorch]
```

## Apex Installation
Install Apex from https://github.com/NVIDIA/apex
using 
```
source <DIR>/.venv/bin/activate
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .
```