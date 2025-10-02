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

TE
```
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html
```

### Dataset Makefile Copy
Copy Makefile code from github to `megatron.core.datasets.Makefile`


### Deps
```
cuda-driver==12.2
cuda-toolkit==12.4
cudnn==9.13.1.26-1 (12.x)
```
