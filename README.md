<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/rlink-dark.png">
  <img src="./img/rlink-light.png" alt="RLink logo.png" width="300px">
</picture>
</div>

# RLink

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-FFFFFF.svg?style=flat&logo=gymnasium&logoColor=black)](https://gymnasium.farama.org/)
[![tyro](https://img.shields.io/badge/tyro-FFFFFF.svg?style=flat&logo=tyro&logoColor=black)](https://brentyi.github.io/tyro/)
[![WandB](https://img.shields.io/badge/W&B-black.svg?style=flat&logo=WeightsAndBiases&logoColor=FFBE00)](https://wandb.ai/site)

RLink is a codebase for reinforcement learning research.
It is not a framework, but a collection of tools and resources for conducting research in the field.
RLink uses several tools including:

- [black](https://black.readthedocs.io/en/stable/) for code formatting
- [ruff](https://docs.astral.sh/ruff/) for code linting
- [PyTorch](https://pytorch.org/) as the deep learning library
- [Gymnasium](https://gymnasium.farama.org/) as the reinforcement learning environment API
- [tyro](https://brentyi.github.io/tyro/) for configuration
- [WandB](https://wandb.ai/site) for experiment tracking and monitoring
- [TensorBoard](https://www.tensorflow.org/tensorboard) for experiment tracking and monitoring
- [pytest](https://docs.pytest.org/en/stable/) for testing

## Get Started

See [install.md](./install.md)

## Code Quality

- **No Warnings:** The code is carefully written to make sure no warnings appear when running it. This focus on quality helps keep the code clean and free from problems caused by ignored warnings.
- **Type Hinting:** Every function and method is clearly annotated with type hints for both inputs and outputs. This makes the code easier to understand and helps prevent mistakes related to incorrect types, making the code more reliable and easier to maintain.
