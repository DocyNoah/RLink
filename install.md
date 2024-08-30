# Installation

## Create a conda environment

```bash
conda create -n rlink python=3.10 -y
conda activate rlink
```

## Install requirements

Install PyTorch 2.4.0

- <https://pytorch.org/get-started/locally/#start-locally>

Install requirements

```bash
pip install -r requirements.txt
```

## Additional requirements

Classic Control Rendering

```bash
pip install "gymnasium[classic-control]"
```

Atari

```bash
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"
pip install opencv-python
```

MuJoCo

```bash
pip install "gymnasium[mujoco]"
```

Record video

- for Mac

```bash
pip install moviepy==1.0.3
brew install ffmpeg
```

- for Ubuntu

```bash
pip install moviepy==1.0.3
sudo apt-get install ffmpeg
```
