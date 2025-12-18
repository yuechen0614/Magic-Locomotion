# 🚀 Magic-Locomotion

## ICCV 2025 Multi-Terrain Humanoid Locomotion Challenge - 2nd Place Solution


[![Challenge Website](https://img.shields.io/badge/Challenge-Human--Robot--Scene%20Interaction-blue?logo=iclr&logoColor=white&style=flat-square)](https://human-robot-scene.github.io/Terrain-Challenge/) [![PDF](https://img.shields.io/badge/Paper-Report%20PDF-red?logo=adobe-acrobat-reader&logoColor=white&style=flat-square)](https://github.com/yuechen0614/Magic-Locomotion/blob/main/docs/report.pdf)

### 🏆 Challenge Introduction
This repository contains the official implementation of the 2nd Place solution for the Multi-Terrain Humanoid Locomotion Challenge (ICCV 2025 Workshop on Human-Robot-Scene Interaction and Collaboration). Our method achieves state-of-the-art performance on diverse terrain locomotion tasks for humanoid robots.


### 📋 Method Overview

Our approach follows a three-stage training paradigm to efficiently transfer privileged information to a lightweight deployable policy:

**1. Teacher Policy with Privileged Information:** First, we leverage all available privileged information (full state observations, terrain priors, etc.) to train a high-performance teacher policy, following the baseline implementation from the original challenge repository.

**2. Progressive Teacher-Student Distillation:** We distill the teacher's expertise into a compact student policy using our customized distillation framework. See implementation details in: rsl_rl/rsl_rl/algorithms/distillation.py

**3. Defect-Aware Fine-Tuning:** Finally, we finetune the student policy on target terrains to optimize adaptability and final performance.

# Original Repo README

## Introduction

This package(challenging_terrain) contains 9 types of terrains (which will be continuously expanded in the future) and a large number of terrains arranged in various combinations. The basic module configuration code based on Legged Gym is provided, allowing users to achieve plug and play of terrain modules within their existing Legged Gym framework. 

This code is compatible with various robots, including but not limited to humanoid robots such as Unitree G1, Unitree H1-2, Fourier GR1-T2, Fourier GRX-N1,which will be continuously added in the future. 

Provided an online data collection module that can store trained policies in dataset format. The evaluation module is embedded in the code and only requires one parameter to quantitatively evaluate the trained policy indicators.

### Installation ###
```bash
conda create -n terrain python=3.8
conda activate terrain
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   #or cu113,cu115,cu121, based on your cuda version

git clone https://github.com/shiki-ta/Humanoid-Terrain-Bench.git
cd Humanoid-Terrain-Bench
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd rsl_rl && pip install -e .
cd legged_gym && pip install -e .
cd challenging_terrain && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
`cd legged_gym/scripts`
1. Train base policy:  
```
python train.py --exptid h1-2 --device cuda:0 --headless --task h1_2_fix
```

2. Training Recovery:
```
python train.py --exptid h1-2 --device cuda:0 --resume --resumeid=test --checkpoint=50000 --headless --task h1_2_fix
```

3. Play base policy:
```
python play.py --exptid test --task h1_2_fix
```

4. record trace as dataset

```
python record_replay.py --exptid test --save
```

### Arguments ###
- --exptid: string,  to describe the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --save: make dataset

### Acknowledgement ###

[legged_gym](https://github.com/leggedrobotics/legged_gym)

[Isaac Gym](https://developer.nvidia.com/isaac-gym)

[extreme parkour](https://github.com/chengxuxin/extreme-parkour)

### Citation
If you found any part of this code useful, please consider citing:
```
@article{fan2025one,
  title={One Policy but Many Worlds: A Scalable Unified Policy for Versatile Humanoid Locomotion},
  author={Fan, Yahao and Gui, Tianxiang and Ji, Kaiyang and Ding, Shutong and Zhang, Chixuan and Gu, Jiayuan and Yu, Jingyi and Wang, Jingya and Shi, Ye},
  journal={arXiv preprint arXiv:2505.18780},
  year={2025}
}
```
