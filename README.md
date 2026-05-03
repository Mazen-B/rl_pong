# Deep Reinforcement Learning for Atari Pong

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Run Models Individually](#run-models-individually)
   - [Run Full Pipeline](#run-full-pipeline)
   - [Selecting a Single Algorithm](#selecting-a-single-algorithm)
5. [Evaluation](#evaluation)
6. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
7. [Reproducibility](#reproducibility)
8. [Implementation Details](#implementation-details)
9. [Hardware and Training Time](#hardware-and-training-time)
10. [References](#references)
11. [License](#license)

---
## Overview
This project implements and compares two deep reinforcement learning algorithms on the Atari Pong environment (`PongNoFrameskip-v4`):
- Deep Q-Network (DQN), implemented from scratch in PyTorch
- Proximal Policy Optimization (PPO), using Stable-Baselines3

The goal is to study their learning behavior, stability, and performance under comparable conditions.

This project was developed as part of the course **Reinforcement Learning** at **IU International University of Applied Sciences**.
It accompanies a paper focused on applying and comparing Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to the Atari Pong environment.

Note that the two algorithms use slightly different preprocessing pipelines (DQN uses the custom wrappers in `src/utils/wrappers.py`, PPO uses Stable-Baselines3's `AtariWrapper` via `make_atari_env`). Both apply frame skipping, grayscale, 84x84 resizing, and 4-frame stacking, but minor defaults may differ. The comparison should therefore be considered approximate rather than strictly identical.

---
## Project Structure
```text
project_root/
├── src/
│   ├── train_dqn.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── main.py
│   └── utils/
│       ├── dqn_model.py
│       ├── replay_buffer.py
│       └── wrappers.py
├── requirements.txt
└── README.md
```
Each training run creates a new directory at the project root in the form `run_dqn_YYYY_MM_DD_T_HH_MM_SS/` or `run_ppo_YYYY_MM_DD_T_HH_MM_SS/`, containing `logs/` and `checkpoints/` subdirectories. These directories are git-ignored.

---
## Installation
```bash
pip install -r requirements.txt
```

If `ale-py` did not bundle ROMs, accept the ROM license:
```bash
AutoROM --accept-license
```
Tested with Python 3.13.

---
## Usage
You can either train and evaluate each model individually or run both sequentially.

### Run Models Individually
#### PPO
Train:
```bash
cd src
python train_ppo.py --timesteps 2000000 --seed 0
```

Evaluate the best model:
```bash
python evaluate.py \
  --model ../run_ppo_<timestamp>/checkpoints/best_model.zip \
  --episodes 10
```

#### DQN
Train:
```bash
cd src
python train_dqn.py --timesteps 2000000 --seed 0
```

Evaluate the final model (replace the filename with the actual one written by the training run):
```bash
python evaluate.py \
  --model ../run_dqn_<timestamp>/checkpoints/dqn_PongNoFrameskip-v4_seed0_final.pt \
  --episodes 10
```

### Run Full Pipeline
Run both models sequentially (train + evaluate):
```bash
cd src
python main.py --timesteps 2000000 --seed 0 --episodes 10
```

This executes:
- PPO training
- PPO evaluation (best model)
- DQN training
- DQN evaluation

### Selecting a Single Algorithm
You can also use `main.py` to run a single algorithm:
```bash
python main.py --algo ppo --timesteps 2000000
python main.py --algo dqn --timesteps 2000000
```

---
## Evaluation
Evaluation runs the trained model on a fixed number of episodes and reports:
- Mean reward
- Standard deviation
- Minimum and maximum
- Median

Results are also written as JSON to the run directory (`evaluation.json`).

Example output:
```text
Evaluation summary
mean  : +20.50
std   : 0.50
min   : +20.0
max   : +21.0
median: +21.0
```

Evaluation is performed with:
- No reward clipping
- Deterministic policy (no exploration)
- Fresh environment instances

The following results were obtained using seed=0 and 2M timesteps:

| Algorithm | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| DQN      | +20.4 | 0.49 | 20 | 21 |
| PPO      | +19.7 | 0.46 | 19 | 20 |

Full evaluation files are available in `results/`.

---
## Monitoring with TensorBoard
While or after training:
```bash
tensorboard --logdir run_ppo_<timestamp>/logs
tensorboard --logdir run_dqn_<timestamp>/logs
```

Or point at the project root to compare runs:
```bash
tensorboard --logdir .
```

---
## Reproducibility
- All experiments use a fixed random seed for Python, NumPy, and PyTorch.
- `cudnn.deterministic` is enabled for DQN, but full determinism on GPU is not guaranteed (cuDNN benchmarking, async kernel launches, and non-deterministic ops can still introduce small variance).
- Training parameters are stored in DQN checkpoints under the `config` key. PPO checkpoints (`.zip`) do not store the env-id; pass `--env-id` explicitly to `evaluate.py` if you trained on a non-default env.
- Each run is isolated in a timestamped directory.
- Evaluation can be repeated on any saved model.

For stronger empirical results, multiple seeds are recommended.

---
## Implementation Details
### Environment
- `PongNoFrameskip-v4`
- Frame skipping: 4
- Frame stacking: 4
- Grayscale conversion and resizing to 84x84

### DQN
- Experience replay buffer (`uint8` storage to keep memory usage low)
- Target network with periodic hard updates
- Epsilon-greedy exploration with linear decay
- Huber loss (`smooth_l1_loss`)
- Gradient clipping

### PPO
- Stable-Baselines3 implementation
- Vectorized environments (`n_envs = 8`)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Best model saved automatically via `EvalCallback`

### Logging
- TensorBoard logs for training metrics
- Periodic checkpoints
- Best model automatically saved (PPO)
- Evaluation results saved as `evaluation.json`

---
## Hardware and Training Time
2,000,000 timesteps is on the order of a few hours on a single mid-range GPU (e.g. RTX 3060) for both PPO and DQN. CPU-only training is feasible but several times slower. Memory usage is dominated by the DQN replay buffer (`buffer_size * 4 * 84 * 84` bytes for stored `uint8` frames; with the default `buffer_size=250_000` this is roughly 7 GB for state plus 7 GB for next_state).

---
## References
- Mnih et al. (2015), Human-level control through deep reinforcement learning
- Schulman et al. (2017), Proximal Policy Optimization Algorithms
- Sutton and Barto (2018), Reinforcement Learning: An Introduction
- Raffin et al. (2021), Stable-Baselines3

---
## License
MIT.