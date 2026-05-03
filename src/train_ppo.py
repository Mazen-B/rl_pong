from __future__ import annotations

import torch
import ale_py
import argparse
import gymnasium as gym

from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from dataclasses import asdict, dataclass
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

gym.register_envs(ale_py)

@dataclass
class PPOConfig:
    env_id: str = "PongNoFrameskip-v4"
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    save_frequency: int = 100_000

def train(cfg: PPOConfig) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_date = datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
    root_dir = Path(__file__).resolve().parent.parent
    run_dir = root_dir / f"run_ppo_{run_date}"

    log_path = run_dir / "logs"
    checkpoint_path = run_dir / "checkpoints"

    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    safe_env_id = cfg.env_id.replace("/", "_")
    run_name = f"ppo_{safe_env_id}_seed{cfg.seed}"

    print(f"[PPO] device={device} env={cfg.env_id} n_envs={cfg.n_envs}")
    print(f"[PPO] config={asdict(cfg)}")

    env = make_atari_env(cfg.env_id,
        n_envs=cfg.n_envs, seed=cfg.seed,
        wrapper_kwargs={"clip_reward": True})
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)

    eval_env = make_atari_env(cfg.env_id,
        n_envs=1, seed=cfg.seed + 10_000,
        wrapper_kwargs={"clip_reward": False})
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecMonitor(eval_env)

    model = PPO(policy="CnnPolicy", env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=str(log_path),
        verbose=1, seed=cfg.seed,
        device=device)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg.save_frequency // cfg.n_envs, 1),
        save_path=str(checkpoint_path),
        name_prefix=run_name)
    eval_cb = EvalCallback(eval_env,
        best_model_save_path=str(checkpoint_path),
        log_path=str(log_path),
        eval_freq=max(50_000 // cfg.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False)

    model.learn(total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=run_name,
        progress_bar=True)

    final_path = checkpoint_path / f"{run_name}_final.zip"
    model.save(final_path)
    print(f"[PPO] saved {final_path}")

    env.close()
    eval_env.close()
    return final_path

def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="Train PPO Pong agent")
    p.add_argument("--env-id", default="PongNoFrameskip-v4")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2.5e-4)
    args = p.parse_args()
    return PPOConfig(
        env_id=args.env_id,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.lr)

if __name__ == "__main__":
    train(parse_args())
