from __future__ import annotations

import time
import torch
import random
import ale_py
import argparse
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim

from pathlib import Path
from datetime import datetime
from utils.dqn_model import DQN
from dataclasses import asdict, dataclass
from utils.wrappers import make_atari_env
from utils.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(ale_py)

@dataclass
class DQNConfig:
    env_id: str = "PongNoFrameskip-v4"
    total_timesteps: int = 2_000_000
    buffer_size: int = 250_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    learning_starts: int = 50_000
    train_frequency: int = 4
    target_update_frequency: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 1_000_000
    grad_clip_norm: float = 10.0
    seed: int = 0
    save_frequency: int = 250_000

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch.
    Note: full determinism on GPU is not guaranteed even with these settings —
    cuDNN benchmarking, non-deterministic ops, and async kernel launches can
    still introduce variance across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    frac = min(step / max(duration, 1), 1.0)
    return start + frac * (end - start)

def select_action(model: DQN, state: np.ndarray, epsilon: float, num_actions: int, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        q_values = model(torch.from_numpy(state).unsqueeze(0).to(device))
        return int(q_values.argmax(dim=1).item())

def train(cfg: DQNConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_date = datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
    root_dir = Path(__file__).resolve().parent.parent
    run_dir = root_dir / f"run_dqn_{run_date}"

    log_path = run_dir / "logs"
    checkpoint_path = run_dir / "checkpoints"
    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    safe_env_id = cfg.env_id.replace("/", "_")
    run_name = f"dqn_{safe_env_id}_seed{cfg.seed}"

    writer = SummaryWriter(log_path)
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()))

    print(f"[DQN] device={device} env={cfg.env_id}")
    print(f"[DQN] config={asdict(cfg)}")

    env = make_atari_env(cfg.env_id, seed=cfg.seed, clip_rewards=True, episodic_life=True)
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    online_net = DQN(in_channels=obs_shape[0], num_actions=num_actions).to(device)
    target_net = DQN(in_channels=obs_shape[0], num_actions=num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_size, obs_shape, device=device)

    state, _ = env.reset(seed=cfg.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    recent_rewards: list[float] = []
    start_time = time.time()

    for step in range(1, cfg.total_timesteps + 1):
        epsilon = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
        action = select_action(online_net, state, epsilon, num_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            recent_rewards.append(episode_reward)
            recent_rewards = recent_rewards[-100:]
            writer.add_scalar("rollout/episode_reward_clipped_train", episode_reward, step)
            writer.add_scalar("rollout/episode_length", episode_length, step)
            writer.add_scalar("rollout/mean_reward_100", float(np.mean(recent_rewards)), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)

            if episode_count % 10 == 0:
                fps = step / max(time.time() - start_time, 1e-6)
                print(f"[step {step:>8d}] ep={episode_count:>5d} "
                    f"reward={episode_reward:>6.1f} mean100={np.mean(recent_rewards):>6.2f} "
                    f"eps={epsilon:.3f} fps={fps:.0f} buffer={len(buffer)}")

            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_count += 1

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0 and len(buffer) >= cfg.batch_size:
            batch = buffer.sample(cfg.batch_size)
            with torch.no_grad():
                next_q = target_net(batch.next_state).max(dim=1).values
                td_target = batch.reward + cfg.gamma * next_q * (1.0 - batch.done)

            q_values = online_net(batch.state).gather(1, batch.action.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_values, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            if step % 1_000 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/q_mean", q_values.mean().item(), step)
                writer.add_scalar("train/td_target_mean", td_target.mean().item(), step)

        if step % cfg.target_update_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

        if step % cfg.save_frequency == 0:
            ckpt_path = checkpoint_path / f"{run_name}_step{step}.pt"
            torch.save(
                {"step": step,
                "model_state": online_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(cfg)},
                ckpt_path)
            print(f"[DQN] saved {ckpt_path}")

    final_path = checkpoint_path / f"{run_name}_final.pt"
    torch.save({"model_state": online_net.state_dict(), "config": asdict(cfg)}, final_path)
    print(f"[DQN] saved {final_path}")
    env.close()
    writer.close()
    return final_path

def parse_args() -> DQNConfig:
    p = argparse.ArgumentParser(description="Train custom DQN Pong agent")
    p.add_argument("--env-id", default="PongNoFrameskip-v4")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=250_000)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    return DQNConfig(
        env_id=args.env_id,
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size)

if __name__ == "__main__":
    train(parse_args())
