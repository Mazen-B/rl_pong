from __future__ import annotations

import json
import torch
import ale_py
import argparse
import numpy as np
import gymnasium as gym

from pathlib import Path
from datetime import datetime
from utils.dqn_model import DQN
from stable_baselines3 import PPO
from utils.wrappers import make_atari_env
from stable_baselines3.common.env_util import make_atari_env as sb3_make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

gym.register_envs(ale_py)

def evaluate_dqn(model_path: Path, episodes: int, seed: int, env_id: str | None, render: bool) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    env_id = env_id or cfg.get("env_id", "PongNoFrameskip-v4")

    env = make_atari_env(env_id=env_id, seed=seed,
        clip_rewards=False, episodic_life=False,
        render_mode="human" if render else None)
    num_actions = env.action_space.n
    model = DQN(in_channels=4, num_actions=num_actions).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores: list[float] = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                q = model(torch.from_numpy(state).unsqueeze(0).to(device))
                action = int(q.argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
        scores.append(total_reward)
        print(f"Episode {ep + 1:02d}/{episodes}: {total_reward:+.1f}")
    env.close()
    return scores

def evaluate_ppo(model_path: Path, episodes: int, seed: int, env_id: str, render: bool) -> list[float]:
    # PPO checkpoints (zip) do not store the training env-id. Caller must pass the same
    # env-id used at training. Default is PongNoFrameskip-v4.
    env = sb3_make_atari_env(env_id, n_envs=1, seed=seed,
        wrapper_kwargs={"clip_reward": False},
        env_kwargs={"render_mode": "human"} if render else None)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path, env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        custom_objects={"clip_range": 0.1,
            "lr_schedule": lambda _: 2.5e-4})
    scores: list[float] = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = env.step(action)
            total_reward += float(reward[0])
            done = bool(dones[0])
        scores.append(total_reward)
        print(f"Episode {ep + 1:02d}/{episodes}: {total_reward:+.1f}")
    env.close()
    return scores

def save_results(model_path: Path, scores: list[float]) -> None:
    run_dir = model_path.parent.parent  # checkpoints/ -> run_dir
    out_file = run_dir / "evaluation.json"

    data = {"model": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores))}

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved evaluation to {out_file}")

def report(scores: list[float]) -> None:
    arr = np.asarray(scores, dtype=np.float32)
    print("\nEvaluation summary")
    print(f"mean  : {arr.mean():+.2f}")
    print(f"std   : {arr.std():.2f}")
    print(f"min   : {arr.min():+.1f}")
    print(f"max   : {arr.max():+.1f}")
    print(f"median: {np.median(arr):+.1f}")

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Pong agent")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-id", default=None,
        help="Must match the env-id used at training. For DQN the env-id is read from the "
             "checkpoint if not given. For PPO it defaults to PongNoFrameskip-v4 — pass "
             "--env-id explicitly if you trained on a different env.")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    if args.model.suffix == ".pt":
        scores = evaluate_dqn(args.model, args.episodes, args.seed, args.env_id, args.render)
    elif args.model.suffix == ".zip":
        if args.env_id is None:
            print("[warn] --env-id not given for PPO checkpoint; defaulting to PongNoFrameskip-v4. "
                "If you trained on a different env, pass --env-id.")
        scores = evaluate_ppo(args.model, args.episodes, args.seed,
            args.env_id or "PongNoFrameskip-v4", args.render)
    else:
        raise ValueError("Use a .pt DQN checkpoint or .zip PPO checkpoint")
    report(scores)
    save_results(args.model, scores)

if __name__ == "__main__":
    main()
