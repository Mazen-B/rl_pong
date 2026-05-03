from __future__ import annotations

import argparse
import traceback

from train_dqn import DQNConfig, train as train_dqn
from train_ppo import PPOConfig, train as train_ppo
from evaluate import evaluate_dqn, evaluate_ppo, report, save_results

def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate PPO and DQN on Pong")
    p.add_argument("--env-id", default="PongNoFrameskip-v4")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    p.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both",
        help="Choose which algorithm to train and evaluate")
    args = p.parse_args()

    if args.algo in ("ppo", "both"):
        try:
            print("\n========== TRAIN PPO ==========")
            ppo_cfg = PPOConfig(
                env_id=args.env_id,
                total_timesteps=args.timesteps,
                seed=args.seed)
            ppo_final_path = train_ppo(ppo_cfg)

            ppo_run_dir = ppo_final_path.parent.parent
            ppo_best_path = ppo_run_dir / "checkpoints" / "best_model.zip"

            if not ppo_best_path.exists():
                raise FileNotFoundError(f"PPO best model not found: {ppo_best_path}")

            print("\n========== EVALUATE PPO BEST MODEL ==========")
            ppo_scores = evaluate_ppo(
                model_path=ppo_best_path,
                episodes=args.episodes,
                seed=args.seed,
                env_id=args.env_id,
                render=args.render)
            report(ppo_scores)
            save_results(ppo_best_path, ppo_scores)
        except Exception:
            print("[PPO] pipeline failed:")
            traceback.print_exc()
            if args.algo == "ppo":
                raise

    if args.algo in ("dqn", "both"):
        try:
            print("\n========== TRAIN DQN ==========")
            dqn_cfg = DQNConfig(
                env_id=args.env_id,
                total_timesteps=args.timesteps,
                seed=args.seed)
            dqn_final_path = train_dqn(dqn_cfg)

            print("\n========== EVALUATE DQN FINAL MODEL ==========")
            dqn_scores = evaluate_dqn(
                model_path=dqn_final_path,
                episodes=args.episodes,
                seed=args.seed,
                env_id=args.env_id,
                render=args.render)
            report(dqn_scores)
            save_results(dqn_final_path, dqn_scores)
        except Exception:
            print("[DQN] pipeline failed:")
            traceback.print_exc()
            if args.algo == "dqn":
                raise

if __name__ == "__main__":
    main()
