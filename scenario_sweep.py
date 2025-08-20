"""
Scenario Sweep: Evaluate trained PPO model across diverse cluster layouts.

Usage:
  python scenario_sweep.py            # auto-detect latest model
  python scenario_sweep.py <model.zip>
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from proactive_navigation_env import ProactiveNavigationEnv


def find_latest_model_path(base_dir: str = ".") -> Tuple[str, str]:
    training_dirs = [d for d in os.listdir(base_dir)
                     if d.startswith("proactive_navigation_training_")
                     and os.path.isdir(os.path.join(base_dir, d))]
    if not training_dirs:
        raise FileNotFoundError("No training dirs found (proactive_navigation_training_*)")

    def parse_ts(name: str) -> datetime:
        s = name.replace("proactive_navigation_training_", "")
        try:
            return datetime.strptime(s, "%Y%m%d_%H%M%S")
        except Exception:
            return datetime.min

    candidates: List[Tuple[datetime, int, str, str]] = []
    for d in training_dirs:
        ts = parse_ts(d)
        root = os.path.join(base_dir, d)
        final_p = os.path.join(root, "final_model.zip")
        if os.path.exists(final_p):
            candidates.append((ts, 3, d, final_p))
        best_p = os.path.join(root, "best_model", "best_model.zip")
        if os.path.exists(best_p):
            candidates.append((ts, 2, d, best_p))
        ckpt_dir = os.path.join(root, "checkpoints")
        if os.path.isdir(ckpt_dir):
            zips = [f for f in os.listdir(ckpt_dir) if f.endswith('.zip')]
            if zips:
                latest_ck = max(zips, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                candidates.append((ts, 1, d, os.path.join(ckpt_dir, latest_ck)))
    if not candidates:
        raise FileNotFoundError("No model files found in training dirs")
    ts, pr, d, path = max(candidates, key=lambda t: (t[0], t[1]))
    return d, path


def run_episode(env: ProactiveNavigationEnv, model: PPO, max_steps: int = 2000):  # Daha fazla adım
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            return info, total_reward
    return info, total_reward


def main():
    parser = argparse.ArgumentParser(description="Scenario sweep and optional rendering")
    parser.add_argument("model", nargs="?", default=None, help="Path to model.zip (optional)")
    parser.add_argument("--render", action="store_true", help="Render a single interactive scenario instead of sweep")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters for render mode")
    parser.add_argument("--safe", type=float, default=40.0, help="Safe margin for render mode")
    parser.add_argument("--goalR", type=float, default=5.0, help="Goal radius for render mode (smaller = more precise)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for render mode")
    parser.add_argument("--assist", action="store_true", help="Enable maneuver assist steering in render mode")
    args = parser.parse_args()

    # Resolve model path
    if args.model and os.path.exists(args.model):
        model_path = args.model
    else:
        _, model_path = find_latest_model_path()
    print(f"🧠 Using model: {model_path}")

    model = PPO.load(model_path)

    # Interactive render mode
    if args.render:
        env = ProactiveNavigationEnv()
        env.dynamic_difficulty = False
        env.allow_los_success = False  # Erken başarıyı devre dışı bırak
        env.set_curriculum(curriculum_num_clusters=args.clusters, safe_margin=args.safe, goal_radius=args.goalR, max_steps=2000)  # Daha fazla adım
        obs, info = env.reset(seed=args.seed)
        total_reward = 0.0
        steps = 0
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)

                # Optional maneuver assist for decisive turns
                if args.assist:
                    try:
                        if hasattr(env, '_is_maneuver_needed') and env._is_maneuver_needed():
                            tx, ty = env._get_maneuver_target()
                        else:
                            tx, ty = env.goal[0], env.goal[1]
                        desired = np.arctan2(ty - env.y, tx - env.x)
                        bearing_err = desired - env.heading
                        while bearing_err > np.pi:
                            bearing_err -= 2*np.pi
                        while bearing_err < -np.pi:
                            bearing_err += 2*np.pi
                        max_delta = env.max_turn_rate * env.dt
                        turn_norm = float(np.clip(1.4 * (bearing_err / max_delta), -1.0, 1.0))
                        if isinstance(action, np.ndarray) and action.shape[0] >= 2:
                            action[0] = turn_norm
                            action[1] = min(float(action[1]), -0.2)
                    except Exception:
                        pass
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                env.render()
                if terminated or truncated or steps >= 2500:  # Daha fazla adım - erken kapanmayı önle
                    print(f"Episode finished: success={info.get('success', False)}, type={info.get('episode_type')}, steps={steps}, total_reward={total_reward:.1f}")
                    print("🎯 Simülasyon tamamlandı! Ekranı kapatmak için pencereyi kapatın.")
                    # Ekranı açık tut - kullanıcı kapatana kadar
                    while True:
                        env.render()
                        plt.pause(0.1)
                        # Pencere kapatıldıysa çık
                        if not plt.get_fignums():
                            break
                    break
        finally:
            env.close()
        return

    # Batch sweep mode (no rendering)
    cluster_counts = [1, 2, 3]
    safe_margins = [35.0, 40.0, 45.0]
    goal_radii = [40.0, 50.0]
    seeds = [0, 1, 2, 3, 4]

    results = []
    for k in cluster_counts:
        for sm in safe_margins:
            for gr in goal_radii:
                successes = 0
                collisions = 0
                lengths = []
                for s in seeds:
                    env = ProactiveNavigationEnv()
                    env.dynamic_difficulty = False
                    env.allow_los_success = False  # require real goal entry
                    env.set_curriculum(curriculum_num_clusters=k, safe_margin=sm, goal_radius=gr, max_steps=1500)  # Daha fazla adım
                    try:
                        info, total_r = run_episode(env, model, max_steps=2000)  # Daha fazla adım
                        if info.get('success', False):
                            successes += 1
                        if info.get('collision', False):
                            collisions += 1
                        lengths.append(info.get('episode_length', 0))
                    finally:
                        env.close()
                mean_len = float(np.mean(lengths)) if lengths else 0.0
                results.append((k, sm, gr, successes/len(seeds), collisions/len(seeds), mean_len))
                print(f"k={k}, safe={sm:.0f}, goalR={gr:.0f} -> success={successes/len(seeds):.2f}, collision={collisions/len(seeds):.2f}, mean_len={mean_len:.1f}")

    best = sorted(results, key=lambda r: (-r[3], r[4], r[5]))[:5]
    print("\n🏁 Top configs (by success desc, collision asc, len asc):")
    for r in best:
        print(f"  clusters={r[0]}, safe={r[1]}, goalR={r[2]} -> success={r[3]:.2f}, collision={r[4]:.2f}, mean_len={r[5]:.1f}")


if __name__ == "__main__":
    main()


