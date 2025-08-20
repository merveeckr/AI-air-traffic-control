"""
Advanced PPO Training for Proactive Navigation
- Broader curriculum with randomized parameters to improve generalization
- Compatible with existing simulation and environment (observation v1)
Usage:
  python advanced_training.py
"""

import os
from datetime import datetime
import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
import torch

from proactive_navigation_env import ProactiveNavigationEnv


class RandomizeEnvCallback(BaseCallback):
	"""Periodically randomize curriculum knobs across all vector envs.
	Keeps observation space consistent; only changes goal_radius, max_steps,
	safe_margin and curriculum_num_clusters to add variety.
	"""
	def __init__(self, env, freq_steps: int = 50_000, verbose: int = 0):
		super().__init__(verbose)
		self.vec_env = env
		self.freq_steps = freq_steps

	def _on_step(self) -> bool:
		if self.num_timesteps % self.freq_steps == 0:
			# Sample a new curriculum window
			goal_radius = float(np.random.uniform(35, 65))
			max_steps = int(np.random.randint(320, 620))
			safe_margin = float(np.random.uniform(35, 50))
			# 0 => let env pick random in reset (dynamic difficulty). Sometimes force 1..3.
			if np.random.rand() < 0.5:
				curr_clusters = 0
			else:
				curr_clusters = int(np.random.randint(1, 4))
			self.vec_env.env_method(
				'set_curriculum',
				goal_radius=goal_radius,
				max_steps=max_steps,
				curriculum_num_clusters=curr_clusters,
				safe_margin=safe_margin,
			)
			if self.verbose:
				print(f"\n🔀 Randomized curriculum: goal_radius={goal_radius:.1f}, max_steps={max_steps}, "
				      f"safe_margin={safe_margin:.1f}, clusters={curr_clusters}")
		return True


class OODTestCallback(BaseCallback):
	"""Out-of-distribution evaluation on harder/unseen layouts during training.
	Runs a brief evaluation on a fresh single env every `freq_steps`.
	"""
	def __init__(self, freq_steps: int = 100_000, n_episodes: int = 10, verbose: int = 1):
		super().__init__(verbose)
		self.freq_steps = freq_steps
		self.n_episodes = n_episodes

	def _on_step(self) -> bool:
		if self.num_timesteps % self.freq_steps == 0:
			# Fresh single env for OOD test
			env = ProactiveNavigationEnv()
			# Harder configuration: force 3 clusters and tighter safe margin
			env.dynamic_difficulty = False
			env.set_curriculum(curriculum_num_clusters=3, safe_margin=35, goal_radius=45, max_steps=500)
			# Disable LOS auto-success to measure true goal reaching
			env.allow_los_success = False
			successes, collisions, lengths = [], [], []
			for _ in range(self.n_episodes):
				obs, _ = env.reset()
				ep_success = False
				while True:
					action, _ = self.model.predict(obs, deterministic=True)
					obs, reward, terminated, truncated, info = env.step(action)
					if terminated or truncated:
						ep_success = bool(info.get('success', False))
						coll = bool(info.get('collision', False))
						successes.append(ep_success)
						collisions.append(coll)
						lengths.append(info.get('episode_length', 0))
						break
			# Print summary
			ms = float(np.mean(successes)) if successes else 0.0
			mc = float(np.mean(collisions)) if collisions else 0.0
			ml = float(np.mean(lengths)) if lengths else 0.0
			if self.verbose:
				print(f"\n📊 [OOD Test] success_rate={ms:.2f}, collision_rate={mc:.2f}, mean_len={ml:.1f} over {len(successes)} eps")
			env.close()
		return True


class EntropyScheduleCallback(BaseCallback):
	"""Linearly decays PPO entropy coefficient from start to end over schedule_steps.
	This emulates adaptive exploration for algorithms like PPO that expect a float ent_coef.
	"""
	def __init__(self, start: float = 0.02, end: float = 0.005, schedule_steps: int = 400_000, verbose: int = 0):
		super().__init__(verbose)
		self.start = float(start)
		self.end = float(end)
		self.schedule_steps = int(max(1, schedule_steps))

	def _on_step(self) -> bool:
		progress = min(1.0, self.num_timesteps / float(self.schedule_steps))
		new_ent = self.start + (self.end - self.start) * progress
		# PPO expects a float ent_coef
		self.model.ent_coef = float(new_ent)
		return True


class AdvancedProactiveTrainer:
	def __init__(self):
		self.config = {
			'total_timesteps': 750_000,
			'learning_rate': 1e-4,
			'n_steps': 2048,
			'batch_size': 256,
			'n_epochs': 10,
			'gamma': 0.995,
			'gae_lambda': 0.98,
			'clip_range': 0.2,
			'ent_coef': 0.02,
			'vf_coef': 0.5,
			'max_grad_norm': 0.3,
			'n_envs': 16,
			'eval_freq': 10_000,
			'save_freq': 100_000,
			'eval_episodes': 20,
			'seed': 123,
		}
		self.training_dir = f"proactive_navigation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		os.makedirs(self.training_dir, exist_ok=True)
		set_random_seed(self.config['seed'])

	def _make_env(self):
		def make_single():
			# dynamic_difficulty True by default -> inherent variety
			env = ProactiveNavigationEnv()
			# Keep observation in v1 to stay compatible with simulation
			env.set_curriculum(goal_radius=50, max_steps=400, curriculum_num_clusters=0, safe_margin=40)
			return Monitor(env)
		return make_single

	def setup(self):
		self.env = make_vec_env(self._make_env(), n_envs=self.config['n_envs'], seed=self.config['seed'])
		self.eval_env = make_vec_env(self._make_env(), n_envs=1, seed=self.config['seed'] + 999)
		# Enable training helpers on all envs
		self.env.env_method('set_training_mode', True, initial_random_steps=5)
		self.env.env_method('set_observation_noise', 0.4)

		policy_kwargs = {
			'net_arch': [dict(pi=[1024, 1024, 512, 256, 128], vf=[1024, 1024, 512, 256, 128])],
			'activation_fn': torch.nn.ReLU,
			'ortho_init': True,
		}

		self.model = PPO(
			"MlpPolicy",
			self.env,
			learning_rate=self.config['learning_rate'],
			n_steps=self.config['n_steps'],
			batch_size=self.config['batch_size'],
			n_epochs=self.config['n_epochs'],
			gamma=self.config['gamma'],
			gae_lambda=self.config['gae_lambda'],
			clip_range=self.config['clip_range'],
			ent_coef=self.config['ent_coef'],
			vf_coef=self.config['vf_coef'],
			max_grad_norm=self.config['max_grad_norm'],
			policy_kwargs=policy_kwargs,
			verbose=1,
			tensorboard_log=f"{self.training_dir}/tensorboard_logs",
		)

		self.callbacks = [
			CheckpointCallback(save_freq=self.config['save_freq'], save_path=f"{self.training_dir}/checkpoints", name_prefix="proactive_navigation_ppo"),
			EvalCallback(self.eval_env, best_model_save_path=f"{self.training_dir}/best_model", log_path=f"{self.training_dir}/eval_logs", eval_freq=self.config['eval_freq'], n_eval_episodes=self.config['eval_episodes'], deterministic=True, render=False),
			RandomizeEnvCallback(self.env, freq_steps=50_000, verbose=1),
			OODTestCallback(freq_steps=100_000, n_episodes=8, verbose=1),
			EntropyScheduleCallback(start=self.config['ent_coef'], end=0.005, schedule_steps=400_000, verbose=0),
		]

	def train(self):
		print("🚀 Advanced PPO training started...")
		# Phase A: Easier randomized (warm-up)
		self.env.env_method('set_curriculum', goal_radius=60, max_steps=350, curriculum_num_clusters=1, safe_margin=45)
		self.model.learn(total_timesteps=150_000, callback=self.callbacks, progress_bar=True)
		# Phase B: Mixed difficulty with randomization
		self.env.env_method('set_curriculum', goal_radius=50, max_steps=450, curriculum_num_clusters=0, safe_margin=40)
		self.model.learn(total_timesteps=300_000, callback=self.callbacks, progress_bar=True)
		# Phase C: Harder settings
		self.env.env_method('set_curriculum', goal_radius=40, max_steps=550, curriculum_num_clusters=0, safe_margin=38)
		remaining = max(0, self.config['total_timesteps'] - 450_000)
		if remaining:
			self.model.learn(total_timesteps=remaining, callback=self.callbacks, progress_bar=True)

		final_path = os.path.join(self.training_dir, 'final_model')
		self.model.save(final_path)
		print(f"💾 Saved final model to: {final_path}.zip")


def main():
	trainer = AdvancedProactiveTrainer()
	trainer.setup()
	trainer.train()


if __name__ == "__main__":
	main()
