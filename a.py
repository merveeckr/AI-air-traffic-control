from stable_baselines3 import PPO
from proactive_navigation_env import ProactiveNavigationEnv

env = ProactiveNavigationEnv()
env.dynamic_difficulty = False
env.set_curriculum(curriculum_num_clusters=2, goal_radius=40, max_steps=600, safe_margin=45)

model = PPO.load("C:\\Users\\merve\\Desktop\\bites-staj\\Yeni klasör (2) - Kopya\\proactive_navigation_training_20250819_194637\\best_model")
model.set_env(env)
model.learn(total_timesteps=100_000, reset_num_timesteps=False)
model.save("proactive_navigation_finetuned")