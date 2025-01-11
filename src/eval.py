from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("logs/best_model.zip")
env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
