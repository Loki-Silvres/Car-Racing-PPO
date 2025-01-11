import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

# Create the CarRacing environment
env_id = "CarRacing-v3"
env = gym.make(env_id, continuous=True, domain_randomize=False)  # Continuous action space

# Wrap the environment for vectorized training
env = DummyVecEnv([lambda: env])

# Define the PPO model
model = PPO(
    "CnnPolicy",  # Use CNN-based policy suitable for image observations
    env,
    verbose=1,
    tensorboard_log="./ppo_carracing_tensorboard/",
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
)

# Define evaluation and checkpoint callbacks
eval_env = gym.make(env_id, continuous=True)
eval_env = DummyVecEnv([lambda: eval_env])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_carracing",
)

# Train the model
timesteps = 1000000  # 1 million timesteps
model.learn(total_timesteps=timesteps, callback=[eval_callback, checkpoint_callback])

# Save the trained model
model.save("ppo_carracing")
print("Training completed and model saved!")

# Close the environments
env.close()
eval_env.close()
