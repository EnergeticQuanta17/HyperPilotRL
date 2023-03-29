import os

import gym
from stable_baselines3 import PPO

env_name = "LunarLander-v2"


models_dir = f"models/{env_name}/PPO"
logdir = f"logs/{env_name}"

if(not os.path.exists(models_dir)):
    os.makedirs(models_dir)

if(not os.path.exists(logdir)):
    os.makedirs(logdir)

env = gym.make(env_name)
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100_000

for i in range(1, 1_000_000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='logs')
    model.save(f"{models_dir}/{TIMESTEPS*i}")