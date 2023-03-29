import gymnasium as gym
import os
from stable_baselines3 import PPO

from PPO_HypConfig import *

env_name = "BipedalWalker-v3"
algorithm = "PPO"

env = gym.make(env_name, render_mode='human')

print("Choose the model to show output on, among the following: ")
model_dir = f"model/optuna/10/{env_name}/{algorithm}"
print(os.listdir(model_dir))
print(model_dir)

model_no = input("Enter execution number: ")
all_files = os.listdir(f"{model_dir}/{model_no}")
print(all_files)
training_till = "_" + input("Select the model: ")

try:
    index = all_files.index(next(s for s in all_files if training_till in s))
    print(index)
except StopIteration:
    print(f"{training_till} is not a substring of any string in the list")
    exit()

model_dir = f"{model_dir}/{model_no}/{all_files[index]}"
model = PPO.load(model_dir, env=env)    

for ep in range(1):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()