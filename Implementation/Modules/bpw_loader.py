import gym
from stable_baselines3 import PPO

# create BipedalWalker environment
env = gym.make('BipedalWalker-v3')

# load pre-trained PPO model
model = PPO.load("bipedalwalker_ppo")

# run two episodes of environment using loaded model
for i in range(2):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total reward for episode {i+1}: {total_reward:.2f}")

# close environment
env.close()
