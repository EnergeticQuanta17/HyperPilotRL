import gym
from stable_baselines3 import PPO
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

# Define the function to train the PPO algorithm
def train_PPO(config):
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0, **config)
    model.learn(total_timesteps=10000)
    mean_reward = model.evaluate(env, n_eval_episodes=5, return_episode_rewards=True)[0].mean()
    tune.report(mean_reward=mean_reward)

# Define the hyperparameters search space
config_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "n_steps": tune.choice([16, 32, 64, 128]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "ent_coef": tune.uniform(0, 0.1),
}

# Define the HyperBandScheduler
scheduler = HyperBandScheduler(
    time_attr="training_iteration",
    metric="mean_reward",
    mode="max",
    max_t=100,
    grace_period=None,
    reduction_factor=3,
    min_t=1  # Specify the minimum time a trial should run for before it can be terminated.
)

# Define the configuration for the hyperparameter tuning experiment
config = {
    "num_samples": 10,
    "config": config_space,
    "scheduler": scheduler,
    "resources_per_trial": {"cpu": 1},
    "checkpoint_at_end": True,
    "verbose": 1,
}

# Start the hyperparameter tuning experiment
analysis = tune.run(train_PPO, **config)

# Print the best hyperparameters and the corresponding mean reward
print("Best hyperparameters: ", analysis.best_config)
print("Best mean reward: ", analysis.best_result["mean_reward"])
