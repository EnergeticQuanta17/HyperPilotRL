from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure, Logger

tmp_path = "tmp"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "log", "tensorboard", "json"])

model = A2C("MlpPolicy", "CartPole-v1", verbose=100, tensorboard_log=tmp_path)
# Set new logger
model.set_logger(new_logger)
model.learn(10000, tb_log_name="1")

for i in range(1, 11):
    model.learn(total_timesteps=1000, reset_num_timesteps=False, tb_log_name="1")