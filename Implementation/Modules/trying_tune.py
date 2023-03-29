import random
import ray
from ray import tune

def trainable(config):
    # Generate two random numbers from the search space
    x = config["x"]
    y = config["y"]
    
    # Compute the sum of the numbers
    result = x + y
    
    # Return the result
    return result

# Define the search space for the hyperparameters
search_space = {
    "x": tune.uniform(0, 1),
    "y": tune.uniform(0, 1)
}

# Create a configuration for `tune.run`
config = {
    "num_samples": 10,
    "config": search_space,
    "resources_per_trial": {"cpu": 1, "gpu": 0},
    "local_dir": "./results"
}

# Start the hyperparameter search using `tune.run`
ray.init(local_mode=True)
analysis = tune.run(trainable, **config)
