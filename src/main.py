from .q_learning import q_learning
from .value_iteration import value_iteration
from .policy_iteration import policy_iteration
from .sarsa import n_sarsa
from .montecarlo import montecarlo
from .tools import plot_value_function, plot_policy

from .env import GridworldEnv

from pathlib import Path
import json
import time

import gymnasium
import numpy as np
import icecream as ic


ALGORITHMS = [q_learning, value_iteration, policy_iteration, n_sarsa, montecarlo]
IS_SLIPPERY = False
SEED = 42

PLOT_RESULTS = True

def create_config(SEED: int, IS_SLIPPERY: bool, algorithm_name: str): 
    config: dict = {
        "algorithm": algorithm_name,
        "seed": SEED,
        "env": "GridworldEnv", 
        "env_params": {
            "is_slippery": IS_SLIPPERY
            },
        "optuna": False,
        "hyperparams": {
            "epsilon": 1e-1,
            "gamma": 0.9,
        }      
    }

    if algorithm_name not in ['policy_iteration', 'value_iteration']:
        config['hyperparams']['n_episodes'] = 1000 

    if algorithm_name in ['q_learning', 'n_sarsa']:
        config['hyperparams']['learning_rate'] = 0.1

    return config



if __name__ == "__main__":
    results = []

    results_dir = Path("results/")
    output_file = Path("grid-world-results.json")
    output_path = results_dir / output_file

    for algorithm in ALGORITHMS:
        print(f"Training algorithm {algorithm.__name__}")
        config = create_config(SEED, IS_SLIPPERY, algorithm.__name__)
        np.random.seed(config.get('seed')) 
        env = GridworldEnv(**config.get('env_params'))

        start_time = time.time()
        V, Q, policy = algorithm(env, **config.get('hyperparams'))
        total_time = time.time() - start_time

        if PLOT_RESULTS:
            plot_value_function(V, env, algorithm_name = algorithm.__name__)
            plot_policy(policy, env, algorithm_name = algorithm.__name__)

        config['Time'] = total_time
        config['V'] = V.tolist()
        config['Q'] = Q.tolist()
        config['policy'] = policy.tolist()

        results.append(config)
        
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


