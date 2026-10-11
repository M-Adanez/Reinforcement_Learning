
from .env import GridworldEnv
from .tools import plot_policy, plot_value_function

import json
import numpy as np
from icecream import ic
from pathlib import Path


def epsilon_greedy(Q, state, epsilon):
    """Select an action using an epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])

def montecarlo(
        env,
        epsilon: float = 0.3,
        gamma: float = 0.99,
        n_episodes: int = 1000
    ) -> tuple:

    n_states = len(env.P)
    n_actions = len(env.P[0])
    Q = np.zeros(shape=(n_states, n_actions))
    returns = np.zeros(shape=(n_states, n_actions))
    num = np.zeros(shape=(n_states, n_actions))

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_data = []
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_data.append( (state, action, reward) )
            state = next_state
        g = 0
        for t in range(len(ep_data)-1, -1, -1):
            state, action, reward = ep_data[t]
            g = gamma * g + reward
            returns[state, action] += g
            num[state, action] += 1
            Q[state, action] = returns[state, action] / num[state, action]

    policy = np.array([
        np.argmax(Q[state])
        for state in range(n_states)
    ])
    V = np.array([
            np.max(Q[state])
            for state in range(n_states)
        ])
    return V, Q, policy

def run_montecarlo(
        config: dict,
        plot_results: bool    
    ) -> dict:

    np.random.seed(config["seed"])
    
    env = GridworldEnv(**config["env_params"])
    V, Q, policy = montecarlo(env, **config["hyperparams"])
    config["V"] = V.tolist()
    config["Q"] = Q.tolist()
    config["policy"] = policy.tolist()

    if plot_results:
        plot_value_function(V, env)
        plot_policy(policy, env)

    return config

if __name__ == "__main__":
    results_dir = Path("results/")
    output_file = Path("montecarlo.json")

    output_path = results_dir / output_file

    results = []
    for is_slippery in [False, True]:
        ic(is_slippery)
        config: dict = {
            "algorithm": "Montecarlo",
            "seed": 123,
            "env": "GridworldEnv",
            "env_params": {
                "is_slippery": is_slippery
            },
            "optuna": False,
            "hyperparams": {
                "epsilon": 0.1,
                "gamma": 0.99,
                "n_episodes": 50_000
            }
        }
        data = run_montecarlo(config, plot_results=True)
        results.append(data)
        
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
