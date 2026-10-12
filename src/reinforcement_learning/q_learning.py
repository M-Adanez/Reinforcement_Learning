
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

def q_learning(
        env,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        n_episodes: int = 1_000
    ) -> tuple:
    """Train a Q-learning agent and return V, Q, and the greedy policy."""

    if isinstance(env, GridworldEnv):
        P = env.P
    else:
        P = env.unwrapped.P
    n_states = len(P)
    n_actions = len(P[0])
    Q = np.zeros(shape=(n_states, n_actions), dtype=np.float64)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # experience
            action = epsilon_greedy(Q, state, epsilon) # off-policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            # ic(state, action, next_state, reward, terminated, truncated)
            done = terminated or truncated

            # q update
            target = reward if done else reward + gamma * np.max(Q[next_state])
            # ic(Q[state, action], target)
            Q[state, action] -= learning_rate * (Q[state, action] - target)
            state = next_state

    policy = np.array([
        np.argmax(Q[state])
        for state in range(n_states)
    ])
    V = np.array([
        np.max(Q[state])
        for state in range(n_states)
    ])
    return V, Q, policy

def run_q_learning(
        config: dict,
        plot_results: bool
    ) -> dict:
    """Run Q-learning with the given configuration and plot the results."""

    np.random.seed(config["seed"])

    env = GridworldEnv(**config["env_params"])
    V, Q, policy = q_learning(env, **config["hyperparams"])
    config["V"] = V.tolist()
    config["Q"] = Q.tolist()
    config["policy"] = policy.tolist()

    if plot_results:
        plot_value_function(V, env)
        plot_policy(policy, env)

    return config

if __name__ == "__main__":
    results_dir = Path("results/")
    output_file = Path("q_learning.json")

    output_path = results_dir / output_file

    results = []

    # env determinista + estocastico
    for is_slippery in [False, True]:
        ic(is_slippery)
        config: dict = {
            "algorithm": "Q-learning",
            "seed": 123,
            "env": "GridworldEnv",
            "env_params": {
                "is_slippery": is_slippery
            },
            "optuna": False,
            "hyperparams": {
                "learning_rate": 0.5,
                "epsilon": 0.3,
                "gamma": 0.9,
                "n_episodes": 10_000
            }
        }
        data = run_q_learning(config, plot_results=True)
        results.append(data)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
