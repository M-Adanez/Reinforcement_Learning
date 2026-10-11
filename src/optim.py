
from .q_learning import q_learning
from .value_iteration import value_iteration
from .policy_iteration import policy_iteration
from .sarsa import n_sarsa
from .montecarlo import montecarlo

import optuna
import gymnasium as gym
from enum import Enum

class Algorithm(Enum):
    VALUE_ITERATION = "value_iteration"
    POLICY_ITERATION = "policy_iteration"
    MONTE_CARLO = "monte_carlo"
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    N_SARSA = "n_sarsa"

    def train(self, env, **hyperparams):
        if self == Algorithm.VALUE_ITERATION:
            return value_iteration(env, **hyperparams)
        elif self == Algorithm.POLICY_ITERATION:
            return policy_iteration(env, **hyperparams)
        elif self == Algorithm.MONTE_CARLO:
            return montecarlo(env, **hyperparams)
        elif self == Algorithm.Q_LEARNING:
            return q_learning(env, **hyperparams)
        elif self == Algorithm.SARSA: # 1-SARSA
            return n_sarsa(env, **hyperparams)
        else: # N-SARSA
            return n_sarsa(env, **hyperparams)

def create_config(
        algorithm_name: str,
        env_name: str,
        seed: int,
        env_params: dict,
        hyperparams: dict
    ) -> dict: 

    config: dict = {
        "algorithm": algorithm_name,
        "seed": seed,
        "env": env_name, 
        "env_params": env_params,
        "optuna": True,
        "hyperparams": hyperparams  
    }
    return config

def run_simulation(
        env,
        policy,
        gamma
    ):
    state, info = env.reset()
    done = False
    i = 0
    g = 0

    while not done:
        action = policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        g += (gamma ** i) * reward
        done = terminated or truncated
        i += 1
        state = next_state
    return g

def evaluate_policy(
        env,
        policy,
        gamma,
        n_runs
    ):
    returns = [
        run_simulation(env, policy, gamma)
        for _ in range(n_runs)
    ]

    return sum(returns) / len(returns)

def objective(trial):
    seed = 123
    algorithm = Algorithm(
        trial.suggest_categorical(
            "algorithm",
            [
                a.value 
                for a in Algorithm
                if a != Algorithm.N_SARSA # TODO: incluir n-sarsa cuando funcione
            ]
        )
    )

    # TODO: fix env
    env_name = ""
    env_params = {}
    env = gym.make(env_name, **env_params)

    if algorithm in (Algorithm.VALUE_ITERATION, Algorithm.POLICY_ITERATION):
        hyperparams = {
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "epsilon": trial.suggest_float(
                "epsilon", 1e-6, 1e-2, log=True) # stop condition delta, no e-greedy
        }
    elif algorithm == Algorithm.MONTE_CARLO:
        hyperparams = {
            "epsilon": trial.suggest_float(
                "epsilon", 0.01, 1.0 # e-greedy
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.90, 0.999
            ),
            "n_episodes": trial.suggest_int(
                "n_episodes", 1_000, 50_000, log=True
            )
        }
    elif algorithm == Algorithm.SARSA:
        hyperparams = {
            "epsilon": trial.suggest_float(
                "epsilon", 0.01, 1.0
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.90, 0.999
            ),
            "n_episodes": trial.suggest_int(
                "n_episodes", 1_000, 50_000, log=True
            ),
            "n": 1
        }
    elif algorithm == Algorithm.N_SARSA:
        hyperparams = {
            "epsilon": trial.suggest_float(
                "epsilon", 0.01, 1.0
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.90, 0.999
            ),
            "n_episodes": trial.suggest_int(
                "n_episodes", 1_000, 50_000, log=True
            ),
            "n": trial.suggest_int(
                "n", 1, 10
            )
        }
    elif algorithm == Algorithm.Q_LEARNING:
        hyperparams = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True
            ),
            "epsilon": trial.suggest_float(
                "epsilon", 0.01, 1.0
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.90, 0.999
            ),
            "n_episodes": trial.suggest_int(
                "n_episodes", 1_000, 50_000, log=True
            )
        }
    else:
        raise ValueError("Unknown algorithm suggested by optuna")

    config = create_config(
        algorithm_name=algorithm.value,
        env_name=env_name,
        seed=seed,
        env_params=env_params,
        hyperparams=hyperparams
    )

    config["V"], config["Q"], config["policy"] = algorithm.train(env, **hyperparams)

    config["avg_return"] = evaluate_policy(
        env=env,
        policy=config["policy"],
        gamma=config["hyperparams"]["gamma"],
        n_runs=100) 

    return config["avg_return"]
    


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="rl_hyperparameter_optimization",
        direction="maximize",
    )

    study.optimize(
        objective,
        n_trials=100,
    )

    # TODO: analizar mejores hyperparams