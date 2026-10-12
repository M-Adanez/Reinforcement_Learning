
from .q_learning import q_learning
from .value_iteration import value_iteration
from .policy_iteration import policy_iteration
from .sarsa import n_sarsa
from .montecarlo import montecarlo

import json
import optuna
import gymnasium as gym
from enum import Enum
from torch.utils.tensorboard import SummaryWriter

class Algorithm(Enum):
    VALUE_ITERATION = "value_iteration"
    POLICY_ITERATION = "policy_iteration"
    MONTE_CARLO = "monte_carlo"
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"

    def train(self, env, **hyperparams):
        if self == Algorithm.VALUE_ITERATION:
            return value_iteration(env, **hyperparams)
        elif self == Algorithm.POLICY_ITERATION:
            return policy_iteration(env, **hyperparams)
        elif self == Algorithm.MONTE_CARLO:
            return montecarlo(env, **hyperparams)
        elif self == Algorithm.Q_LEARNING:
            return q_learning(env, **hyperparams)
        else: # SARSA, no N-SARSA
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
    success = False

    while not done:
        action = policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        g += (gamma ** i) * reward
        success = success or (reward == 20) # taxi-v4 
        done = terminated or truncated
        i += 1
        state = next_state
        
    return g, success

def evaluate_policy(
        env,
        policy,
        gamma,
        n_runs
    ):
    results = [
        run_simulation(env, policy, gamma)
        for _ in range(n_runs)
    ]

    returns = [result[0] for result in results]
    successes = [result[1] for result in results]

    return {
        "avg_return": sum(returns) / n_runs,
        "success_rate": sum(successes) / n_runs,
    }

def objective(trial, writer):
    seed = 123
    algorithm = Algorithm(
        trial.suggest_categorical(
            "algorithm",
            [
                a.value 
                for a in Algorithm
            ]
        )
    )

    env_name = "Taxi-v4"
    env_params = {"is_rainy": True}
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
                "epsilon-greedy", 0.01, 1.0 # e-greedy
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
                "epsilon-greedy", 0.01, 1.0
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
    elif algorithm == Algorithm.Q_LEARNING:
        hyperparams = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True
            ),
            "epsilon": trial.suggest_float(
                "epsilon-greedy", 0.01, 1.0
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

    metrics = evaluate_policy(
        env=env,
        policy=config["policy"],
        gamma=config["hyperparams"]["gamma"],
        n_runs=100,
    )

    config["avg_return"] = metrics["avg_return"]
    config["success_rate"] = metrics["success_rate"]

    writer.add_scalar(
        f"{algorithm.value}/avg_return",
        config["avg_return"],
        trial.number,
    )
    writer.add_scalar(
        f"{algorithm.value}/success_rate",
        config["success_rate"],
        trial.number,
    )

    return float(config["avg_return"])
    


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/taxi_rl")
    try:
        study = optuna.create_study(
            study_name="taxi_rl_optimization",
            storage="sqlite:///optuna_results.db",
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(
            lambda trial: objective(trial, writer),
            n_trials=1000,
            n_jobs=-1,
            show_progress_bar=True
        )

        results = {
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
                for trial in study.trials
            ],
        }

        with open("./results/taxi_rl_results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Best value:", study.best_value)
        print("Best parameters:", study.best_params)
        print("Results saved to taxi_rl_results.json")



    finally:
        writer.close()
