
from .q_learning import q_learning
from .value_iteration import value_iteration
from .policy_iteration import policy_iteration
from .sarsa import n_sarsa
from .montecarlo import montecarlo
from .optim import Algorithm

import os
import time
import pickle
import optuna as opt
import numpy as np
import gymnasium as gym
from PIL import Image

SHOULD_TRAIN = False
SHOULD_SAVE_SIMULATION = False

study_name = "taxi_rl_optimization"
env_name = "Taxi-v4"
env_params = {"is_rainy": True}

study = opt.load_study(
    study_name=study_name,
    storage="sqlite:///optuna_results_taxi_v1.db",
)

print("# BEST CONFIGURATION")
print("\tBest value:", study.best_value)
print("\tBest parameters:", study.best_params)
print("\tBest trial:", study.best_trial.number)


# Training
policy_path = "./results/taxi_policy.pkl"

if SHOULD_TRAIN:
    train_env = gym.make(env_name, **env_params)

    params = study.best_params.copy()
    algorithm_name = params.pop("algorithm")

    agent = Algorithm(algorithm_name)
    V, Q, policy = agent.train(train_env, **params)

    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    with open(policy_path, "wb") as f:
        pickle.dump(policy, f)
    
    train_env.close()
else:
    with open(policy_path, "rb") as f:
        policy = pickle.load(f)


# Verification
if SHOULD_SAVE_SIMULATION:
    gif_path = "./results/04_taxi.gif"
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)


    def get_frame(env: gym.Env) -> Image.Image:
        frame = env.render()

        if frame is None or isinstance(frame, list):
            raise RuntimeError("Expected a single RGB frame from env.render()")

        return Image.fromarray(np.asarray(frame))


    render_env = gym.make(
        env_name,
        render_mode="rgb_array",
        **env_params,
    )

    frames = []

    state, info = render_env.reset()
    frames.append(get_frame(render_env))

    done = False
    truncated = False

    while not (done or truncated):
        action = policy[state]
        state, reward, done, truncated, info = render_env.step(action)
        frames.append(get_frame(render_env))

    render_env.close()

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=False,
    )

    print(f"GIF saved to {gif_path}")
else:
    render_env = gym.make(env_name, render_mode="human", **env_params)
    state, info = render_env.reset()

    done = False
    truncated = False
    while not (done or truncated):
        action = policy[state]
        state, reward, done, truncated, info = render_env.step(action)
        time.sleep(0.5)

    render_env.close()
