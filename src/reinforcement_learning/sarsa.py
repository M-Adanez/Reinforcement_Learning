from reinforcement_learning.env import GridworldEnv
from reinforcement_learning.tools import plot_value_function, plot_policy
import numpy as np


#CREO QUE TIENES MAL UNA COSA
def epsilon_greedy(q,state,epsilon,n_actions):
    """Select an action using an epsilon-greedy policy."""
    best=np.argmax(q[state])

    if np.random.random() < epsilon:
        choices=np.delete(np.arange(n_actions),best)
        return np.random.choice(choices)
    return best

def n_sarsa(env: GridworldEnv,
            e:float,
            lr:float,
            gamma:float,
            seed:int,
            n_episodes:int,
            n:int = 1 ):
    np.random.seed(seed)
    n_states=len(env.P)
    n_actions=len(env.P[0])
    policy=np.zeros(n_states)
    q=np.zeros((n_states,n_actions))
    rewards=[]

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        action=epsilon_greedy(q,state,e,n_actions)
        steps=0
        while not done or steps<=n:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action=epsilon_greedy(q,next_state,e,n_actions)

            done=terminated or truncated
            rewards.append(reward)
            
            current_q=q[state][action]
            q[state][action]= current_q - lr * (current_q - (reward + (gamma * q[next_state][next_action])))

            state=next_state
            action=next_action
            steps+=1
            

        g=0
        for i in range(len(rewards)):
            g+=(gamma**i)*rewards[i]
        g+=(gamma**(len(rewards)))*q[state][action]

        


    policy = np.array([
        np.argmax(q[state]) for state in range(n_states)
    ])

    v = np.array([
        np.max(q[state]) for state in range(n_states)
    ])
    return v, q, policy 
    


for is_slippery in [True,False]:
    config: dict = {
                "algorithm": "SARSA",
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
    env=GridworldEnv(is_slippery=is_slippery)
    v,q,policy=n_sarsa(env=env,
                       e=config["hyperparams"]["epsilon"],
                       lr=config["hyperparams"]["learning_rate"],
                       gamma=config["hyperparams"]["gamma"],
                       seed=config["seed"],
                       n_episodes=config["hyperparams"]["n_episodes"])

    