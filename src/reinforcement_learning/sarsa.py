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
            n:int = 1 ):
    n_states=len(env.P)
    n_actions=len(env.P[0])
    policy=np.zeros(n_states)
    q=np.zeros((n_states,n_actions))

    episodes=range(1_000)#????
    for episode in episodes:
        state, _ = env.reset()
        done = False
        action=epsilon_greedy(q,state,e,n_actions)

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done=terminated or truncated

            next_action=epsilon_greedy(q,next_state,e,n_actions)
            current_q=q[state][action]
            q[state][action]= current_q - lr * (current_q - (reward + (gamma * q[next_state][next_action])))

            state=next_state
            action=next_action

    policy = np.array([
        np.argmax(q[state]) for state in range(n_states)
    ])

    v = np.array([
        np.max(q[state]) for state in range(n_states)
    ])
    return v, q, policy 
    

env=GridworldEnv(is_slippery=False)
v,q,policy=n_sarsa(env=env,
        e=0.3,
        lr=0.5,
        gamma=0.9)

plot_value_function(v, env)
plot_policy(policy, env)