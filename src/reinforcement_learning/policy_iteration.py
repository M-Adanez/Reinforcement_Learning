
from .env import GridworldEnv

import numpy as np

def policy_iteration(env, gamma = 0.999, epsilon = 1e-8):
    'Policy iteration algorithm'

    if isinstance(env, GridworldEnv):
        P = env.P
    else:
        P = env.unwrapped.P
    n_states = len(P)  
    n_actions = len(P[0])

    
    delta = 0.0  
    V_pi = np.zeros(n_states)  
    V_pi_new = np.zeros(n_states)
    Q_pi = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)  
    policy_new = np.zeros(n_states, dtype=int)  
  
    while True:  
        
        while True: 
            # Policy evaluation

            for s in P:
                a = policy[s]  

                for prob, s_prime, reward, done in P[s][a]:
                    future_value = 0.0 if done else V_pi[s_prime]
                    V_pi_new[s] += prob * (reward + gamma * future_value)

            delta = np.max(np.abs(V_pi_new - V_pi))
            if delta <= epsilon:  
                break
            else:
                V_pi = V_pi_new.copy()  
                V_pi_new = np.zeros(n_states)

        
        for s in P:
            # Policy improvement

            for a in P[s]:
                # Obtain Q_pi 

                for prob, s_prime, reward, done in P[s][a]:
                    future_value = 0.0 if done else V_pi_new[s_prime]
                    Q_pi[s][a] += prob * (reward + gamma * future_value)


            a = np.argmax(Q_pi[s])
            policy_new[s] = a

        if np.array_equal(policy_new, policy):
            break  

        else:
            policy = policy_new.copy()
            policy_new = np.zeros(n_states, dtype=int)
            Q_pi = np.zeros((n_states, n_actions))
            V_pi = V_pi_new.copy()
            V_pi_new = np.zeros(n_states)

    return V_pi_new, Q_pi, policy
