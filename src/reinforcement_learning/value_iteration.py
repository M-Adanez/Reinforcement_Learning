import numpy as np

def value_iteration(env, gamma = 0.999, epsilon = 1e-8):
    '''Value iteration algo using q-function''' 

    P = env.P
    n_states = len(P)
    n_actions = len(P[0])

    delta = 0
    V = np.zeros(n_states)
    V_new = np.zeros(n_states)
    Q_new = np.zeros((n_states, n_actions))

    while True:
        for s in P:
            for a in P[s]:
                for prob, s_prime, reward, done in P[s][a]: # for all posible future state if we take acion a in s
                    future_value = 0.0 if done else V[s_prime]  
                    Q_new[s][a] += prob * (reward + gamma * future_value)

            V_new[s] = max(Q_new[s])
        
        delta = np.max(np.abs(V_new - V))

        if delta <= epsilon:  
            break
        else: 
            V = V_new.copy()  
            V_new = np.zeros(n_states)
            Q_new = np.zeros((n_states, n_actions))

    policy = np.argmax(Q_new, axis = 1)
    
    return V_new, Q_new, policy

