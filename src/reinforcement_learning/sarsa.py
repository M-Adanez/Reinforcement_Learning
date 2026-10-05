from reinforcement_learning.env import GridworldEnv
import numpy as np
def n_sarsa(P,n:int=1):
    n_states=len(P)
    n_actions=len(P[0])
    reward=0
    print(P)
    q=np.zeros((n_states,n_actions))
    for key,val in P.items():
        print(key,":",val)
        # Prob,next_state,reward,terminal
    

env=GridworldEnv(is_slippery=False)
n_sarsa(env.P)
