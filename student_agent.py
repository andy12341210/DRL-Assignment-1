# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
with open("taxi_dqn_model.pkl", "rb") as f:
 Q_table = pickle.load(f)



def get_action(obs):
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item()

    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

