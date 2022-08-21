import tensorflow as tf
import gym
import time
import numpy as np
from tqdm import tqdm
import os

SEED = 43
tf.random.set_seed(SEED)

## creating a policy gradient
def pg_policy(observation, model): ## observation from the game, model which we defined
    left_probability = model.predict(observation[np.newaxis])  #probability value between 0 and 1
    action = int(np.random.rand() > left_probability) 
    return action

def show_one_episode(policy, model, n_max_steps=500, seed=43):
    env = gym.make("CartPole-v1")
    obs = env.reset()
    for step in range(n_max_steps):
        env.render()
        action = policy(obs, model)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    print(f"steps  = {step}")
    return step, obs

trained_model = tf.keras.models.load_model(r'C:\Users\Anagha M\Python Projects\Reinforcement_Learning-CartPole\src\model_at_Mon_Aug_15_00_02_22_2022_.h5')
show_one_episode(pg_policy, trained_model)


