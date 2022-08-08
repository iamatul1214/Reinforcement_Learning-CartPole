import gym
import numpy
import time
import numpy as np
from tqdm import tqdm

env = gym.make("CartPole-v1")

def basic_policy(PoleAngle):
    if PoleAngle < 0:  ## falling left
        return 0       ## moving left
    return 1

total_rewards = list()

N_episodes = 200
N_steps = 200

for episode in range(N_episodes):
    rewards = 0
    # CartPoistion, CartVelocity, PoleAngle, PoleAngularVelocity
    observations = env.reset()
    PoleAngle = observations[2]
    for step in tqdm(range(N_steps)):
        env.render()
        action = basic_policy(PoleAngle)
        observations, reward,done,info = env.step(action)
        time.sleep(0.0001)
        rewards+=reward
        if done: ## fallen
            break
    total_rewards.append(rewards)


stats = {
    "mean": np.mean(total_rewards),
    "std": np.std(total_rewards),
    "min": np.min(total_rewards),
    "max": np.max(total_rewards)
}

print(f"Some stats = {stats}")