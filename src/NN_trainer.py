import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import gym
import re
import os

SEED = 43
tf.random.set_seed(SEED)
env = gym.make("CartPole-v1")
## parameters
n_iterations = 125
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95
learning_rate = 0.01

obs = env.reset(seed=SEED)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.binary_crossentropy

LAYERS = [
    tf.keras.layers.Dense(5, activation = 'relu'),
    # tf.keras.layers.Dense(2, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # here to get the probability of left, and right = 1 - left
]
model = tf.keras.Sequential(LAYERS)

def play_one_step(env,observation, model, loss_fn):
    with tf.GradientTape() as tape:
        left_probability = model(observation[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_probability) # gives true or false
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target,left_probability))
    grads = tape.gradient(loss,model.trainable_variables) # dc/dw
    new_observation, reward, done, info = env.step(int(action))
    return new_observation,reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = list()
    all_grads = list()
    for episode in range(n_episodes):
        current_rewards = list()
        current_grads = list()
        observation = env.reset()
        for step in range(n_max_steps):
            observation, rewards, done, grads = play_one_step(env,observation,model, loss_fn)
            current_rewards.append(rewards)
            current_grads.append(grads)
            if done:
                break
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    N = len(rewards)
    for step in range(N - 2, -1, -1):
        # a_n + a_n+1*gamma
        discounted[step] = discounted[step] + discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = list()
    for reward in all_rewards:
        # discounted rewards
        drs = discount_rewards(reward, discount_factor)
        all_discounted_rewards.append(drs)

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    normalize_rewards = list()
    for discounted_rewards in all_discounted_rewards:
        nrs = (discounted_rewards - reward_mean) / reward_std
        normalize_rewards.append(nrs)
    return normalize_rewards

def save_model():
    unique_name = re.sub(r"[\s+:]", "_", time.asctime())
    model_name = f"model_at_{unique_name}_.h5"
    model.save(model_name)
    print(f"model is saved as '{model_name}'")

for epoch in range(n_iterations):
    all_rewards , all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn
    )
    total_rewards = sum(map(sum, all_rewards))
    print(f"epoch:{epoch + 1}/{n_iterations}, mean rewards: {total_rewards/n_episodes_per_update}")
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
    all_mean_grads = list()
    N= len(model.trainable_variables)
    for var_index in range(N):
        temp_reduce_mean = list()
        for episode_index, final_rewards in enumerate(all_final_rewards): # rewards for every episode
            for step, final_reward in enumerate(final_rewards): # several steps
                result = final_reward * all_grads[episode_index][step][var_index]
                temp_reduce_mean.append(result)
        mean_grads = tf.reduce_mean(temp_reduce_mean, axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads,model.trainable_variables))

unique_name = re.sub(r"[\s+:]", "_", time.asctime())
model_name = f"model_at_{unique_name}_.h5"
model.save(model_name)
print(f"model is saved as '{model_name}'")