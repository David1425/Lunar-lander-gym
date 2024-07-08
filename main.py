import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

load_file = "updatedModel2.keras"

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

model = keras.models.load_model(load_file)

print(observation)
print(env.observation_space.shape)

gamma = 0.99
eps = np.finfo(np.float32).eps.item()

num_actions = 4

print(model.summary())

for t in range(10):
    observation, info = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for _ in range(500):
            state = tf.convert_to_tensor(observation)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break
        
        

env.close()
