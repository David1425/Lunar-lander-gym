import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

load_file = "updatedModel1.keras"
best_save = "test_model1.keras"
save_file = ""

env = gym.make("LunarLander-v2", render_mode="ansi")
observation, info = env.reset()

model = keras.models.load_model(load_file)

print(observation)
print(env.observation_space.shape)

gamma = 0.99
eps = np.finfo(np.float32).eps.item()

num_inputs = 8
num_actions = 4
num_hidden = 128
best = -500

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

inputs = layers.Input(shape=(num_inputs))
common = layers.Dense(num_hidden, activation='relu')(inputs)
action = layers.Dense(num_actions, activation='softmax')(common)
critic = layers.Dense(1)(common)

# model = keras.Model(inputs=inputs, outputs=[action, critic])

print(model.summary())

for t in range(1000000):
    observation, info = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for _ in range(500):
            state = tf.convert_to_tensor(observation)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0,0])

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0,action]))

            observation, reward, terminated, truncated, info = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if terminated or truncated:
                break
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []

        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))

        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
    
    print("episode %d, reward: %.3f" % (t, running_reward))
    if save_file != "":
        if t > 50 and running_reward > best:
            model.compile()
            model.save(best_save)
            best = running_reward
            print("saved model")
        if t > 50 and running_reward > 200:
            model.compile()
            model.save(save_file)
            print("saved model")
            break
        if t % 50 == 0:
            model.compile()
            model.save(save_file)
            print("saved model")

env.close()
state = tf.convert_to_tensor(observation)
state = tf.expand_dims(state, 0)
if save_file != "":
    model.compile()
    model.save(save_file)
    print("saved model")
