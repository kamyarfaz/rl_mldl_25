"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

def train_agent(env_id, algorithm='PPO', total_timesteps=100000):
    # Create environment
    env = gym.make(env_id)

    # Select algorithm
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1)
    else:
        raise ValueError("Unsupported algorithm. Choose 'PPO' or 'SAC'.")

    # Create evaluation callback
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)

    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the model
    model_path = f"{algorithm}_hopper_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model

def main():
    train_env_id = 'CustomHopper-source-v0'
    test_env_id = 'CustomHopper-target-v0'

    # Train PPO agent
    print("Training PPO agent...")
    ppo_model = train_agent(train_env_id, algorithm='PPO')

    # Train SAC agent
    print("Training SAC agent...")
    sac_model = train_agent(train_env_id, algorithm='SAC')

    # Evaluate models
    print("Evaluating PPO agent...")
    evaluate_agent(ppo_model, test_env_id)

    print("Evaluating SAC agent...")
    evaluate_agent(sac_model, test_env_id)

def evaluate_agent(model, env_id, n_episodes=10):
    env = gym.make(env_id)
    total_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
    avg_reward = sum(total_rewards) / n_episodes
    print(f"Average Reward over {n_episodes} Evaluation Episodes: {avg_reward}")

if __name__ == '__main__':
    main()