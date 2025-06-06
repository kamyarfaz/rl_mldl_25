"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from env.custom_hopper import *
from agent import Agent, Policy, ActorCriticAgent, REINFORCEAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def apply_domain_randomization(env):
    # Randomize the masses of the Hopper's links, except the torso
    body_names = env.sim.model.body_names
    for i, body_name in enumerate(body_names):
        if body_name != 'torso':  # Do not randomize the torso mass
            env.sim.model.body_mass[i] *= np.random.uniform(0.8, 1.2)


def train_with_domain_randomization(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    agent = REINFORCEAgent(env, learning_rate=learning_rate)
    
    # Domain randomization parameters
    mass_range = (0.8, 1.2)  # Randomize mass between 80% and 120% of original
    friction_range = (0.8, 1.2)  # Randomize friction between 80% and 120% of original
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        states = []  # Store states for policy update
        
        # Apply domain randomization at the start of each episode
        env.model.body_mass[1:] *= np.random.uniform(*mass_range)  # Randomize mass of all bodies except the base
        env.model.geom_friction[1:, 0] *= np.random.uniform(*friction_range)  # Randomize friction of all geoms except the base
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = agent.policy_network(state_tensor)
            action = np.random.choice(len(action_probs.squeeze().detach().numpy()), 
                                    p=action_probs.squeeze().detach().numpy())
            
            # Store state before taking action
            states.append(state)  # Store original numpy array state
            
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[0][action]))
            
            state = next_state
            
        # Convert states to numpy arrays if they aren't already
        states = [np.array(s) for s in states]
        agent.update_policy(rewards, log_probs, states)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")
    
    env.close()
    return agent


def train_sb3_agent(env_id, algorithm='PPO', total_timesteps=100000):
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
    # Train and evaluate with domain randomization
    print("Training with domain randomization...")
    train_with_domain_randomization()

    # Train and evaluate without domain randomization
    print("Training without domain randomization...")
    train_reinforce()

    # Train and evaluate using Actor-Critic
    print("Training with Actor-Critic...")
    train_actor_critic()

    # Train and evaluate using PPO
    print("Training with PPO...")
    train_sb3_agent('CustomHopper-source-v0', algorithm='PPO')

    # Train and evaluate using SAC
    print("Training with SAC...")
    train_sb3_agent('CustomHopper-source-v0', algorithm='SAC')


def train_reinforce(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    agent = REINFORCEAgent(env, learning_rate=learning_rate)
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        states = []  # Store states for policy update
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = agent.policy_network(state_tensor)
            action = np.random.choice(len(action_probs.squeeze().detach().numpy()), 
                                    p=action_probs.squeeze().detach().numpy())
            
            # Store state before taking action
            states.append(state)  # Store original numpy array state
            
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[0][action]))
            
            state = next_state
            
        # Convert states to numpy arrays if they aren't already
        states = [np.array(s) for s in states]
        agent.update_policy(rewards, log_probs, states)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")
    
    env.close()
    return agent


def evaluate_agent(env, agent, n_episodes=10):
    total_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Handle different agent types
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            else:  # ActorCriticAgent
                action, _, _ = agent.get_action(state, evaluation=True)
                action = action.detach().cpu().numpy()  # Detach before converting to numpy
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward: {episode_reward}")

    avg_reward = sum(total_rewards) / n_episodes
    print(f"Average Reward over {n_episodes} Evaluation Episodes: {avg_reward}")


def train_actor_critic(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = ActorCriticAgent(policy=policy, device='cpu')

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            agent.store_outcome(state, next_state, log_prob, reward, done)
            state = next_state
        actor_loss, critic_loss = agent.update_policy()

        # Logging
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

    # Evaluate the agent after training
    evaluate_agent(env, agent)


if __name__ == '__main__':
	main()