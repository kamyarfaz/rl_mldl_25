"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy, REINFORCEAgent, ActorCriticAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model.mdl")

	

def train_reinforce(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    agent = REINFORCEAgent(env, learning_rate)

    for episode in range(n_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_prob = torch.log(torch.tensor(agent.policy_network(state)[action]))
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Update policy
        agent.update_policy(rewards, log_probs)

        # Logging
        total_reward = sum(rewards)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


def evaluate_agent(env, agent, n_episodes=10):
    total_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward: {episode_reward}")

    avg_reward = sum(total_rewards) / n_episodes
    print(f"Average Reward over {n_episodes} Evaluation Episodes: {avg_reward}")


def train_actor_critic(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    agent = ActorCriticAgent(env, learning_rate)

    for episode in range(n_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        states = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_prob = torch.log(torch.tensor(agent.policy_network(state)[action]))
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)
            state = next_state

        # Update policy
        agent.update_policy(rewards, log_probs, states)

        # Logging
        total_reward = sum(rewards)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # Evaluate the agent after training
    evaluate_agent(env, agent)


if __name__ == '__main__':
	train_actor_critic()