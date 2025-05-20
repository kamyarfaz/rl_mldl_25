import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)
        
        return normal_dist, value


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class REINFORCEAgent:
    def __init__(self, env, learning_rate=1e-3, baseline=False):
        self.env = env
        self.policy_network = PolicyNetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.baseline = baseline
        self.value_network = CriticNetwork(state_dim=env.observation_space.shape[0]) if baseline else None

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy_network(state)
        action = np.random.choice(len(action_probs.squeeze().detach().numpy()), p=action_probs.squeeze().detach().numpy())
        return action

    def update_policy(self, rewards, log_probs, states):
        discounted_rewards = self.compute_discounted_rewards(rewards)
        if self.baseline:
            states = torch.tensor(states, dtype=torch.float32)
            values = self.value_network(states).squeeze()
            advantages = discounted_rewards - values.detach()

            # Update value network
            value_loss = nn.functional.mse_loss(values, discounted_rewards)
            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()
        else:
            advantages = discounted_rewards

        # Update policy network
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            # Ensure log_prob and advantage are at least 1-dimensional
            if log_prob.dim() == 0:
                log_prob = log_prob.unsqueeze(0)
            if isinstance(advantage, torch.Tensor) and advantage.dim() == 0:
                advantage = advantage.unsqueeze(0)
            policy_loss.append((-log_prob * advantage).unsqueeze(0))
        
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def compute_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return torch.tensor(discounted_rewards, dtype=torch.float32)


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


class ActorCriticAgent(Agent):
    def __init__(self, policy, device='cpu'):
        super().__init__(policy, device)
        self.value_loss_fn = nn.MSELoss()

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # Compute returns and advantages using bootstrapping
        with torch.no_grad():
            # Get value estimates for current states and next states
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            
            # Compute TD targets and advantages
            next_values = next_values.squeeze(-1)
            values = values.squeeze(-1)
            
            # Handle terminal states
            next_values = next_values * (1 - done)
            
            # Compute TD targets: r + gamma * V(s')
            td_targets = rewards + self.gamma * next_values
            
            # Compute advantages: TD target - V(s)
            advantages = td_targets - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute actor (policy) loss
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Compute critic (value) loss
        critic_loss = self.value_loss_fn(values, td_targets.detach())
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss  # 0.5 is a coefficient to balance actor and critic losses
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities, value """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None, value

        else:   # Sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob, value

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        super().store_outcome(state, next_state, action_log_prob, reward, done)


def train_actor_critic(env_name='CustomHopper-source-v0', n_episodes=1000, learning_rate=1e-3):
    # Initialize environment and agent
    env = gym.make(env_name)
    agent = ActorCriticAgent(policy=Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0]))

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
        total_reward = sum(agent.rewards)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

    # Evaluate the agent after training
    evaluate_agent(env, agent)

