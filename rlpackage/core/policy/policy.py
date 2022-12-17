"""Define policies"""
from copy import deepcopy
import torch
import numpy as np

from core.replay_buffer.replay_buffer import ArrayReplayBuffer
from core.model.model import DQNModel, PPOCritic, PPOPolicy
from core.environment.env import EnvInfo

from torch.distributions.normal import Normal

def freeze(model:torch.nn.Module):
    """Freeze Model parameters

    Args:
        model (torch.nn.Module): the model to freeze
    """

    for param in model.parameters():
        param.requires_grad = False

class Policy():
    """Base policy"""
    def __init__(self,
                 env_info:EnvInfo,
                 replay_buffer:ArrayReplayBuffer,
                 optimizer:torch.optim,
                 batch_size:int):

        self.env_info = env_info
        self.action_space = env_info.action_space
        self.observation_space = env_info.observation_space
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.optimizer = optimizer

    def act(self, observation:torch.tensor, deterministic:bool=False):
        """function that given a state returns the action"""
        raise NotImplementedError("This policy is not implemented yet.")

    def train(self):
        """
        Sample from the replay buffer to train the policy
        """
        raise NotImplementedError("The training policy is not implemented yet.")

class RandomPolicy(Policy):
    """Random Agent"""
    def act(self, observation:torch.tensor):
        return self.action_space.sample()

    def train(self):
        """There is no need to train"""
        return None

class PPO(Policy):
    """PPO implementation"""
    def __init__(self,
                 env_info:EnvInfo,
                 replay_buffer:ArrayReplayBuffer,
                 optimizer:torch.optim,
                 batch_size:int=256,
                 eps_clip:float=0.2,
                 gamma:float=0.99,
                 gae_lambda:float=0.95,
                 training_epochs:int=10,
                 horizon:int=100):

        super().__init__(env_info, replay_buffer, optimizer, batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.critic = PPOCritic(self.observation_space.shape[0], 1).to(self.device)
        self.policy = PPOPolicy(self.observation_space.shape[0],
                                self.action_space.shape[0]
                                ).to(self.device)

        self.old_critic = deepcopy(self.critic)
        self.old_policy = deepcopy(self.policy)

        self.critic.train()
        self.policy.train()
        self.old_critic.eval()
        self.old_policy.eval()
        freeze(self.old_critic)
        freeze(self.old_policy)

        self.eps_clip = eps_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.training_epochs = training_epochs
        self.horizon = horizon

        self.optimizer = optimizer([
                        {'params': self.critic.parameters(), 'lr': 1e-3},
                        {'params': self.policy.parameters(), 'lr': 1e-3}
                    ])

        self.train_step = 0


    def act(self, observation:torch.Tensor, deterministic:bool=False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mean, std = self.old_policy(obs)

        if deterministic:
            return mean.detach().cpu().numpy()

        dist = Normal(loc=mean, scale=std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), action_log_prob.detach().cpu().numpy()

    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return None, {}

        sample = self.replay_buffer.sample_trajectories(self.batch_size, horizon=self.horizon)

        eps_info = (torch.as_tensor(sample.obs, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(sample.act, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(sample.rew, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(sample.done, dtype=torch.long, device=self.device),
                                    torch.as_tensor(sample.next_obs, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(sample.act_log_prob, dtype=torch.float32, device=self.device)
        )
        obs, act, rew, done, next_obs, act_log_prob = eps_info
        obs, act, rew, done, next_obs, act_log_prob = obs[0], act[0], rew[0], done[0], next_obs[0], act_log_prob[0]
        episode_end = torch.nonzero(done) #TODO:Check if where sort the indicies
        if len(episode_end) == 0:
            mask = torch.ones(self.horizon, dtype=bool, device=self.device)
        else:
            mask = torch.zeros(self.horizon, dtype=bool, device=self.device)
            mask[:episode_end[0, 0]+1] = 1

        mean_loss = 0

        for _ in range(self.training_epochs):
            self.optimizer.zero_grad()
            adv = torch.zeros(self.horizon, device=self.device)
            value = self.critic(obs)[:, 0]
            next_value = self.critic(next_obs)[:, 0]

            with torch.no_grad():
                for t in reversed(range(self.horizon)):
                    delta = (rew[t] + (1-done[t]) * self.gamma * (next_value[t] - value[t])) * mask[t]
                    if t == self.horizon - 1:
                        adv[t] = delta
                    else:
                        adv[t] = delta + (self.gamma * self.gae_lambda) * adv[t+1]

            (mean, std), (mean_old, std_old) = self.policy(obs), self.old_policy(obs)
            log_probs = Normal(mean, std).log_prob(act)
            old_log_probs = Normal(mean_old, std_old).log_prob(act)
            ratio = torch.exp(log_probs - old_log_probs)
            adv = adv.unsqueeze(-1)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv

            policy_loss = -torch.min(surr1[mask], surr2[mask])
            value_loss = 0.5 * torch.nn.functional.mse_loss(rew[mask] + self.gamma * (1-done[mask]) * next_value[mask], value[mask])
            loss = torch.mean(policy_loss + value_loss)
            mean_loss += loss

            loss.backward()
            self.optimizer.step()

        self.train_step +=1

        self.old_policy.load_state_dict(self.policy.state_dict())

        return mean_loss/self.training_epochs, {"buffers", "envs"}

class DQN(Policy):
    """DQN implementation"""
    def __init__(self, env_info:EnvInfo, replay_buffer:ArrayReplayBuffer, optimizer:torch.optim, batch_size:int=256, epsilon:float=0.2, gamma:float=0.99):
        super().__init__(env_info=env_info, replay_buffer=replay_buffer, optimizer=optimizer, batch_size=batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.critic = DQNModel(self.observation_space.shape[0], self.action_space.n).to(self.device)
        self.target_critic = deepcopy(self.critic)
        freeze(self.target_critic)

        self.critic.train()
        self.optimizer = optimizer(self.critic.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon
        self.gamma = gamma
        self.train_step = 0

    def act(self, observation:torch.tensor, deterministic:bool=False):
        #If we tackle multiple environments, we return one action per env
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.shape[1:] == self.observation_space.shape: #In case we use multiple environments
            actions = torch.argmax(self.critic(obs), axis=-1).detach().cpu().numpy()
            mask = (np.random.rand(obs.shape[0]) < self.epsilon ) & (not deterministic)
            return np.where(mask, self.action_space.sample(), actions)


        if (torch.rand(1) < self.epsilon) and not deterministic:
            return self.action_space.sample()

        obs = torch.unsqueeze(torch.as_tensor(observation, dtype=torch.float32, device=self.device), axis=0)
        return torch.argmax(self.critic(obs), axis=-1)[0].item()

    def train(self):
        if (self.replay_buffer.size < 2 * self.batch_size):
            return None, {}

        sample = self.replay_buffer.sample(self.batch_size)

        obs, act, rew, done, next_obs = (torch.as_tensor(sample.obs, dtype=torch.float32, device=self.device),
                                        torch.as_tensor(sample.act, dtype=torch.long, device=self.device),
                                        torch.as_tensor(sample.rew, dtype=torch.float32, device=self.device),
                                        torch.as_tensor(sample.done, dtype=torch.long, device=self.device),
                                        torch.as_tensor(sample.next_obs, dtype=torch.float32, device=self.device)
        )

        self.optimizer.zero_grad()

        q = self.critic(obs)
        q_a = torch.gather(q, 1, torch.unsqueeze(act, dim=-1))[:, 0]
        loss = self.loss_fn(q_a, rew + self.gamma * (1-done) * torch.max(self.target_critic(next_obs), dim=-1)[0])

        loss.backward()
        self.optimizer.step()

        self.train_step +=1

        if self.train_step % 100 == 0:
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)

        return loss, {}


