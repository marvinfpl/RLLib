import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from buffer import RolloutBuffer

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden, init=False, activation=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x)->torch.Tensor:
        return self.net(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, n_states, hidden, init=False, activation=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x)->torch.Tensor:
        return self.net(x)
    
class PPOAgent:
    def __init__(self, env_name:str, gamma:float=0.99, lr_actor:float=4e-5, lr_critic:float=4e-5, eps_clip:float=0.2, max_grad_norm:float=1.0,
                lambda_gae:float=0.95, batch_size:int=128, hidden:int=128, rollouts:int=2048, epochs:int=10, is_action_continuous:bool=False,
                is_state_continuous:bool=False, n_env:int=1):
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.lambda_gae = lambda_gae
        self.batch_size = batch_size
        self.rollouts = rollouts
        self.epochs = epochs
        self.is_action_continuous = self.is_action_continuous
        self.is_state_continuous = self.is_state_continuous
        self.n_env = n_env

        env = gym.make(env_name)
        if is_state_continuous:
            n_states = env.observation_space.shape[0]
        else:
            n_states = env.observation_space.n
        if is_action_continuous:
            n_actions = env.action_space.shape[0]
        else:
            n_actions = env.action_space.n 

        env.close()

        self.actor = ActorNetwork(n_states, n_actions, hidden)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = CriticNetwork(n_states, hidden)

        self.buffer = RolloutBuffer()

    def compute_gae(self):
        pass
    
    def learn(self, env:gym.Env, timesteps:int=100_000):
        pass

    def evaluate(self, episodes:int=10):
        pass
