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
                is_state_continuous:bool=False, n_env:int=1, warmup:int=1000, entropy_coef:float=0.02, value_loss_coef:float=0.5):
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
        self.warmup = warmup
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        env = gym.make(env_name)
        if is_state_continuous:
            self.n_states = env.observation_space.shape[0]
        else:
            self.n_states = env.observation_space.n
        if is_action_continuous:
            self.n_actions = env.action_space.shape[0]
        else:
            self.n_actions = env.action_space.n 

        env.close()

        self.actor = ActorNetwork(self.n_states, self.n_actions, hidden)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = CriticNetwork(self.n_states, hidden)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def get_state(self, state)->torch.Tensor:
        if self.is_state_continuous:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            if state.dim == 1:
                state_t = torch.zeros(self.n_states) # is this okay for a tensor input? probably not
                state_t[state] = 1.0
            else:
                states = []
                for s in state:
                    s_t = torch.zeros(self.n_states)
                    s_t[s] = 1.0
                    states.append(s_t)
                    state_t = torch.stack(states)
        return state_t
    
    def evaluate_action(self, state):#->int torch.Tensor, torch.Tensor, torch.Tensor: handle multi outputs?
        state_t = self.get_state(state)
        logits = self.actor.forward(state_t)
        value = self.critic.forward(state_t)
        if self.is_action_continuous:
            dist = torch.distributions.Normal(mu=logits.mean(), scale=torch.ones(1.0))
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropies = dist.entropy()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropies = dist.entropy()
        return action.item(), value, logprob, entropies

    def compute_gae(self, next_value):
        advantages = []
        gae = 0.0
        values_t = torch.cat(self.buffer.values + [next_value])
        dones_t = torch.tensor(self.buffer.dones)
        rewards_t = torch.tensor(self.buffer.rewards)
        for t in reversed(range(rewards_t.size(0))):
            delta = rewards_t[t] + self.gamma * values_t[t+1] * (1.0 - dones_t[t]) - values_t[t]
            gae = self.lambda_gae * gae * (1.0 - dones_t[t]) + delta
            advantages.insert(0, gae)
        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def train(self, next_value):
        if len(self.buffer) < self.warmup:
            return
        
        states = torch.from_numpy(self.buffer.states).float()
        values = torch.cat(self.buffer.values).unsqueeze(-1)
        old_logprobs = torch.cat(self.buffer.logprobs).unsqueeze(-1)

        advantages = self.compute_gae(next_value)
        returns = advantages + values

        for _ in range(self.epochs):
            idx = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.batch_size):
                end = start+self.batch_size
                mb_idx = idx[start:end]

                mb_states = states[mb_idx]
                mb_values = values[mb_idx].detach()
                mb_logp = old_logprobs[mb_idx].detach()
                mb_advantages = advantages[mb_idx].detach()
                mb_returns = returns[mb_idx].detach()

                _, new_values, new_logp, entropies = self.evaluate_action(mb_states)

                ratio = torch.exp(new_logp - mb_logp)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).pow(2).mean() - self.entropy_coef * entropies.mean()

                self.optim_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim_actor.step()

                clipped_values = torch.clamp(new_values - mb_values, -self.eps_clip, self.eps_clip) + mb_values
                unclipped_values = mb_returns - new_values
                critic_loss = torch.max(clipped_values, unclipped_values).pow(2).mean()
                
                self.optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim_critic.step()
                
        self.buffer.clear()



    def learn(self, env:gym.Env, timesteps:int=100_000):
        state, _ = env.reset()

        for step in range(1,1+timesteps):
            action, value, logprob, _ = self.evaluate_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.append(state, action, reward, done, value, logprob)

            state = next_state

            if done:
                state, _ = env.reset()

            if step % self.rollouts == 0:
                _, next_value, _ = self.evaluate_action(state)
                self.train(next_value)

        env.close()
            

    def evaluate(self, env: gym.Env, episodes:int=10):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(episodes):
            action, _, _, _ = self.evaluate_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                print(f"Total Reward: {total_reward}")
                total_reward = 0.0
                state, _ = env.reset()
        env.close()


if __name__ == '__main__':
    env_name = "MountainCar-v0"
    model = PPOAgent(env_name=env_name)
    train_env = gym.make(env_name)
    model.learn(train_env, 300_000)
    eval_env = gym.make(env_name, render_mode="human")
    model.evaluate(eval_env)