import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import settings


@dataclass
class PolicyAction:
    pressed: dict

    def __getitem__(self, key):
        return self.pressed.get(key, False)


class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc = nn.Linear(obs_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, obs, hx=None):
        x = torch.tanh(self.fc(obs))
        x, (h, c) = self.lstm(x, hx)
        logits = self.policy(x)
        value = self.value(x)
        return logits, value, (h, c)


class PPOAgent:
    def __init__(self, obs_size: int, action_size: int):
        self.device = torch.device("cpu")
        self.model = ActorCriticLSTM(obs_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0007)
        self.hidden = None
        self.action_size = action_size

    def reset_hidden(self):
        self.hidden = None

    def act(self, obs_tensor: torch.Tensor, explore: bool = True):
        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,obs)
        logits, value, self.hidden = self.model(obs_tensor, self.hidden)
        logits = logits.squeeze(0).squeeze(0)
        value = value.squeeze(0).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits)
        logprob = dist.log_prob(action)
        return action.item(), logprob.detach(), value.detach(), dist.entropy().detach()

    def evaluate_actions(self, obs_batch, actions, h0=None):
        logits, values, _ = self.model(obs_batch, h0)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, values.squeeze(-1), entropy

    def update(self, storage):
        # storage fields are already batched tensors (B,T,...)
        obs = storage["obs"].to(self.device)
        actions = storage["actions"].to(self.device)
        old_logprobs = storage["logprobs"].to(self.device)
        returns = storage["returns"].to(self.device)
        advantages = storage["advantages"].to(self.device)
        B, T, _ = obs.shape
        clip_eps = settings.PPO_CLIP
        value_coef = settings.PPO_VALUE_COEF
        entropy_coef = settings.PPO_ENTROPY_COEF

        for _ in range(settings.PPO_EPOCHS):
            logprobs, values, entropy = self.evaluate_actions(obs, actions)
            ratio = torch.exp(logprobs - old_logprobs)
            adv = advantages
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.PPO_MAX_GRAD_NORM)
            self.optimizer.step()


class RolloutStorage:
    def __init__(self, n_steps, obs_size):
        self.n_steps = n_steps
        self.obs_size = obs_size
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_advantages(self, last_value):
        T = len(self.rewards)
        advantages = [0] * T
        returns = [0] * T
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(self.dones[t])
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + settings.GAMMA * next_value * mask - self.values[t]
            gae = delta + settings.GAMMA * settings.LAMBDA * mask * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
        return returns, advantages
