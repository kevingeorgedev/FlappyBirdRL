from replay import ReplayMemory, Transition
from model import DuelDQNet
import torch
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, lr, n_obs, n_actions, gamma, device, capacity, seq_length) -> None:
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy = DuelDQNet(n_observations=n_obs, n_actions=n_actions, seq_length=seq_length, dim=512).to(device)
        self.target = DuelDQNet(n_observations=n_obs, n_actions=n_actions, seq_length=seq_length, dim=512).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.replay = ReplayMemory(capacity)

    def train(self, batch: Transition):
        states      = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions     = torch.Tensor(batch.action).float()
        rewards     = torch.Tensor(batch.reward)
        masks       = torch.Tensor(batch.mask)

        logits      = self.policy(states).squeeze(1)
        next_logits = self.target(next_states).squeeze(1)
        
        logits = torch.sum(logits.mul(actions), dim=1)
        target = rewards + masks * self.gamma * next_logits.max(1)[0]

        loss = self.criterion(logits, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss