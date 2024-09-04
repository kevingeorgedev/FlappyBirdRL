from replay import ReplayMemory, Transition
from model import DuelDQNet
import torch
import torch.nn as nn
import torch.optim as optim

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

class Agent:
    def __init__(self, lr, n_obs, n_actions, gamma, device, seq_length) -> None:
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy = DuelDQNet(n_observations=n_obs, n_actions=n_actions, max_seq_length=seq_length, dim=512).to(device)
        self.target = DuelDQNet(n_observations=n_obs, n_actions=n_actions, max_seq_length=seq_length, dim=512).to(device)
        self.criterion = LogCoshLoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

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