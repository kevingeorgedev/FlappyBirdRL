import torch
import torch.nn as nn
import torch.optim as optim
from replay import Transition
from copy import deepcopy

class DuelDQNet(nn.Module):
    def __init__(self, n_observations, n_actions, seq_length, dim) -> None:
        super(DuelDQNet, self).__init__()

        self.dim = dim
        self.n_actions = n_actions

        self.projection = nn.Sequential(
            nn.Linear(n_observations, dim), # Projection Layer
            nn.ReLU()
        )

        self.positional_embedding = nn.Parameter(torch.randn(seq_length, dim))

        self.block1 = self._encoder_block()
        self.block2 = self._encoder_block()

        self.fc_adv = nn.Linear(dim, n_actions)
        self.fc_val = nn.Linear(dim, 1)

    def _encoder_block(self):
        return nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.MultiheadAttention(embed_dim=self.dim, num_heads=8, batch_first=True),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, x):
        x = self.projection(x) + self.positional_embedding

        residual1 = x
        x, _ = self.block1[1](self.block1[0](x).unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1) + residual1

        residual2 = x
        x, _ = self.block2[1](self.block2[0](x).unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1) + residual2

        adv = self.fc_adv(x).view(-1, self.n_actions)
        val = self.fc_val(x).view(-1, 1)
        
        qval = val + (adv - adv.mean(dim=1, keepdim=True))
        return qval
    
    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action[0].item()