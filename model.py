import torch
import torch.nn as nn
import torch.optim as optim
from replay import Transition

class DuelDQNet(nn.Module):
    def __init__(self, n_observations, n_actions, seq_length, dim) -> None:
        super(DuelDQNet, self).__init__()

        self.dim = dim
        self.n_actions = n_actions
        self.n_observations = n_observations

        self.projection = nn.Sequential(
            nn.Linear(n_observations, dim), # Projection Layer
            nn.ReLU()
        )

        self.positional_embedding = nn.Parameter(torch.randn(seq_length, dim))

        self.norm = nn.LayerNorm(self.dim)
        self.attn1 = nn.MultiheadAttention(embed_dim=self.dim, num_heads=8, batch_first=True)
        self.block1 = self._encoder_block()

        self.attn2 = nn.MultiheadAttention(embed_dim=self.dim, num_heads=8, batch_first=True)
        self.block2 = self._encoder_block()

        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def _encoder_block(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.GELU(),

            nn.Linear(self.dim, self.dim, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.projection(x)
        x += self.positional_embedding

        x = self.norm(x)
        x = x.unsqueeze(1)
        attn_output, _ = self.attn1(x, x, x)
        x = attn_output.squeeze(1)
        x = self.norm(x)
        x = self.block1(x)
        
        x = self.norm(x)
        x = x.unsqueeze(1)
        attn_output, _ = self.attn2(x, x, x)
        x = attn_output.squeeze(1)
        x = self.norm(x)
        x = self.block1(x)

        adv: torch.Tensor = self.fc_adv(x)
        adv = adv.view(-1, self.n_actions)
        val: torch.Tensor = self.fc_val(x)
        val = val.view(-1, 1)
        
        qval = val + (adv - adv.mean(dim=1, keepdim=True))
        return qval
    
    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action[0].item()