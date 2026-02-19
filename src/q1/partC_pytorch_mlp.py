import torch.nn as nn
import torch.nn.functional as F

# class TorchMLP(nn.Module):
#     def __init__(self, in_dim: int, h1: int = 64, h2: int = 32, out_dim: int = 4):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, h1)
#         self.fc2 = nn.Linear(h1, h2)
#         self.fc3 = nn.Linear(h2, out_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)  # logits


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int = 128, h2: int = 64, h3: int = 32, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)