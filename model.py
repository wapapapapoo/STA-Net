import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.eeg_net = nn.Sequential(
            nn.Conv1d(28, 64, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fnirs_net = nn.Sequential(
            nn.Conv1d(72, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, eeg, fnirs):
        eeg = self.eeg_net(eeg).squeeze(-1)
        fnirs = self.fnirs_net(fnirs).squeeze(-1)

        x = torch.cat([eeg, fnirs], dim=1)

        return self.fc(x)
