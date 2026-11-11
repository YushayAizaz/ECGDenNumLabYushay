import torch.nn as nn
from typing import Any


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2),
            
            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2)
        )
        self.fc = nn.Linear(1404, 2500)

    def forward(self, inp: Any):
        x_c = self.cnn(inp)
        x_c = x_c.view(x_c.size(0), 1, -1)
        return self.fc(x_c)
