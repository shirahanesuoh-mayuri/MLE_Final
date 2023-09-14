import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np

from.callback_rule import act_rule
from .stateTofeatures import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQN(nn.Module):

    learning_rate = 0.0005

    def __init__(self, channel_in, channel_out):
        super(DQN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channel_in, out_channels= 128, kernel_size=2, stride=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels= 640, kernel_size=1, stride=1)
        self.linear0 = nn.Linear(in_features=4, out_features=256)
        self.linear1 = nn.Linear(in_features=256, out_features=channel_out)

        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    def forward(self, x):
        x = state_to_features(x).float()
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        print(x)
        return x
    def train_step(self, old_state, action, new_state, reward):
        if action is not None:

            state_action_value = self.forward(old_state).unsqueeze(0)

            target = T.tensor(ACTIONS.index(act_rule(old_state)), dtype=T.long).unsqueeze(0)

            loss = self.loss(state_action_value, target)

            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class Double_Q_Net(nn.Module):
    def __init__(self, featuren_in, output):
        super(Double_Q_Net, self).__init__()