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
        '''self.conv0 = nn.Conv2d(in_channels=channel_in, out_channels= 16, kernel_size=1, stride=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels= 64, kernel_size=1, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()'''
        self.linear0 = nn.Linear(in_features=channel_in, out_features=64)
        self.linear1 = nn.Linear(in_features=64, out_features= 32)
        self.linear2 = nn.Linear(in_features=32, out_features=channel_out)

        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    def forward(self, x):
        x = state_to_features(x)
        """ x = self.conv0(x)
         x = self.relu0(x)
         x = self.conv1(x)
         x = self.relu1(x)
         x = self.conv2(x)
         x = self.relu2(x)
         x = torch.flatten(x, 1)"""
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
    def train_step(self, old_state, action, new_state, reward):
        if action is not None:
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