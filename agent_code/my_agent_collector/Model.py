import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np

from.callback_rule import act_rule
from .stateTofeatures import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class Net(nn.Module):
    def __init__(self, n_feature, n_output, n_hidden):
            super(Net,self).__init__()
            self.in_Layer = nn.Linear(n_feature, n_hidden)
            self.out_Layer = nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = self.in_Layer(x)
        x = F.relu(x)
        x = self.out_Layer(x)
        return x

class DQN():
    def __init__(self, n_features, n_actions):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = 20
        self.learning_rate = 0.0095
        self.gamma = 0.75
        self.epsilon_max = 0.8
        self.replace_target_iter = 200
        self.memory_size = 500
        self.batch_size = 32
        self.epsilon_increment = None
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features*2+2))

        self.loss_Func = nn.MSELoss()
        self.cost_his = []

        self.build_net()


    def build_net(self):
        self.q_Eval = Net(self.n_features, self.n_actions, self.n_hidden)
        self.q_Target = Net(self.n_features, self.n_actions, self.n_hidden)
        self.optimizer = torch.optim.RMSprop(self.q_Eval.parameters(), lr=self.learning_rate)


