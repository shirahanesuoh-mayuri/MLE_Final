import os
import pickle
import random
import torch as T
import torch.nn.functional as F
import numpy as np

from .Model import DQN
from .callback_rule import act_rule
from .stateTofeatures import state_to_features
from typing import List
#New added code
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99  # discount factor
MAX_EPSILON = 1  # exploration rate at start
MIN_EPSILON = 0.1  # minimum exploration rate
EPSILON_DECAY_DURATION = 10000  # number of steps over which exploration rate is reduced

# this agent is for the shortest route for coin_collection(Mission 4.1)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    self.steps_done = 0

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DQN(4, 5)
    else:
        self.logger.info("Loading model from saved state.")
        self.model = T.load("my-saved-model.pt")  # Using torch.load()
    # The code beyond is for model loading
    self.optimizer = T.optim.RMSprop(self.model.parameters())

#old version
# def act(self, game_state: dict) -> str:
#     """
#     Your agent should parse the input, think, and take a decision.
#     When not in training mode, the maximum execution time for this method is 0.5s.
#
#     param self: The same object that is passed to all of your callbacks.
#     param game_state: The dictionary that describes everything on the board.
#     :return: The action to take as a string.
#     """
#      todo Exploration vs exploitation
#     sample = random.random()
#     self.steps_done += 1
#     epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
#               np.exp(-1. * self.steps_done / EPSILON_DECAY_DURATION)
#
#     if self.train and sample < epsilon:
#         return np.random.choice(ACTIONS)
#
#     with T.no_grad():
#         return np.random.choice(ACTIONS, p=F.softmax(self.model.forward(game_state), dim=0).detach().numpy())


#new version
def act(self, game_state: dict) -> str:
    self.steps_done += 1

    # Calculate epsilon
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                np.exp(-1. * self.steps_done / EPSILON_DECAY_DURATION)

    # Exploration vs exploitation
    if self.train and random.random() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        with T.no_grad():
           features = state_to_features(game_state)
           # features_tensor = T.tensor(features, dtype=T.float)
           #features_tensor = state_to_features(game_state)
           features_tensor = T.tensor(features, dtype=T.float)
           action_probs = F.softmax(self.model.forward(features_tensor), dim=1).detach().numpy()
           # print("Before squeeze:", action_probs.shape)  # 查看原始的shape
           # print(f"Sum of action_probs: {action_probs.sum()}")
           action_probs = action_probs / action_probs.sum()
           action_probs = action_probs.squeeze()
           # print("After squeeze:", action_probs.shape)  # 确认更改后的shape
           return np.random.choice(ACTIONS, p=action_probs)

    def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
        T.save(self.model.state_dict(), "my-saved-model.pt")

    if self.train:
        next_state = state_to_features(game_state)  # 获取下一个状态
        reward = ...  # 根据game_state或events计算奖励
        action_index = ACTIONS.index(action)  # 将动作字符串转换为索引
        self.memory.push(features_tensor, action_index, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        # Convert batches to tensors
        batch_state = T.tensor(batch_state, dtype=T.float)
        batch_action = T.tensor(batch_action, dtype=T.long)
        batch_next_state = T.tensor(batch_next_state, dtype=T.float)
        batch_reward = T.tensor(batch_reward, dtype=T.float)

        # Compute Q values
        state_action_values = self.model(batch_state).gather(1, batch_action.unsqueeze(-1))
        next_state_values = self.model(batch_next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + batch_reward

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     action = np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1])
    #     return action
    #
    # action = np.random.choice(ACTIONS, p=F.softmax(self.model.forward(game_state), dim=0).detach().numpy())
    #
    # self.logger.debug("Querying model for action.")
    #
    # self.logger.debug(f'step:{game_state["step"]}')
    # self.logger.debug(f'action:{action}')
    #
    #
    #
    # return action








