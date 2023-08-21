import os
import pickle
import random
import torch as T
import torch.nn.functional as F

import numpy as np

from .Model import DQN
from .callback_rule import act_rule
from .stateTofeatures import state_to_features


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# this agent is for the shortest route for coin_collection(Mission 4.1)

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

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        #self.model = DQN(1, 6)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    # The code beyond is for model loading




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action = np.random.choice(ACTIONS, p=[.175, .175, .175, .175, .1, .2])
        return action
    #prediction = F.softmax(self.model.forward(game_state)).detach().numpy()
    #action = np.random.choice(ACTIONS, p=prediction.flatten())
    action = np.random.choice(ACTIONS, p=[.125, .125, .125, .125, .1, .4])
    self.logger.debug("Querying model for action.")
    #self.logger.debug(f'prediction:{prediction}')
    self.logger.debug(f'step:{game_state["step"]}')
    self.logger.debug(f'feature:{state_to_features(game_state)}')

    #self.logger.debug(f'game state:\n{game_state["field"]}')
    #self.logger.debug(f'coins:{game_state["coins"]}')
    #self.logger.debug(f'self: {game_state["self"][3]}')
    self.logger.debug(f'action:{action}')



    return action








