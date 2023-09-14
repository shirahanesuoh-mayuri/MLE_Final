from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .stateTofeatures import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if old_game_state["self"][1] == new_game_state["self"][1]:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    # Check coins in the agent's vicinity
    coins_nearby = count_coins_in_vicinity(new_game_state, radius=3)  # you can set the radius as you like
    if coins_nearby > 1:  # if there's more than one coin nearby, give a reward
        reward = 0  # Initialize reward
        reward += coins_nearby * 50  # for example, give a reward of 50 for each coin in vicinity


def count_coins_in_vicinity(game_state, radius):
    """Count coins in the agent's vicinity."""
    agent_position = game_state["self"][3]
    coins = game_state["coins"]
    coins_in_vicinity = [coin for coin in coins if max(abs(agent_position[0] - coin[0]), abs(agent_position[1] - coin[1])) <= radius]
    return len(coins_in_vicinity)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,

        #e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -10,  # idea: the custom event is bad
        # e.MOVED_UP: 0.5,
        # e.MOVED_DOWN: 0.5,
        # e.MOVED_LEFT: 0.5,
        # e.MOVED_RIGHT: 0.5,
        e.INVALID_ACTION: -30,
        e.WAITED: -30,
        # e.BOMB_DROPPED: -2,  # Dropped a bomb
        # e.CRATE_DESTROYED: 10,  # Destroyed a crate
        # e.KILLED_SELF: -20,  # Killed itself
        # e.KILLED_OPPONENT: 20  # Killed an opponent
        #e.CRATE_DESTROYED:
    }
    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    # reward_sum = 0
    # for event in events:
    #     if event in game_rewards:
    #         reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

