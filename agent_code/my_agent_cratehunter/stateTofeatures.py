import numpy as np

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    _, _, bombholder, (x_self, y_self) = game_state["self"]

    if bombholder == 0:
        (x_bomb, y_bomb) = game_state["bombs"][0]
    #feature_matrix_shape = game_state["field"].shape
    walls = game_state["field"]


    #coin_map =  np.zeros(feature_matrix_shape)
    for (x, y) in game_state["coins"]:
        walls[x, y] = 100
    for (x, y) in game_state["bombs"]:
        walls[x, y] = -100

    field_matrix = np.copy(walls)
    up_situation = field_matrix[x_self, y_self-1]
    down_situation = field_matrix[x_self, y_self+1]
    left_situation = field_matrix[x_self-1, y_self]
    right_situation = field_matrix[x_self+1, y_self]


    coin_feature = np.array(([up_situation, down_situation, left_situation, right_situation]))

    return coin_feature





