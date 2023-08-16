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
    count_down = -1
    # For example, you could construct several channels of equal shape, ...
    _, _, bombholder, (x_self, y_self) = game_state["self"]
    feature_matrix_shape = game_state["field"].shape
    walls = game_state["field"]
    explosion = np.zeros(feature_matrix_shape)



    #coin_map =  np.zeros(feature_matrix_shape)
    if bombholder == 0:
        for (x, y), c in game_state["bombs"]:
            count_down = c
            walls[x, y] = -50
        if count_down in range(0, 4):
            for (x, y), c in game_state["bombs"]:
                for i in range(1, 4):
                    if walls[x-i, y] == -1:
                        break
                    walls[x-i, y] = -25-30/(c+2)
                for i in range(1, 4):
                    if walls[x, y-i] == -1:
                        break
                    walls[x, y-i] = -25-30/(c+2)
                for i in range(1, 4):
                    if walls[x, y+i] == -1:
                        break
                    walls[x, y+i] = -25-30/(c+2)
                for i in range(1, 4):
                    if walls[x+i, y] == -1:
                        break
                    walls[x+i, y] = -25-30/(c+2)
    for (x, y) in game_state["coins"]:
        walls[x, y] = 100
    print(count_down)
    field_matrix = np.copy(walls)
    up_situation = field_matrix[x_self, y_self-1]
    down_situation = field_matrix[x_self, y_self+1]
    left_situation = field_matrix[x_self-1, y_self]
    right_situation = field_matrix[x_self+1, y_self]
    my_situation = field_matrix[x_self, y_self]


    game_feature = np.array(([up_situation, down_situation, my_situation, left_situation, right_situation,count_down]))
    np.append(game_feature, count_down)
    print(game_state["step"])
    print(game_feature)
    print(walls)

    return game_feature





