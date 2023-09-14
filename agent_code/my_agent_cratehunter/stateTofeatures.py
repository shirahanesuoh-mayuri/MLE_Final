import numpy as np
import torch as T
import math

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
# Here is the original value of countdown and position, to avoid the None type in feature.
    position_b = [-1, -1]
    count_down = [-1]
    # For example, you could construct several channels of equal shape, ...
    _, _, bombholder, (x_self, y_self) = game_state["self"]
    walls = game_state["field"]





    #coin_map =  np.zeros(feature_matrix_shape)
    if 0 in bombholder :
        for i in range(0, len(game_state["bombs"])):
            for (x, y), c in game_state["bombs"]:
                count_down = c
                position_b = (x, y)
                walls[x, y] = -50
        # if count_down in range(0, 4):
        #     for (x, y), c in game_state["bombs"]:
        #         for i in range(1, 4):
        #             if walls[x-i, y] == -1:
        #                 break
        #             walls[x-i, y] = -25-30/(c+2)
        #         for i in range(1, 4):
        #             if walls[x, y-i] == -1:
        #                 break
        #             walls[x, y-i] = -25-30/(c+2)
        #         for i in range(1, 4):
        #             if walls[x, y+i] == -1:
        #                 break
        #             walls[x, y+i] = -25-30/(c+2)
        #         for i in range(1, 4):
        #             if walls[x+i, y] == -1:
        #                 break
        #             walls[x+i, y] = -25-30/(c+2)
##Here, we used a list of dx, dy tuples to represent the four possible directions: left (-1, 0), right (1, 0), up (0, -1) and down (0, 1 ).
#
# Then, we use a for loop to iterate through the directions and take up to three steps in each direction (range(1, 4)), just like you did in your original code.
#
# In this way, we have replaced the original four with two nested for loops and eliminated duplicate code.
        if count_down in range(0, 4):
            for (x, y), c in game_state["bombs"]:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for i in range(1, 4):
                        new_x, new_y = x + dx * i, y + dy * i
                        if walls[new_x, new_y] == -1:
                            break
                        walls[new_x, new_y] = -25 - 30 / (c + 2)
    for (x, y) in game_state["coins"]:
        walls[x, y] = 100


    field_matrix = np.copy(walls)
    up_situation = field_matrix[x_self, y_self-1]
    down_situation = field_matrix[x_self, y_self+1]
    left_situation = field_matrix[x_self-1, y_self]
    right_situation = field_matrix[x_self+1, y_self]
    my_situation = field_matrix[x_self, y_self]
#计算炸弹与角色之间的距离特征
    dis_feature_bomb = math.sqrt((x_self - position_b[0])**2 + (y_self - position_b[1])**2)

    game_feature = np.array(([up_situation, down_situation, left_situation, right_situation]))
    bomb_feature = np.append(position_b, count_down)
    bomb_feature = np.append(bomb_feature, my_situation)
    game_feature = np.vstack((game_feature, bomb_feature))


    T.tensor(game_feature)




    return game_feature



