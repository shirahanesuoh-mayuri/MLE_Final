import numpy as np
import torch as T

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
# Need a list to store the value of the countdown of different bombs
    count_down = [-1, -1, -1, -1, -1]
    storage_c = [-1]
    storage_x_y = [-1, -1]
# To store the value
    _, _, bombholder, (x_self, y_self) = game_state["self"]
    feature_matrix_shape = game_state["field"].shape
    walls = game_state["field"]



#this part is ok
# To calculate the weight and danger field
    for _,_,bomber_man,(x_other, y_other) in game_state["others"]:
    # Here is the weight of others
        walls[x_other, y_other] = 75
        if bombholder == 0 or not bomber_man:
            for (x, y), c in game_state["bombs"]:
                storage_c = np.append(storage_c, c)
                storage_x_y = np.vstack((storage_x_y, (x, y)))
    storage_c = np.array(storage_c)
    storage_x_y = np.array(storage_x_y)
    index = np.where((storage_c >= 0) & (storage_c < 4))[0]
    if len(index) > 0 :
        for idx in index:
            x = storage_x_y[idx, 0]
            y = storage_x_y[idx, 1]
            walls[x, y] = -50
            count_down[idx-1] = storage_c[idx]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 4):
                    new_x, new_y = x + dx * i, y + dy * i
                    if walls[new_x, new_y] == -1:
                        break
                    walls[new_x, new_y] = -25 - 30 / (storage_c[idx] + 2)

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
# Here, the count_down is a list, so need an iteration to find all the danger field
# In this situation, this part need to be recode
#        for i in range(0, len(count_down)):
#            if count_down[i] in range(0, 4):
#                for (x, y), c in game_state["bombs"]:
#                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                        for i in range(1, 4):
#                            new_x, new_y = x + dx * i, y + dy * i
#                            if walls[new_x, new_y] == -1:
#                                break
#                            walls[new_x, new_y] = -25 - 30 / (c + 2)


# Here is the weight of coins
    for (x, y) in game_state["coins"]:
        walls[x, y] = 100




    field_matrix = np.copy(walls)
    up_situation = field_matrix[x_self, y_self-1]
    down_situation = field_matrix[x_self, y_self+1]
    left_situation = field_matrix[x_self-1, y_self]
    right_situation = field_matrix[x_self+1, y_self]
    my_situation = field_matrix[x_self, y_self]

# combine values to special structure matrix as feature
    game_feature = np.array(([up_situation, down_situation, left_situation, right_situation, my_situation]))
    game_feature = np.vstack((game_feature, count_down))
    #print(game_feature)
    #print(field_matrix)
    T.tensor(game_feature)





    return game_feature





