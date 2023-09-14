import numpy as np
import torch
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
    position_b = []
    count_down = -1
    # For example, you could construct several channels of equal shape, ...
    _, _, bombholder, (x_self, y_self) = game_state["self"]
    walls = game_state["field"]
    field_matrix = np.copy(walls)





    #coin_map =  np.zeros(feature_matrix_shape)
    if not bombholder:
        for i in range(0, len(game_state["bombs"])):
            for (x, y), c in game_state["bombs"]:
                count_down = c
                position_b = (x, y)
                field_matrix[x, y] = -50
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
                        if field_matrix[new_x, new_y] == -1:
                            break
                        field_matrix[new_x, new_y] = -25 - 30 / (c + 2)
    for (x, y) in game_state["coins"]:
        field_matrix[x, y] = 100



    up_situation = field_matrix[x_self, y_self-1]
    up_field = walls[x_self, y_self-1]
    down_situation = field_matrix[x_self, y_self+1]
    down_field = walls[x_self, y_self+1]
    left_situation = field_matrix[x_self-1, y_self]
    left_field = walls[x_self-1, y_self]
    right_situation = field_matrix[x_self+1, y_self]
    right_field = walls[x_self+1, y_self]
    my_situation = field_matrix[x_self, y_self]
    my_field = walls[x_self, y_self]
#计算炸弹与角色之间的距离特征
    if len(position_b) == 0:
        dis_feature = np.zeros(5)
    else:
        dis_feature_bomb = math.sqrt((x_self - position_b[0])**2 + (y_self - position_b[1])**2)
        dis_feature_bomb_up = math.sqrt((x_self - position_b[0])**2 + (y_self-1 - position_b[1])**2)
        dis_feature_bomb_down = math.sqrt((x_self - position_b[0]) ** 2 + (y_self+1 - position_b[1]) ** 2)
        dis_feature_bomb_left = math.sqrt((x_self-1 - position_b[0]) ** 2 + (y_self - position_b[1]) ** 2)
        dis_feature_bomb_right = math.sqrt((x_self+1 - position_b[0]) ** 2 + (y_self - position_b[1]) ** 2)
        dis_feature = np.array(
            ([dis_feature_bomb_up, dis_feature_bomb_down, dis_feature_bomb_left, dis_feature_bomb_right, dis_feature_bomb]))
#向箱子和墙方向添加权重,使其再有炸弹存在的情况下更加危险，向炸弹距离施加反比例权重距离越远越好，减小可通行地块的危险度（即使它正在爆炸路径上）
    situation_feature = np.array(([up_situation, down_situation, left_situation, right_situation, my_situation]))
    field_feature = np.array(([up_field, down_field, left_field, right_field, my_field]))
    death_risk = np.sum(np.abs(field_feature))
    for idx in range(0, len(situation_feature)):
        s_f = situation_feature[idx]
        f_f = field_feature[idx]
        if s_f == f_f:
            if f_f == 1:
                s_f = -35 - dis_feature[idx]
            if f_f == -1:
                s_f = -35 - dis_feature[idx]
            if f_f == 0:
                s_f = 25 + dis_feature[idx]
            situation_feature[idx]=s_f

        if s_f < -1:
            if f_f == 1:
                s_f = s_f - 10 + dis_feature[idx]*10
            if f_f == -1:
                s_f = s_f - 10 + dis_feature[idx]*10
            if f_f == 0:
                s_f = s_f + 10 + dis_feature[idx]*10
            situation_feature[idx] = s_f




    #feature_layer1 = situation_feature[:4]
    #feature_layer2 = field_feature[:4]
    #feature_layer3 = np.array([situation_feature[4], field_feature[4], count_down, death_risk])
    #game_feature = np.vstack((feature_layer1, feature_layer2))
    #game_feature = np.vstack((game_feature, feature_layer3))

    game_feature = np.vstack((situation_feature, field_feature))
    game_feature = torch.unsqueeze(torch.from_numpy(game_feature), dim=0)



    return game_feature



