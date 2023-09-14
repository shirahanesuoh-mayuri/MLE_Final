import numpy as np
#original version
# def state_to_features(game_state: dict) -> np.array:
#     """
#     *This is not a required function, but an idea to structure your code.*
#
#     Converts the game state to the input of your model, i.e.
#     a feature vector.
#
#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.
#
#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
#     # This is the dict before the game begins and after it ends
#
#     if game_state is None:
#         return None
#
#     # For example, you could construct several channels of equal shape, ...
#     (x_self, y_self) = game_state["self"][3]
#     #feature_matrix_shape = game_state["field"].shape
#     walls = game_state["field"]
#
#     #coin_map =  np.zeros(feature_matrix_shape)
#     for (x, y) in game_state["coins"]:
#         walls[x, y] = 100
#
#     field_matrix = np.copy(walls)
#     up_situation = field_matrix[x_self, y_self-1]
#     down_situation = field_matrix[x_self, y_self+1]
#     left_situation = field_matrix[x_self-1, y_self]
#     right_situation = field_matrix[x_self+1, y_self]
#
#
#     coin_feature = np.array(([up_situation, down_situation, left_situation, right_situation]))
#
#     return coin_feature

#New version
def coin_distance_in_direction(agent_position, coins, direction):
    x, y = agent_position
    distances = []

    if direction == 'UP':
        distances = [y - coin_y for coin_x, coin_y in coins if coin_x == x and coin_y < y]
    elif direction == 'DOWN':
        distances = [coin_y - y for coin_x, coin_y in coins if coin_x == x and coin_y > y]
    elif direction == 'LEFT':
        distances = [x - coin_x for coin_x, coin_y in coins if coin_y == y and coin_x < x]
    elif direction == 'RIGHT':
        distances = [coin_x - x for coin_x, coin_y in coins if coin_y == y and coin_x > x]

    if distances:
        return min(distances)
    else:
        return None  # No coins in the given direction
def state_to_features(game_state: dict) -> np.array:
    features = []

    if game_state is None:
        return None
    # (x_self, y_self) = game_state["self"][3]
    # 获取位置、墙壁和硬币的信息
    x_self, y_self = game_state['self'][3]
    coins = game_state['coins']
    walls = game_state["field"]

    # 获取扩展的视野
    padded_matrix = np.pad(walls, 2, mode='constant', constant_values=0)
    x_self += 2
    y_self += 2
    extended_view = padded_matrix[x_self - 2:x_self + 3, y_self - 2:y_self + 3]

    # 为墙壁和硬币创建一个特征频道，这次在扩展的视野中
    coin_channel = (extended_view == 100).astype(int)
    wall_channel = (extended_view == -1).astype(int)

    # for (x, y) in game_state["coins"]:
    #     walls[x, y] = 100

    # field_matrix = np.copy(walls)

    for x, y in coins:
        rel_x, rel_y = x - x_self + 2, y - y_self + 2
        if 0 <= rel_x < 5 and 0 <= rel_y < 5:
            coin_channel[rel_x, rel_y] = 1

    wall_channel[extended_view == -1] = 1

    # 为field_matrix添加2的零填充
    # padded_matrix = np.pad(field_matrix, 2, mode='constant', constant_values=0)

    # 更新x_self和y_self的坐标以考虑到padding
    # x_self += 2
    # y_self += 2
    # extended_view = padded_matrix[x_self - 2:x_self + 3, y_self - 2:y_self + 3]

    # 1. 距离最近的金币的方向
    # coin_dists = [(x - x_self, y - y_self) for x, y in game_state["coins"]]
    # closest_coin_dist = min(coin_dists, key=lambda t: abs(t[0]) + abs(t[1]))
    # coin_dir = np.array(closest_coin_dist)

    # 计算金币方向特征
    coin_dists = [(x - x_self, y - y_self) for x, y in coins]
    closest_coin_dist = min(coin_dists, key=lambda t: abs(t[0]) + abs(t[1]))
    coin_dir_x, coin_dir_y = closest_coin_dist
    coin_direction_channel = np.zeros_like(extended_view)
    if 0 <= coin_dir_x + 2 < 5 and 0 <= coin_dir_y + 2 < 5:  # 防止索引超出范围
        coin_direction_channel[coin_dir_x + 2, coin_dir_y + 2] = 1

    # 2. 扩展的视野
    # extended_view = padded_matrix[x_self - 2:x_self + 3, y_self - 2:y_self + 3].flatten()

    # 3. New features based on coin distances in each direction
    #
    # 合并所有特征以形成通道
    features = np.stack([wall_channel, coin_channel, extended_view, coin_direction_channel], axis=0)

    return features


