from collections import deque
from random import shuffle

import numpy as np
# import callbacks


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    """Original Code"""
    # if len(targets) == 0: return None
    #
    # frontier = [start]
    # parent_dict = {start: start}
    # dist_so_far = {start: 0}
    # best = start
    # best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    #
    # while len(frontier) > 0:
    #     current = frontier.pop(0)
    #     # Find distance from current position to all targets, track closest
    #     d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
    #     if d + dist_so_far[current] <= best_dist:
    #         best = current
    #         best_dist = d + dist_so_far[current]
    #     if d == 0:
    #         # Found path to a target's exact position, mission accomplished!
    #         best = current
    #         break
    #     # Add unexplored free neighboring tiles to the queue in a random order
    #     x, y = current
    #     neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
    #     shuffle(neighbors)
    #     for neighbor in neighbors:
    #         if neighbor not in parent_dict:
    #             frontier.append(neighbor)
    #             parent_dict[neighbor] = current
    #             dist_so_far[neighbor] = dist_so_far[current] + 1
    # if logger: logger.debug(f'Suitable target found at {best}')
    # # Determine the first step towards the best found target tile
    # current = best
    # while True:
    #     if parent_dict[current] == start: return current
    #     current = parent_dict[current]
    """New added code"""
    """Find direction of closest target that can be reached via free tiles."""
    if not targets:
        return None

    return bfs_for_targets(free_space, start, targets)
"""New added code"""
def bfs_for_targets(free_space, start, targets):
    """Use BFS to find the closest target."""
    frontier = deque([start])
    parent_dict = {start: start}

    while frontier:
        current = frontier.popleft()

        # If a target is found
        if current in targets:
            break

        # Neighbors: UP, DOWN, LEFT, RIGHT
        x, y = current
        neighbors = [(nx, ny) for nx, ny in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                     if free_space[nx, ny] and (nx, ny) not in parent_dict]

        for neighbor in neighbors:
            frontier.append(neighbor)
            parent_dict[neighbor] = current

    # If no target found, return None
    if current not in targets:
        return None

    # Determine the first step towards the best found target tile
    while parent_dict[current] != start:
        current = parent_dict[current]
    return current
def determine_possible_moves(arena, game_state, x, y):
    """Determine possible moves from the current position."""
    directions = [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    bombs = [pos for pos, timer in game_state['bombs']]
    others = [pos for _, _, _, pos in game_state['others']]

    valid_tiles = [(i, j) for i, j in directions
                   if arena[i, j] == 0 and
                   game_state['explosion_map'][i, j] <= 1 and
                   (i, j) not in bombs and
                   (i, j) not in others]
    return valid_tiles

def get_valid_actions(valid_tiles, x, y):
    """Get a list of valid actions based on valid tiles."""
    actions = []
    if (x-1, y) in valid_tiles: actions.append('LEFT')
    if (x+1, y) in valid_tiles: actions.append('RIGHT')
    if (x, y-1) in valid_tiles: actions.append('UP')
    if (x, y+1) in valid_tiles: actions.append('DOWN')
    if (x, y) in valid_tiles: actions.append('WAIT')
    return actions

def act_rule(game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    #self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    #if game_state["round"] != self.current_round:
    #    reset_self(self)
    #    self.current_round = game_state["round"]
    # Gather information about the game state
    """Old version code"""

    # """arena = game_state['field']
    # _, score, bombs_left, (x, y) = game_state['self']
    # bombs = game_state['bombs']
    # bomb_xys = [xy for (xy, t) in bombs]
    # others = [xy for (n, s, b, xy) in game_state['others']]
    # coins = game_state['coins']
    # bomb_map = np.ones(arena.shape) * 5
    # for (xb, yb), t in bombs:
    #     for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
    #         if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #             bomb_map[i, j] = min(bomb_map[i, j], t)
    #
    # # If agent has been in the same location three times recently, it's a loop
    # #if self.coordinate_history.count((x, y)) > 2:
    # #    self.ignore_others_timer = 5
    # #else:
    # #    self.ignore_others_timer -= 1
    # #self.coordinate_history.append((x, y))
    #
    # # Check which moves make sense at all
    # directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    # valid_tiles, valid_actions = [], []
    # for d in directions:
    #     if ((arena[d] == 0) and
    #             (game_state['explosion_map'][d] <= 1) and
    #             (bomb_map[d] > 0) and
    #             (not d in others) and
    #             (not d in bomb_xys)):
    #         valid_tiles.append(d)
    # if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    # if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    # if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    # if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    # if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    # #if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    # #self.logger.debug(f'Valid actions: {valid_actions}')
    #
    # # Collect basic action proposals in a queue
    # # Later on, the last added action that is also valid will be chosen
    # action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    # shuffle(action_ideas)
    #
    # # Compile a list of 'targets' the agent should head towards
    # dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
    #              and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    # crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    # targets = coins + dead_ends + crates
    # # Add other agents as targets if in hunting mode or no crates/coins left
    # #if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
    # #    targets.extend(others)
    #
    # # Exclude targets that are currently occupied by a bomb
    # targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    #
    # # Take a step towards the most immediately interesting target
    # free_space = arena == 0
    # #if self.ignore_others_timer > 0:
    # #    for o in others:
    # #        free_space[o] = False
    # d = look_for_targets(free_space, (x, y), targets, None)
    # if d == (x, y - 1): action_ideas.append('UP')
    # if d == (x, y + 1): action_ideas.append('DOWN')
    # if d == (x - 1, y): action_ideas.append('LEFT')
    # if d == (x + 1, y): action_ideas.append('RIGHT')
    # #if d is None:
    # #    self.logger.debug('All targets gone, nothing to do anymore')
    # #    action_ideas.append('WAIT')
    #
    # # Add proposal to drop a bomb if at dead end
    # if (x, y) in dead_ends:
    #     action_ideas.append('BOMB')
    # # Add proposal to drop a bomb if touching an opponent
    # if len(others) > 0:
    #     if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
    #         action_ideas.append('BOMB')
    # # Add proposal to drop a bomb if arrived at target and touching crate
    # if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
    #     action_ideas.append('BOMB')
    #
    # # Add proposal to run away from any nearby bomb about to blow
    # for (xb, yb), t in bombs:
    #     if (xb == x) and (abs(yb - y) < 4):
    #         # Run away
    #         if (yb > y): action_ideas.append('UP')
    #         if (yb < y): action_ideas.append('DOWN')
    #         # If possible, turn a corner
    #         action_ideas.append('LEFT')
    #         action_ideas.append('RIGHT')
    #     if (yb == y) and (abs(xb - x) < 4):
    #         # Run away
    #         if (xb > x): action_ideas.append('LEFT')
    #         if (xb < x): action_ideas.append('RIGHT')
    #         # If possible, turn a corner
    #         action_ideas.append('UP')
    #         action_ideas.append('DOWN')
    # # Try random direction if directly on top of a bomb
    # for (xb, yb), t in bombs:
    #     if xb == x and yb == y:
    #         action_ideas.extend(action_ideas[:4])
    #
    # # Pick last action added to the proposals list that is also valid
    # while len(action_ideas) > 0:
    #     a = action_ideas.pop()
    #     if a in valid_actions:
    #         # Keep track of chosen action for cycle detection
    #         if a == 'BOMB':
    #             #self.bomb_history.append((x, y))
    #             continue
    #
    #         return a
    """New version"""
    """Determine the agent's next action."""
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']

    # Determine possible moves
    valid_tiles = determine_possible_moves(arena, game_state, x, y)
    valid_actions = get_valid_actions(valid_tiles, x, y)

    # Prioritize targets: Coins > Dead Ends > Crates > Others
    coins = game_state['coins']
    dead_ends = [(i, j) for i in range(1, 16) for j in range(1, 16)
             if arena[i, j] == 0 and
             sum(1 for dx, dy in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                 if arena[dx, dy] == 0) == 1]
    crates = [(i, j) for i in range(1, 16) for j in range(1, 16) if arena[i, j] == 1]
    others = [pos for _, _, _, pos in game_state['others']]

    targets = coins + dead_ends + crates + others

    # Exclude targets with bombs
    bombs = [pos for pos, _ in game_state['bombs']]
    targets = [t for t in targets if t not in bombs]

    # Find the best direction to move towards target
    best_direction = look_for_targets(arena == 0, (x, y), targets)

    # Prioritize bombing when in dead end or next to an opponent
    if best_direction:
         if best_direction == (x, y-1) and 'UP' in valid_actions:
            return 'UP'
         if best_direction == (x, y+1) and 'DOWN' in valid_actions:
            return 'DOWN'
         if best_direction == (x-1, y) and 'LEFT' in valid_actions:
           return 'LEFT'
         if best_direction == (x+1, y) and 'RIGHT' in valid_actions:
           return 'RIGHT'
    if (x, y) in dead_ends and 'BOMB' in valid_actions:
        return 'BOMB'
    if any(abs(o_x - x) <= 1 and abs(o_y - y) <= 1 for o_x, o_y in others) and 'BOMB' in valid_actions:
        return 'BOMB'

    # Default to WAIT if no action is decided
        return 'WAIT'