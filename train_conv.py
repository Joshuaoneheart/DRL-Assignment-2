import copy
import wandb
from tqdm import tqdm
from copy import deepcopy
import random
import math
import numpy as np
from collections import defaultdict, deque
from env import Game2048Env
import pickle
from functools import partial
from model import Conv
import torch
import torch.nn as nn
from torch.optim import Adam
from puct import MCTS_PUCT, PUCTNode


def softmax(x):
        # 為了數值穩定性，從每個元素中減去最大值
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def create_env_from_state(env, state, score):
    # Create a deep copy of the environment with the given state and score.
    new_env = copy.deepcopy(env)
    new_env.board = state.copy()
    new_env.score = score
    return new_env

def eval_board(board, n_empty): 
    grid = board

    utility = 0
    smoothness = 0

    big_t = np.sum(np.power(grid, 2))
    s_grid = np.sqrt(grid)
    smoothness -= np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
    smoothness -= np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
    smoothness -= np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
    smoothness -= np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
    smoothness -= np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
    smoothness -= np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
    
    empty_w = 100000
    smoothness_w = 3

    empty_u = n_empty * empty_w
    smooth_u = smoothness ** smoothness_w
    big_t_u = big_t

    utility += big_t
    utility += empty_u
    utility += smooth_u

    return utility

class NTupleApproximator:
    def __init__(self, board_size, model):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.model = model
        # Generate symmetrical transformations for each pattern
        

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        arr = np.array(pattern)
        res = set()

        # 旋轉 4 次
        for k in range(4):
            rotated = np.rot90(arr, k)
            res.add(tuple(map(tuple, rotated)))
            flipped = np.fliplr(rotated)  # 先旋轉再翻轉
            res.add(tuple(map(tuple, flipped)))

        return [list(map(list, r)) for r in res]


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        out = torch.zeros((30, 16))
        for i in range(16):
            out[self.tile_to_index(board[i // 4][i % 4])][i] = 1
        return out.float().to("cuda")

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        return self.model(self.get_feature(board).unsqueeze(0))[1]

import copy
import math
import random
import numpy as np
from collections import defaultdict

# TODO: Define the action transformation functions (i.e., rot90_action, rot180_action, etc.)
# Note: You have already defined transformation functions for patterns before.


# Note: PolicyApproximator is similar to the value approximator but differs in key aspects.
class PolicyApproximator:
    def __init__(self, board_size, model):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want.
        """
        self.board_size = board_size
        self.actions = [0, 1, 2, 3]
        # Weight structure: [pattern_idx][feature_key][action]
        self.model = model
        # Generate the 8 symmetrical transformations for each pattern and store their types.
        self.symmetry_patterns = []
        self.symmetry_types = []  # Store the type of symmetry transformation (rotation or reflection)

        # TODO: Define corresponding action transformation functions for each symmetry.


    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        pass

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        out = torch.zeros((30, 16))
        for i in range(16):
            out[self.tile_to_index(board[i // 4][i % 4])][i] = 1
        return out.to("cuda").float()

    def predict(self, board):
        # TODO: Predict the policy (probability distribution over actions) given the board state.
        return self.model(self.get_feature(board).unsqueeze(0))[0].squeeze(0)
def state_to_tuple(state):
    res = []
    for i in range(16):
        res.append(state[i // 4][i % 4])
    return tuple(res)
dp = {}
def chance(state, score, env, depth, value_approximator):
    empty_cells = list(zip(*np.where(state == 0)))
    n_empty = len(empty_cells)
    if state_to_tuple(state) in dp and dp[state_to_tuple(state)] < 6:
        return eval_board(state, n_empty) + value_approximator.value(state) 
    if depth >= 3:
        return eval_board(state, n_empty) + value_approximator.value(state) 

    if n_empty == 0:
        _, utility, _ = maximize(state, score, env, depth + 1, value_approximator)
        return utility
    chance_2 = (.9 * (1 / n_empty))
    chance_4 = (.1 * (1 / n_empty))
    possible_tiles = []
    for empty_cell in empty_cells:
        possible_tiles.append((empty_cell, 2, chance_2))
        possible_tiles.append((empty_cell, 4, chance_4))
    u_s = 0
    for t in possible_tiles:
        new_state = deepcopy(state)
        new_state[t[0][0]][t[0][1]] = t[1]
        _, utility, _ = maximize(new_state, score, env, depth + 1, value_approximator)
        u_s += utility * t[2]
    return u_s

def maximize(state, score, env, depth, value_approximator):
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    move_board = []
    for j in legal_moves:
        sim_env = create_env_from_state(env, state, score)
        next_state, new_score, done, info = sim_env.step(j)
        move_board.append((j, info["before_add"], new_score, done))
    max_utility = float('-inf')
    best_direction = None
    dist = np.zeros(4)
    for mb in move_board:
        if mb[3]:
            u = 0
        else:
            u = chance(mb[1], mb[2], env, depth + 1, value_approximator)
        dist[mb[0]] = u + mb[2] - score
        if u + mb[2] - score >= max_utility:
            max_utility = u + mb[2] - score
            best_direction = mb[0]
    return best_direction, max_utility, softmax(dist)

def self_play_training_policy_with_mcts(env, mcts, model, policy_approximator, num_episodes=50):
    gamma=0.99
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        states = []
        rewards = []
        next_states = []
        dists = []
        pbar = tqdm()
        previous_score = env.score

        while not done:
            # Create the root node for the TD-MCTS tree
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            best_action, _, target_distribution = maximize(state, previous_score, env, 0, value_approximator)
            prev_score = env.score
            pbar.set_postfix({"Score":prev_score})
            pbar.update(1)
            # Run multiple simulations to build the MCTS search tree


            next_state, reward, done, info = env.step(best_action)

            states.append(copy.deepcopy(state))
            rewards.append(reward - previous_score)
            next_states.append(copy.deepcopy(info["before_add"]))
            dists.append(best_action)
            previous_score = reward

            state = next_state
        value_losses = []
        policy_losses = []
        for i in reversed(range(len(states))):
            G = rewards[i] + gamma * value_approximator.value(states[i]).squeeze().squeeze()
            value = value_approximator.value(states[i]).squeeze().squeeze()
            dist = policy_approximator.predict(states[i])
            policy_loss = nn.CrossEntropyLoss()(dist, torch.tensor(dists[i]).long().to("cuda"))
            value_loss = nn.MSELoss()(value, G)
            TD_error = abs((G - value).item())
            dp[state_to_tuple(states[i])] = TD_error
            opt.zero_grad()
            (policy_loss + value_loss).backward()
            opt.step()
            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
        print(f"Policy Loss {np.mean(policy_losses)}, Value Loss {np.mean(value_losses)}")
        print(f"Episode {episode+1}/{num_episodes} finished, final score: {env.score}")

model = Conv().to("cuda")
model.load_state_dict(torch.load("model.pkl", map_location="cuda", weights_only=False))
value_approximator = NTupleApproximator(board_size=4, model=model)

policy_approximator = PolicyApproximator(board_size=4, model=model)
wandb.init(project= "DRL-2")
env = Game2048Env()
mcts = MCTS_PUCT(env, 
        value_approximator=value_approximator, 
        policy_approximator=policy_approximator,
        iterations=10,
        c_puct=1.41,
        rollout_depth=2,
        gamma=0.99)
with open("dp.pkl", "rb") as fp:
    dp = pickle.load(fp)

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
for _ in range(5000):
    self_play_training_policy_with_mcts(env, mcts, model, policy_approximator, num_episodes=1)
    with open("model.pkl", "wb") as fp:
        torch.save(model.state_dict(), fp)
    with open("dp.pkl", "wb") as fp:
        pickle.dump(dp, fp)
