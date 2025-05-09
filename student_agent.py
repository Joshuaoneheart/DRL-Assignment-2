# Remember to adjust your student ID in meta.xml
import os
import ctypes
import numpy as np
import pickle
import random
import gym
from copy import deepcopy
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

row_left_table = [0] * 65536
row_right_table = [0] * 65536
col_up_table = [0] * 65536
col_down_table = [0] * 65536
heur_score_table = [0] * 65536
score_table = [0] * 65536
ROW_MASK = 0xFFFF
FULL_MASK = 0xFFFFFFFFFFFFFFFF
COL_MASK = 0x000F000F000F000F
SCORE_LOST_PENALTY = 200000.0
SCORE_MONOTONICITY_POWER = 4.0
SCORE_MONOTONICITY_WEIGHT = 47.0
SCORE_SUM_POWER = 3.5
SCORE_SUM_WEIGHT = 11.0
SCORE_MERGES_WEIGHT = 700.0
SCORE_EMPTY_WEIGHT = 270.0

def unpack_col(row):
    tmp = row
    return (tmp | (tmp << 12) | (tmp << 24) | (tmp << 36)) & COL_MASK

def reverse_row(row):
    return ((row >> 12) | ((row >> 4) & 0x00F0)  | ((row << 4) & 0x0F00) | (row << 12)) & ROW_MASK

def init_tables():
    for row in range(65536):
        line = [(row >>  0) & 0xf,
                (row >>  4) & 0xf,
                (row >>  8) & 0xf,
                (row >> 12) & 0xf]

        score = 0
        for i in range(4):
            rank = line[i]
            if rank >= 2:
                score += (rank - 1) * (1 << rank)
        score_table[row] = score

        s = 0
        empty = 0
        merges = 0

        prev = 0
        counter = 0
        for i in range(4):
            rank = line[i]
            s += pow(rank, SCORE_SUM_POWER)
            if rank == 0:
                empty += 1
            else:
                if prev == rank: 
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank
            
        if counter > 0:
            merges += 1 + counter

        monotonicity_left = 0
        monotonicity_right = 0
        for i in range(1, 4):
            if line[i-1] > line[i]:
                monotonicity_left += pow(line[i-1], SCORE_MONOTONICITY_POWER) - pow(line[i], SCORE_MONOTONICITY_POWER)
            else:
                monotonicity_right += pow(line[i], SCORE_MONOTONICITY_POWER) - pow(line[i-1], SCORE_MONOTONICITY_POWER)
        heur_score_table[row] = SCORE_LOST_PENALTY + \
        SCORE_EMPTY_WEIGHT * empty + \
            SCORE_MERGES_WEIGHT * merges - \
            SCORE_MONOTONICITY_WEIGHT * min(monotonicity_left, monotonicity_right) - \
            SCORE_SUM_WEIGHT * s

        for i in range(3):
            j = None
            for j in range(i + 1, 4):
                if line[j] != 0:
                    break
            if j == 4: 
                break

            if line[i] == 0:
                line[i] = line[j]
                line[j] = 0
                i -= 1
            elif line[i] == line[j]:
                if line[i] != 0xf:
                    line[i] += 1
                line[j] = 0

        result = (line[0] <<  0) | \
                       (line[1] <<  4) | \
                       (line[2] <<  8) | \
                       (line[3] << 12)
        rev_result = reverse_row(result)
        rev_row = reverse_row(row)

        row_left_table [    row] =                row  ^                result
        row_right_table[rev_row] =            rev_row  ^            rev_result
        col_up_table   [    row] = unpack_col(    row) ^ unpack_col(    result)
        col_down_table [rev_row] = unpack_col(rev_row) ^ unpack_col(rev_result)
for suffix in ['so', 'dll', 'dylib']:
    dllfn = 'bin/2048.' + suffix
    if not os.path.isfile(dllfn):
        continue
    ailib = ctypes.CDLL(dllfn)
    break
else:
    print("Couldn't find 2048 library bin/2048.{so,dll,dylib}! Make sure to build it first.")
    exit()

ailib.find_best_move.argtypes = [ctypes.c_uint64]
ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
ailib.score_toplevel_move.restype = ctypes.c_float
ailib.execute_move.argtypes = [ctypes.c_int, ctypes.c_uint64]
ailib.execute_move.restype = ctypes.c_uint64

def create_env_from_state(env, state, score):
    # Create a deep copy of the environment with the given state and score.
    new_env = copy.deepcopy(env)
    new_env.board = state.copy()
    new_env.score = score
    return new_env

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid
        tmp = deepcopy(self.board)

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {"before_add": tmp}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

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


def chance(state, score, env, depth):
    empty_cells = list(zip(*np.where(state == 0)))
    n_empty = len(empty_cells)
    if depth >= 3:
        return eval_board(state, n_empty)

    if n_empty == 0:
        _, utility, _ = maximize(state, score, env, depth + 1)
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
        _, utility, _ = maximize(new_state, score, env, depth + 1)
        u_s += utility * t[2]
    return u_s

def maximize(state, score, env, depth):
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
            u = chance(mb[1], mb[2], env, depth + 1)
        dist[mb[0]] = u + mb[2] - score
        if u + mb[2] - score >= max_utility:
            max_utility = u + mb[2] - score
            best_direction = mb[0]
    return best_direction, max_utility, 0

def from_c_board(n):
    board = []
    i = 0
    for ri in range(4):
        row = []
        for ci in range(4):
            item = (n >> (4 * i)) & 0xf
            row.append(0 if item == 0 else 1 << item)
            i += 1
        board.append(row)
    return board

def execute_move_0(board):
    ret = board
    t = transpose(board)
    ret ^= col_up_table[(t >>  0) & ROW_MASK] <<  0
    ret ^= col_up_table[(t >> 16) & ROW_MASK] <<  4
    ret ^= col_up_table[(t >> 32) & ROW_MASK] <<  8
    ret ^= col_up_table[(t >> 48) & ROW_MASK] << 12
    return ret

def execute_move_1(board):
    ret = board
    t = transpose(board)
    ret ^= col_down_table[(t >>  0) & ROW_MASK] <<  0
    ret ^= col_down_table[(t >> 16) & ROW_MASK] <<  4
    ret ^= col_down_table[(t >> 32) & ROW_MASK] <<  8
    ret ^= col_down_table[(t >> 48) & ROW_MASK] << 12
    return ret

def execute_move_2(board):
    ret = board
    ret ^= row_left_table[(board >>  0) & ROW_MASK] <<  0
    ret ^= row_left_table[(board >> 16) & ROW_MASK] << 16
    ret ^= row_left_table[(board >> 32) & ROW_MASK] << 32
    ret ^= row_left_table[(board >> 48) & ROW_MASK] << 48
    return ret

def execute_move_3(board):
    ret = board
    ret ^= row_right_table[(board >>  0) & ROW_MASK] <<  0
    ret ^= row_right_table[(board >> 16) & ROW_MASK] << 16
    ret ^= row_right_table[(board >> 32) & ROW_MASK] << 32
    ret ^= row_right_table[(board >> 48) & ROW_MASK] << 48
    return ret

def execute_move(move, board):
    if(move == 0):
        return execute_move_0(board)
    elif move == 1:
        return execute_move_1(board)
    elif move == 2:
        return execute_move_2(board)
    elif move == 3:
        return execute_move_3(board)
    return None
def transpose(x):
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)
def valid_action(board):
    out = []
    for i in range(4):
        if execute_move(i, board) != board:
            out.append(i)
    return out
with open("dp.pkl", "rb") as fp:
    dp = pickle.load(fp)
def to_c_board(m):
    board = 0
    i = 0
    for row in m:
        for c in row:
            board |= int((0 if c == 0 else np.log2(c))) << (4*i)
            i += 1
    return board
init_tables()
ailib.init_tables()
def get_action(state, score):
    board = to_c_board(state)
    legal_moves = valid_action(board)
    if score > 700000:
        best_action = random.choice(legal_moves)
    elif board in dp:
        best_action = dp[board]
    else:
        best_action = ailib.find_best_move(board)
    return best_action # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


