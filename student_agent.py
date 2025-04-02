# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

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
    return best_direction, max_utility, softmax(dist)

def get_action(state, score):
    env = Game2048Env()
    best_action, _, _ = maximize(state, score, env, 0)
    return best_action # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


