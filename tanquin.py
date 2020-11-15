import numpy as np
import random
from typing import List
from abc import ABC, abstractmethod
from heapq import heappop, heappush
from collections import defaultdict
import datetime
import itertools
from sympy.combinatorics.permutations import Permutation
import math




class State(ABC):
    @abstractmethod
    def is_goal(self):
        pass

    @abstractmethod
    def next_states(self):
        pass

    @abstractmethod
    def is_solvable(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def update_computation_progress(self, computation_progress):
        pass
        


class Board(State):
    width = 3
    height = 3
    values = list(range(width * height))
    
    def __init__(self, board=None):
        self.board = np.zeros((self.width, self.height))
        if board is not None:
            self.board = board
 
    def next_states(self):
        empty_x, empty_y = self.find_position(0)
        boards = []
        neighbors = [(empty_x -1, empty_y), (empty_x, empty_y -1), (empty_x, empty_y +1), (empty_x +1, empty_y)]
        for neighbor in neighbors:
            if self.valid_position(neighbor):
                board = self.board.copy()
                board[neighbor[0]][neighbor[1]], board[empty_x][empty_y] = board[empty_x][empty_y], board[neighbor[0]][neighbor[1]]
                boards.append(self.__class__(board=board))
        return boards
    
    def is_solvable(self):
        """
        A board is solvable only if the parity of the permutation (signature) is the same
        as the parity of the empty cell.
        Check https://fr.wikipedia.org/wiki/Taquin#Configurations_solubles_et_insolubles
        """
        permutation_parity = Permutation(self.board.flatten()).signature()
        empty_position = self.find_position(0)
        distance = empty_position[0] + empty_position[1]
        empty_parity = 1 if (distance%2 ==0) else -1
        return empty_parity == permutation_parity
    
    def render(self):
        td_template = "<td>{value}</td>"
        tr_template = "<tr>{cells}</tr>"
        table_template = "<table>{rows}</table>"
        rows = []
        for row in self.board:
            tds = []
            for value in row:
                tds.append(td_template.format(value=(str(value) if value !=0 else ' ') + " "))
            rows.append(tr_template.format(cells="\n".join(tds)))
        return table_template.format(rows="\n".join(rows))
    
    def update_computation_progress(self, computation_progress, processed):
        if computation_progress:
            computation_progress.update(processed, math.factorial(self.width * self.height))
        
    @classmethod
    def valid_position(cls, position):
        return cls.width>position[0]>=0 and cls.height>position[1]>=0


    def find_position(self, value):
        x, y = np.where(self.board == value)
        return (x[0], y[0])
    
    @classmethod
    def random(cls, from_goal=None):
        if from_goal:
            res = cls.goal()
            for _ in range(from_goal):
                next_states = res.next_states()
                res = random.choice(next_states)
            return res
        else:
            choices = random.sample(set(cls.values), len(cls.values))
            choices = np.array(choices)
            choices = choices.reshape((cls.width, cls.height))
            res = cls()
            res.board = choices
            return res
    
    @classmethod
    def random_from_goal(cls):
        return cls.random(from_goal=500)

    @classmethod
    def goal(cls):
        goal = np.array(cls.values)
        goal = goal.reshape(cls.width, cls.height)
        res = Board()
        res.board = goal
        return res

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)
    
    def __str__(self):
        res = ""
        for row in self.board:
            for value in row:
                res += (str(value) if value !=0 else ' ') + " "
            res += "\n"
        return res
    
    def __repr__(self):
        return "\n" + self.__str__()

    def __hash__(self):
        return hash(str(self.board))
    
    def is_goal(self):
        return self == self.goal()

    
    @classmethod
    def h1(cls,state):
        total = 0
        for element in np.nditer(state.board):
            if cls.goal().find_position(element) != state.find_position(element):
                total += 1
        return total      
 
    @classmethod
    def h2(cls, state):
        total = 0
        for index, element in np.ndenumerate(state.board):
            correct_position = cls.goal().find_position(element)
            total += abs(index[0] - correct_position[0]) + \
                abs(index[1] - correct_position[1])
        return total


class Maze(State):
    width = 10
    height = 10
    wall = "X"
    empty = " "
    player = "O"
    start = "S"
    goal = "F"
    start_position = (0, 0)
    goal_position = (width-1, height-1)
    
    def __init__(self, maze=None):
        self.maze = np.full((self.width, self.height), self.empty)
        if maze is not None:
            self.maze = maze
        self.start_position = (0, 0)
        self.goal_position = (self.width-1, self.height-1)
 
    def next_states(self):
        player_x, player_y = self.find_position(self.player)
        mazes = []
        neighbors = [(player_x -1, player_y), (player_x, player_y -1), (player_x, player_y +1), (player_x +1, player_y)]
        for neighbor in neighbors:
            if self.valid_position(neighbor):
                maze = self.maze.copy()
                maze[neighbor[0]][neighbor[1]], maze[player_x][player_y] = self.player, self.empty
                mazes.append(self.__class__(maze=maze))
        return mazes

    def render(self):
        wall_color = "#919294"
        start_color = "#9ca9ff"
        end_color = "#a1ff85"
        empty_color = "#ffffff"
        td_template = '<td style="background-color:{color}">{value}</td>'
        tr_template = "<tr>{cells}</tr>"
        table_template = "<table>{rows}</table>"
        rows = []
        for x, row in enumerate(self.maze):
            tds = []
            for y, value in enumerate(row):
                to_show = value
                color = None
                if (x, y) == self.start_position:
                    color = start_color
                elif (x, y) == self.goal_position:
                    color = end_color
                elif value == self.wall:
                    color = wall_color
                else:
                    color = empty_color
                
                td = td_template.format(value=to_show, color = color)
                tds.append(td)
            rows.append(tr_template.format(cells="\n".join(tds)))
        return table_template.format(rows="\n".join(rows))
    
    def update_computation_progress(self, computation_progress, processed):
        if computation_progress:
            computation_progress.update(processed, 1 + len(np.where(self.maze == self.empty)))

    def is_solvable(self):
        return True

    def valid_position(self, position):
        return self.width>position[0]>=0 and self.height>position[1]>=0 and \
            self.maze[position[0]][position[1]] == self.empty
            

    def find_position(self, value):
        x, y = np.where(self.maze == value)
        return (x[0], y[0])
    
    @classmethod
    def random(cls):
        pass
    
    @classmethod
    def fixture_10_by_10(cls):
        res = cls()
        walls = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9),
            (1, 5), (1, 7), (1, 8), (1, 9),
            (2, 3), (2, 5),
            (3, 1), (3, 3), (3, 5), (3, 7), (3, 8),
            (4, 1), (4, 3), (4, 5), (4, 7),
            (5, 1), (5, 2), (5, 3), (5, 5), (5, 7), (5, 9),
            (6, 1), (6, 5), (6, 7), (6, 9),
            (7, 1), (7, 3), (7, 4), (7, 5), (7, 7), (7, 9),
            (8, 0), (8, 7), (8, 9),
            (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7),
        ]
        for index, element in np.ndenumerate(res.maze):
            if index in walls:
                res.maze[index[0]][index[1]] = cls.wall
        res.maze[cls.start_position[0]][cls.start_position[1]] = cls.player
        return res


    def __eq__(self, other):
        return np.array_equal(self.maze, other.maze)
    
    def __str__(self):
        res = ""
        for x, row in enumerate(self.maze):
            for y, value in enumerate(row):
                to_show = value
                if (x, y)==self.goal_position and value==self.empty:
                    to_show = self.goal
                elif (x, y)==self.start_position and value==self.empty:
                    to_show = self.start
                
                res += value + " "
            res += "\n"
        return res
    
    def __repr__(self):
        return "\n" + self.__str__()

    def __hash__(self):
        return hash(str(self.maze))
    
    def is_goal(self):
        return self.find_position(self.player) == self.goal_position
     
    @classmethod
    def h1(cls, state):
        player_postion = state.find_position(state.player)
        return abs(player_postion[0] - state.goal_position[0]) + \
            abs(player_postion[1] - state.goal_position[1])
     
    @classmethod
    def h2(cls, state):
        player_postion = state.find_position(state.player)
        x = player_postion[0] - state.goal_position[0]
        y = player_postion[1] - state.goal_position[1]
        return math.sqrt(x*x + y*y)



class PriorityQueue():
    REMOVED = '<removed-task>'
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()


    def add_task(self, task, priority=0):
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        return None

class AStar():
    @staticmethod
    def reconstract_path(came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path
    
    @classmethod
    def a_star(cls, start, computation_progress = None, 
               heuristic = None, distance = lambda state1, state2: 1):
        if not start.is_solvable():
            return None
        open_set = PriorityQueue()
        open_set.add_task(start, priority=heuristic(start))
        closed_set = set()
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = heuristic(start)

        current = open_set.pop_task()
        while current:
            closed_set.add(current)
            if len(closed_set) %1000 == 0:
                start.update_computation_progress(computation_progress, len(closed_set))

            if current.is_goal():
                return cls.reconstract_path(came_from, current)

            for neighbor in current.next_states():
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + distance(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    open_set.add_task(neighbor, priority=f_score[neighbor])
            current = open_set.pop_task()
        return None

if __name__ == "__main__":
    maze_or_board = "maze"
    if maze_or_board == "maze":
        start_maze = Maze.fixture_10_by_10()
        print(start_maze)

        path = AStar.a_star(start_maze, heuristic=Maze.h1)
    
    elif maze_or_board == "board":
        start_board = Board.random()
        print(start_board)
        path = AStar.a_star(start_board, heuristic=Board.h2)
    
    if path:
        print("path found")
        for index, state in enumerate(path):
            print("step", index)
            print(state)
    else :
        print("no path is found. The state is considered as not solvable")

        

    
