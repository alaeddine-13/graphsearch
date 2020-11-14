import numpy as np
import random
from typing import List
from abc import ABC, abstractmethod
from heapq import heappop, heappush
from collections import defaultdict
import datetime
import itertools




class State(ABC):
    @abstractmethod
    def is_goal(self):
        pass

    @abstractmethod
    def next_states(self):
        pass

class Board(State):
    width = 3
    height = 3
    values = [' '] + list(range(1, width * height))
    
    def __init__(self, board=None):
        self.board = np.zeros((self.width, self.height))
        if board is not None:
            self.board = board
 
    def next_states(self):
        empty_x, empty_y = self.find_position(' ')
        boards = []
        neighbors = [(empty_x -1, empty_y), (empty_x, empty_y -1), (empty_x, empty_y +1), (empty_x +1, empty_y)]
        for neighbor in neighbors:
            if self.valid_position(neighbor):
                board = self.board.copy()
                board[neighbor[0]][neighbor[1]], board[empty_x][empty_y] = board[empty_x][empty_y], board[neighbor[0]][neighbor[1]]
                boards.append(Board(board=board))
        return boards
        
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
                res += str(value) + " "
            res += "\n"
        return res
    
    def __repr__(self):
        res = "\n"
        for row in self.board:
            for value in row:
                res += str(value) + " "
            res += "\n"
        return res

    def __hash__(self):
        return hash(str(self.board))
    
    def is_goal(self):
        return self == self.goal()

    def __lt__(self, other):
        lt_comparison = self.board < other.board
        eq_comparison = self.board < other.board
        for index, element in np.ndenumerate(lt_comparison):
            if element:
                return True
            elif not eq_comparison[index[0]][index[1]]:
                return False
        return False

    
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
    def a_star(cls, start, heuristic, distance = lambda state1, state2: 1):
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
            if current.is_goal():
                return cls.reconstract_path(came_from, current)

            for neighbor in current.next_states():
                if neighbor in closed_set:
                    if len(closed_set) % 500 ==0:
                        print(len(closed_set))
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
    rand_board = Board.random()
    print(rand_board)

    path = AStar.a_star(rand_board, Board.h2)
    for index, element in enumerate(path):
        print("step", index)
        print(element)

    
