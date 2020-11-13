import numpy as np
import random
from typing import List
from abc import ABC, abstractmethod
from heapq import heappop, heappush
from collections import defaultdict
import datetime




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
    def random(cls):
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
        open_set = [(heuristic(start), start)]
        closed_set = set()
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = heuristic(start)

        while len(open_set):
            _, current = heappop(open_set)
            if current in closed_set:
                print(current)
            closed_set.add(current)
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
                    # TODO improve time complexity by better finding a neighbor in open_set
                    # https://docs.python.org/3.7/library/heapq.html#priority-queue-implementation-notes
                    # also reply to this https://stackoverflow.com/questions/28488674/a-search-in-python-priority-queue
                    for index, element in enumerate(open_set):
                        if element[1] == neighbor:
                            open_set[index] = (f_score[neighbor], neighbor)
                            break
                    else:
                        heappush(open_set, (f_score[neighbor], neighbor))
            
        return None






if __name__ == "__main__":
    rand_board = Board.goal().next_states()[1].next_states()[1] #Board.random()
    print(rand_board)

    path = AStar.a_star(rand_board, Board.h2)
    for index, element in enumerate(path):
        print("step", index)
        print(element)

    
