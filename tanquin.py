import numpy as np
import random
from typing import List

class Board():
    values = [' ',1,2,3,4,5,6,7,8]
    width = 3
    height = 3
    
    def __init__(self, board=None):
        self.board = np.zeros((self.width, self.height))
        if board is not None:
            self.board = board
 
    def next_boards(self):
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
        

    def is_goal(self):
        return self == self.goal()
    
    @classmethod
    def h1(cls):
        pass

    @classmethod
    def h2(cls):
        pass

if __name__ == "__main__":
    rand_board = Board.random()
    print(rand_board)

    print("next boards")
    for board in rand_board.next_boards():
        print(board)

    
