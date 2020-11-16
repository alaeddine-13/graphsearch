import streamlit as st
import numpy as np
import pandas as pd
import time
from operator import itemgetter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from state import Board, Maze
from strategy import AStar, Dijkstra
from utils import ComputationProgress


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Solves games using graph search algorithms",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--game', 
        help="The Game you'd like to solve (taquin and maze)",
        default='taquin'
    )
    parser.add_argument('--strategy',
        help="The algorithm used to solve the problem (a_star or dijkstra)",
        default='a_star')
    
    args = parser.parse_args()
    
    if args.game == "maze":
        start_maze = Maze.fixture_10_by_10()
        print(start_maze)
        if args.strategy == "a_star":
            path = AStar.solve(start_maze, heuristic=Maze.h1)
        
        elif args.strategy == "dijkstra":
            path = Dijkstra.solve(start_maze)
    
    elif args.game == "taquin":
        start_board = Board.random()
        print(start_board)
        if args.strategy == "a_star":
            path = AStar.solve(start_board, heuristic=Board.h2)
        
        elif args.strategy == "dijkstra":
            path = Dijkstra.solve(start_board, heuristic=Board.h2)
        
    
    if path:
        print("path found")
        for index, state in enumerate(path):
            print("step", index)
            print(state)
    else :
        print("no path is found. The state is considered as not solvable")
