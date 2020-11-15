import streamlit as st
import numpy as np
import pandas as pd
import time
from tanquin import Board, AStar, Maze
from operator import itemgetter

class ComputationProgress():
    def __init__(self):
        self.latest_iteration = st.empty()
        self.bar = st.progress(0)

    def update(self, iteration, maximum):
        self.latest_iteration.text(f'Discovered {iteration} out of {maximum} states in the search space')
        self.bar.progress(int(iteration*100/maximum))
    
    def done(self):
        self.latest_iteration.markdown('## Done')
        self.bar.progress(100)
    
    def fail(self):
        self.latest_iteration.text('No Path is found')

games = {
    "Taquin": {
        "class": Board,
        "params":{
            "heuristic": {
                "Number of mis-placed pieces": Board.h1,
                "Sum of manhatten distances to each correct position": Board.h2
            },
            "generate": {
                "Generate a random state from all states (whether solvable or not)": Board.random,
                "Generate a random state from solvable states only": Board.random_from_goal
            }
        }
    },

    "Maze": {
        "class": Maze,
        "params":{
            "heuristic": {
                "Manhatten distance to the goal postion": Maze.h1,
                "Euclidean distance to the goal position": Maze.h2
            },
            "generate": {
                "Fixed 10 by 10 maze": Maze.fixture_10_by_10,
            }
        }
    }
}

strategies = {
    "A*": {
        "class": AStar,
        "method": AStar.a_star,
        "routable_params": ["heuristic"]
    }
}

st.title('Game Solving using Graph search')
st.write("Pick a game and a strategy to solve it and then click the start button")

chosen_game_name = st.sidebar.selectbox(
    'Which game do you want to solve?',
    list(games.keys()))
chosen_game = games.get(chosen_game_name, games["Taquin"])

game_params = {}
for key, value in chosen_game.get("params", {}).items():
    chosen_param_name = st.sidebar.selectbox(
        key,
        list(value.keys()))
    chosen_param = value.get(
        chosen_param_name
    )
    game_params[key] = chosen_param


chosen_strategy = strategies.get(st.sidebar.selectbox(
    'Which strategy to use?',
    list(strategies.keys())), strategies["A*"])

strategy_routable_params = {}
for routable_param in chosen_strategy.get("routable_params", []):
    strategy_routable_params[routable_param] = game_params.get(routable_param, None)


generate_button = st.button('Generate state and solve')
if generate_button:
    start = game_params["generate"]()

    st.markdown(start.render(), unsafe_allow_html=True)
    computation_progress = ComputationProgress()

    path = chosen_strategy["method"](
        start, computation_progress=computation_progress,
        **strategy_routable_params)
    if path:
        computation_progress.done()
        st.write("path found")
        for index, board in enumerate(path):
            st.write(f"step {index}")
            st.markdown(board.render(), unsafe_allow_html=True)
    else :
        computation_progress.fail()
        st.write("no path is found. The state is considered as not solvable")

