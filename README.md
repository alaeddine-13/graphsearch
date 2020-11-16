# Overview
The project consists in applying graph search algorithms like A* and Dijkstra to solve problems in an efficient way. All problems are represented as classes that must implement the `State` abstract class. Graph search algorithms also must implement the `Strategy` abstract class.
The project can be executed from the console or from the web interface, using a Streamlit app. The project is deployed here: https://ai.alaeddineabdessalem.com
# Problems
We use the supported graph search algorithms to solve the following problems:
## Taquin
Also known as 15 Puzzle in [wikipedia](https://en.wikipedia.org/wiki/15_puzzle) is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile missing. The puzzle also exists in other sizes, particularly the smaller 8 puzzle. In our case, the size is 3x3.
For the 3x3 size, there are 362880 distinct states, and not all of them are possible to solve, no matter how many moves are made (Check in this [wiki](https://en.wikipedia.org/wiki/15_puzzle#Solvability)). Given a random state, it might be solvable or not. It's possible to determine theoretically if this state is possible to solve or not (check this [wiki](https://fr.wikipedia.org/wiki/Taquin#Configurations_solubles_et_insolubles) -sorry it's in french-). We are exposing 2 different ways to select a random state : from the entire state space and from the solvable states only. Know that when a state is not solvable, the algorithm will not process it and will flag that it's not solvable at the beginning. To determine whether a state is solvable or not, we compute the parity of the permutation and the parity of the empty cell. If both parities are equal, then the state is considered as solvable (check this [wiki](https://fr.wikipedia.org/wiki/Taquin#Configurations_solubles_et_insolubles)).

## Maze
The problem consists in finding the shortest path from the start position to the end position. The maze is represented as a grid consisting in empty cells and wall cells. We provide 2 heuristics for A* algorithm which are the manhatten and euclidean distances between the player and the final position.
