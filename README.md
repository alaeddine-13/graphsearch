
# Problems
We use the supported graph search algorithms to solve the following problems:
## Taquin
Also known as 15 Puzzle in [wikipedia](https://en.wikipedia.org/wiki/15_puzzle) is a sliding puzzle that consists of a frame of numbered square tiles in random order with one tile missing. The puzzle also exists in other sizes, particularly the smaller 8 puzzle. In our case, the size is 3x3.
For the 3x3 size, there are 362880 distinct states, and not all of them are possible to solve, no matter how many moves are made (Check in this [wiki](https://en.wikipedia.org/wiki/15_puzzle#Solvability)). Given a random state, it might be solvable or not. It's possible to determine theoretically if this state is possible to solve or not (check this [wiki](https://fr.wikipedia.org/wiki/Taquin#Configurations_solubles_et_insolubles) -sorry it's in french-). We are exposing 2 different ways to select a random state : from the entire state space and from the solvable states only. Know that when a state is not solvable, the algorithm will not process it and will flag that it's not solvable at the beginning.
