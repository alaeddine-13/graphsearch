from abc import ABC, abstractmethod, abstractclassmethod
from collections import defaultdict
from utils import PriorityQueue


class Strategy(ABC):
    @staticmethod
    def reconstract_path(came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path
    
    @abstractclassmethod
    def solve(cls, start):
        pass


class AStar(Strategy):
    
    @classmethod
    def solve(cls, start, computation_progress = None, analysis_graph = None,
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
                result = cls.reconstract_path(came_from, current)
                if analysis_graph:
                    analysis_graph.update(len(result), len(closed_set))
                return result

            for neighbor in current.next_states():
                #if neighbor in closed_set:
                #    continue

                tentative_g_score = g_score[current] + distance(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                    open_set.add_task(neighbor, priority=f_score[neighbor])
            current = open_set.pop_task()
        return None


class Dijkstra(Strategy):
    @classmethod
    def solve(cls, start, computation_progress = None, 
               distance = lambda state1, state2: 1):
        
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        
        open_set = PriorityQueue()
        open_set.add_task(start, priority=distances[start])
        
        closed_set = set()
        came_from = {}

        current = open_set.pop_task()
        while current:
            closed_set.add(current)
            print(len(closed_set))
            if len(closed_set) %1000 == 0:
                start.update_computation_progress(computation_progress, len(closed_set))

            if current.is_goal():
                return cls.reconstract_path(came_from, current)

            for neighbor in current.next_states():
                if neighbor in closed_set:
                    continue
                
                new_distance = distances[current] + distance(current, neighbor)
                if new_distance < distances[neighbor] :
                    came_from[neighbor] = current
                    distances[neighbor] = new_distance
                    open_set.add_task(neighbor, priority=new_distance)
            current = open_set.pop_task()
        return None