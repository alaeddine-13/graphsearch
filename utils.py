from heapq import heappop, heappush
import itertools
import streamlit as st

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