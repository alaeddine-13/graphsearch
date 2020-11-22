from heapq import heappop, heappush
import itertools
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

class AnalysisGraph():
    filename = "./analysis.csv"
    
    def __init__(self, name):
        self.name = name
        
        try:
            self.df = pd.read_csv(self.filename)
        except FileNotFoundError:
            self.df = pd.DataFrame({"name": [], "depth_len": [], "visited_len": []})

    def update(self, depth_len, visited_len):
        self.df = self.df.append(
            {"name": self.name, "depth_len": depth_len, "visited_len": visited_len},
            ignore_index=True
        )
        self.df.to_csv(self.filename, index=False)
    
    def done(self):
        st.write("Performance report:")
        st.write(self.name)
        self.df[self.df["name"] == self.name]\
            .plot(kind='scatter', x='depth_len',y='visited_len',color='blue')
        st.pyplot(plt)

