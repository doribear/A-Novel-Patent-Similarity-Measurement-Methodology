import pandas as pd
import numpy as np

from tqdm import tqdm

# datas
worker1 = pd.read_excel("", sheet_name = "")
worker2 = pd.read_excel("", sheet_name = "")
worker3 = pd.read_excel("", sheet_name = "")

# calculates values
sumr = worker1['similarity'] + worker2['similarity'] + worker3['similarity']
average = sumr / 3
distance = ((average - worker1['similarity']) ** 2) + ((average - worker2['similarity']) ** 2) + ((average - worker3['similarity']) ** 2)
threshold = 8

expert_ratings = []

for r, a in zip(distance, average):
    if r >= threshold:
        expert_ratings.append(-100)
    else:
        expert_ratings.append(a)

# eport results divide data needing expert rating & rights rating
worker1['similarity'] = expert_ratings

expert = worker1[worker1['similarity'] == -100]
worker1 = worker1[worker1['similarity'] != -100]
worker1.to_excel('')
expert.to_excel('')