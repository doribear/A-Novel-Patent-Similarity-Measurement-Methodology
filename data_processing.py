import pandas as pd
import numpy as np

from tqdm import tqdm

# datas
gsu = pd.read_excel("data/final_rating_workers_data.xlsx", sheet_name = "gsu")
jw = pd.read_excel("data/final_rating_workers_data.xlsx", sheet_name = "jw")
ck = pd.read_excel("data/final_rating_workers_data.xlsx", sheet_name = "ck")

# calculates values
sumr = gsu['similarity'] + jw['similarity'] + ck['similarity']
average = sumr / 3
distance = ((average - gsu['similarity']) ** 2) + ((average - jw['similarity']) ** 2) + ((average - ck['similarity']) ** 2)
threshold = 8

expert_ratings = []

for r, a in zip(distance, average):
    if r >= threshold:
        expert_ratings.append(-100)
    else:
        expert_ratings.append(a)

# eport results divide data needing expert rating & rights rating
gsu['similarity'] = expert_ratings

expert = gsu[gsu['similarity'] == -100]
gsu = gsu[gsu['similarity'] != -100]
gsu.to_excel('data/final_data.xlsx')
expert.to_excel('data/expert_data.xlsx')