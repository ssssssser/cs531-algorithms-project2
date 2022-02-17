import numpy as np
import pandas as pd
import pickle
import pulp
import math
import random
from part_4_3 import learning_based_greedy
from part_4_1and2 import online_greedy, online_weighted_greedy

ds = [[5, 10], [[0.1, 1], [1, 1]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
epsilon = 0.2
m = len(ds[2])
selected_bids, total_revenue = learning_based_greedy(m, epsilon, ds)
print("learn_based_greedy")
print(selected_bids)
print(total_revenue)
selected_bids, revenue = online_greedy(ds)
print("online greedy")
print(selected_bids)
print(revenue)
selected_bids, revenue = online_weighted_greedy(ds)
print(selected_bids)
print(revenue)


