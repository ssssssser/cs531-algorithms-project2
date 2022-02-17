import numpy as np
import pandas as pd
import pickle
import pulp
import math
import random
from gen import load
from part_4_3 import learning_based_greedy
from part_4_1and2 import online_greedy, online_weighted_greedy

def run_online_algo(ds):
    #revenuee output for each algorithms
    selected_bids_1, rev_1 = online_greedy(ds)
    selected_bids_2, rev_2 = online_weighted_greedy(ds)
    m = len(ds[2])
    selected_bids_3, rev_3 = learning_based_greedy(m, 0.05, ds)
    selected_bids_4, rev_4 = learning_based_greedy(m, 0.1, ds)
    selected_bids_5, rev_5 = learning_based_greedy(m, 0.2, ds)
    rev_ds = [rev_1, rev_2, rev_3, rev_4, rev_5]
    return rev_ds

def random_rev_stat(ds):
    revs = []
    #shuffle queries
    for i in range(100):
        random.shuffle(ds[2])
        rev_i = run_online_algo(ds)
        revs.append(rev_i)
    revs2 = np.array(revs)
    #compute mean and std
    rev_mean = []
    rev_std = []
    for j in range(5):
        rev_j = revs2[:,j]
        j_mean = np.mean(rev_j)
        j_std = np.std(rev_j)
        rev_mean.append(j_mean)
        rev_std.append(j_std)
    return rev_mean, rev_std

def main():
    #import dataset
    ds0 = load('ds0')
    ds1 = load('ds1')
    ds2 = load('ds2')
    ds3 = load('ds3')
    datasets = [ds0,ds1,ds2,ds3]

    #results for 4_5_1
    revenue = []
    for ds in datasets:
        rev_ds = run_online_algo(ds)
        revenue.append(rev_ds)

    revenue_df = pd.DataFrame(revenue,columns=['online greedy','online weighted greedy','learning(e=0.05)',
                                    'learning(e=0.1)','learning(e=0.2)'],
                                    index=['ds0','ds1','ds2','ds3'])
    revenue_df.to_excel('4_5_1_result.xlsx')



    #4_5_3
    #create 100 instances, get the means and standard deviations of each revenues
    total_rev_mean = []
    total_rev_std = []
    for ds in datasets:
        rev_mean, rev_std = random_rev_stat(ds)
        total_rev_mean.append(rev_mean)
        total_rev_std.append(rev_std)
    total_rev_mean_df = pd.DataFrame(total_rev_mean,columns=['online greedy','online weighted greedy','learning(e=0.05)',
                                    'learning(e=0.1)','learning(e=0.2)'],
                                    index=['ds0','ds1','ds2','ds3'])
    total_rev_std_df = pd.DataFrame(total_rev_std,columns=['online greedy','online weighted greedy','learning(e=0.05)',
                                    'learning(e=0.1)','learning(e=0.2)'],
                                    index=['ds0','ds1','ds2','ds3'])

    total_rev_mean_df.to_excel('revenue_mean_shuffle.xlsx')
    total_rev_std_df.to_excel('revenue_std_shuffle.xlsx')

if __name__ == "__main__":
    main()
