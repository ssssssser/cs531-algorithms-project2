import os
import numpy as np
import random
import pandas as pd
import copy
import pickle


def online_greedy(ds):
    '''
    :param ds: input dataset
    :return:
    '''
    #get parameters from dataset
    #the number of advertisers:
    n = len(ds[0])
    #the number of keywords
    r = len(ds[1][0])
    #the length of queries
    m = len(ds[2])

    #bugets
    B = ds[0]
    #bid price matrix
    W = ds[1]
    #query
    queries = ds[2]

    #initialize M
    M = [0]*n

    selected_bids = [-1]*m #不能设0，0代表第一个advertiser
    #for each time step m, choose the highest feasible bid and update M
    for t in range(m):
        #find keyword j
        j = queries[t]
        #initialize
        highest_ad_j = -1
        highest_bid_j = 0

        for i in range(n):
            #compare bid for this keyword and unspent money
            if (B[i]-M[i]) >= W[i][j]:
                if W[i][j] > highest_bid_j: #tie: pick the former advertiser (by order)
                    highest_ad_j = i
                    highest_bid_j = W[i][j]
        #update M and selected_bids
        if highest_ad_j > -1:
            M[highest_ad_j] = M[highest_ad_j]+highest_bid_j
            selected_bids[t] = highest_ad_j
    revenue = sum(M)
    return selected_bids, revenue

def online_weighted_greedy(ds):
    '''
    :param ds: input dataset
    :return:
    '''
    #get parameters from dataset
    #the number of advertisers:
    n = len(ds[0])
    #the number of keywords
    r = len(ds[1][0])
    #the length of queries
    m = len(ds[2])

    #bugets
    B = ds[0]
    #bid price matrix
    W = ds[1]
    #query
    queries = ds[2]

    #initialize M and phi
    M = [0]*n
    phi = [1]*n

    selected_bids = [-1]*m
    #for each time step m, choose the highest feasible phi*bid and update M
    for t in range(m):
        #find keyword j
        j = queries[t]
        #initialize
        highest_ad_j = -1
        #highest_bid_j = 0
        phi_bid_j = 0
        for i in range(n):
            #compare bid for this keyword and unspent money
            if (B[i]-M[i]) >= W[i][j]:
                if (phi[i]*W[i][j]) > phi_bid_j:
                    highest_ad_j = i
                    phi_bid_j = phi[i]*W[i][j]
        #update M and selected_bids and phi
        if highest_ad_j > -1:
            M[highest_ad_j] = M[highest_ad_j]+W[highest_ad_j][j]
            selected_bids[t] = highest_ad_j
            phi[highest_ad_j] = 1 - np.exp(M[highest_ad_j] / B[highest_ad_j] - 1)
    revenue = sum(M)
    return selected_bids, revenue




