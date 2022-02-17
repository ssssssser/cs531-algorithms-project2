import numpy as np
import pickle
import pulp
import math
import copy as cp


def load(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def online_greedy(ds):
    """
    :param ds: input dataset
    :return: selected bids revenue and used money
    """
    ## get parameters from dataset
    # the number of advertisers:
    n = len(ds[0])
    # the number of keywords
    # r = len(ds[1][0])
    # the length of queries
    m = len(ds[2])

    # bugets
    B = ds[0]
    # bid price matrix
    W = ds[1]
    # query
    queries = ds[2]

    # initialize used money of each user
    M = [0]*n
    # initialize selected bids
    selected_bids = [-1]*m
    # for each time step m, choose the highest feasible bid and update M
    for t in range(m):
        # find keyword j
        j = queries[t]
        # initialize
        highest_ad_j = -1
        highest_bid_j = -1

        for i in range(n):
            # compare bid for this keyword and unspent money
            if (B[i]-M[i]) >= W[i][j]:
                if W[i][j] > highest_bid_j:
                    highest_ad_j = i
                    highest_bid_j = W[i][j]
        # update M and selected_bids
        if highest_ad_j > -1:
            M[highest_ad_j] = M[highest_ad_j]+highest_bid_j
            selected_bids[t] = highest_ad_j
    revenue = sum(M)
    return selected_bids, revenue, M


def online_weighted_greedy(ds, M, weights):
    """
    :param ds: dataset
    :param weights: weights for each advertiser
    :param M: spent money of each advertiser in phase 1
    :return: selected bids, revenue
    """
    # get parameters from dataset
    # the number of advertisers:
    n = len(ds[0])
    # the number of keywords
    r = len(ds[1][0])
    # the length of queries
    m = len(ds[2])

    # bugets
    B = ds[0]
    # bid price matrix
    W = ds[1]
    # query
    queries = ds[2]

    # M_2 is the total spent money on phase 1 and 2
    M_2 = cp.deepcopy(M)

    selected_bids = [-1]*m
    # for each time step m, choose the highest feasible bid and update M
    for t in range(m):
        # find keyword j
        j = queries[t]
        # initialize
        highest_ad_j = -1
        # highest_bid_j = 0
        phi_bid_j = -1
        for i in range(n):
            # compare bid for this keyword and unspent money
            if (B[i]-M_2[i]) >= W[i][j]:
                if (weights[i] * W[i][j]) > phi_bid_j:
                    highest_ad_j = i
                    phi_bid_j = weights[i] * W[i][j]
        # update M and selected_bids and phi
        if highest_ad_j > -1:
            M_2[highest_ad_j] = M_2[highest_ad_j]+W[highest_ad_j][j]
            selected_bids[t] = highest_ad_j

    # compute revenue
    total_spend_phase_2 = np.array(M_2) - np.array(M)
    revenue = total_spend_phase_2.sum()

    return selected_bids, revenue


def phase_1(bids, budgets, m, epsilon, query_stream):
    """
    learning phase
    :param m: estimated total number of queries
    :param epsilon: fraction of learning phase
    :param query_stream: stream of queries
    :return: alpha: learned weights; selected_bids, total_revenue returned by greedy algorithm.
    """
    ## call the greedy function to get selected bids and total revenue for epsilon*m steps
    # compute list of query used for phase_1
    phase_1_steps = math.ceil(epsilon*m)
    phase_1_query = query_stream[0: phase_1_steps]
    # select bids using greedy algorithm
    selected_bids, total_revenue, M = online_greedy([budgets, bids, phase_1_query])

    ## solve the LP and get weights alpha
    alpha = get_weights(bids, budgets, phase_1_query, epsilon)

    return selected_bids, total_revenue, alpha, M


def get_weights(bids, budgets, query, epsilon):
    """
    :param bids: matrix of bids
    :param budgets: list of budgets of each advertiser
    :param query: query used for learning
    :return: weights learned
    """
    # construct LP problem
    W_for_learn = get_W_for_learn(budgets, bids, query)
    # create problem
    prob = pulp.LpProblem("learn_weight", pulp.LpMinimize)
    # create variables alpha_i
    n_of_advertisers = len(bids)
    alpha = []
    for i in range(n_of_advertisers):
        alpha.append(pulp.LpVariable(name="alpha" + str(i), lowBound=0, cat=pulp.LpContinuous))
    # create variables beta_i
    n_of_learn_queries = len(query)
    beta = []
    for j in range(n_of_learn_queries):
        beta.append(pulp.LpVariable(name="beta" + str(j), lowBound=0, cat=pulp.LpContinuous))

    # print(alpha)
    # print(beta)
    # include the constraints
    for i in range(n_of_advertisers):
        for j in range(n_of_learn_queries):
            prob += W_for_learn[i][j] * (alpha[i] - 1) + beta[j] >= 0
    # include the objective function
    obj_var = []
    for i in range(n_of_advertisers):
        obj_var.append(epsilon * budgets[i] * alpha[i])
    for j in range(n_of_learn_queries):
        obj_var.append(beta[j])
    prob += pulp.lpSum(obj_var)
    # print(prob)
    # solve problem
    status = prob.solve()
    # record the alpha
    alpha_value = []
    for i in range(n_of_advertisers):
        alpha_value.append(pulp.value(alpha[i]))

    return alpha_value


def get_W_for_learn(B, W, query_stream):
    """
    :param B: list of budgets
    :param W: matrix of bids
    :param query_stream:
    :return:
    """
    n_advertisers = len(B)
    n_query_for_learn = len(query_stream)
    W_for_learn = []
    for i in range(n_advertisers):
        W_i_for_learn = []
        for j in range(n_query_for_learn):
            key_word = query_stream[j]
            W_i_for_learn.append(W[i][key_word])
        W_for_learn.append(W_i_for_learn)
    return W_for_learn


def phase_2(bids, budgets, m, epsilon, query_stream, weights, M):
    """
    :param m: estimated total number of queries
    :param epsilon: fraction of learning phase
    :param query_stream: stream of queries
    :param alpha: learned weights
    :return: selected bids in phase 2, total revenue in phase 2.
    """
    phase_1_steps = math.ceil(epsilon * m)
    phase_2_stream = query_stream[phase_1_steps:-1]
    phase_2_stream.append(query_stream[-1])
    print(phase_1_steps)
    print(phase_2_stream)
    selected_bids, total_revenue = online_weighted_greedy([budgets, bids, phase_2_stream], M, weights)
    return selected_bids, total_revenue


def learning_based_greedy(m, epsilon, ds):
    """
    :param m: estimated total number of queries
    :param epsilon: fraction of learning phase
    :param ds: dataset
    :return: selected bids, total revenue
    """
    budgets = ds[0]
    bids = ds[1]
    query_stream = ds[2]
    # phase 1
    selected_bids_1, total_revenue_1, alpha, M = phase_1(bids, budgets, m, epsilon, query_stream)
    # turn alpha into weights
    alpha = np.array(alpha)
    weights = 1 - alpha
    weights += 0.01 # break tie when all elements in weights = 0
    print("wieghts:", weights)
    # phase 2
    selected_bids_2, total_revenue_2 = phase_2(bids, budgets, m, epsilon, query_stream, weights, M)

    # combine the selected bids and total revenues from two bids
    print(selected_bids_1)
    print(selected_bids_2)
    total_revenue = total_revenue_1 + total_revenue_2
    selected_bids = selected_bids_1 + selected_bids_2

    return selected_bids, total_revenue


def main():
    ds0 = load("ds0")
    ds1 = load("ds1")
    ds2 = load("ds2")
    ds3 = load("ds3")
    datasets = [ds0, ds1, ds2, ds3]
    epsilons = [0.05, 0.1, 0.2]
    i = 0
    for ds in datasets:
        for epsilon in epsilons:
            m = len(ds[2]) * 1
            selected_bids, total_revenue = learning_based_greedy(m, epsilon, ds)
            print(f"dataset: {i}, epsilon: {epsilon}")
            print(selected_bids)
            print(total_revenue)
        i = i+1


if __name__ == "__main__":
    main()

