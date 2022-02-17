import random
import math
import numpy as np
import pickle


def noisy_nz_1d(row, dec=2):
    """Adds small Gaussian noise, truncated to dec decimals,
    to each element of row.
    """
    noisy = []
    for x in row:
        if x == 0:
            noisy.append(x)
        else:
            noisy.append(round(x + random.gauss(0, 0.1), dec))
    return noisy


def noisy_nz_2d(rows, dec=2):
    """Adds small Gaussian noise, truncated to dec decimals,
    to each element in each row of rows.
    """
    return [noisy_nz_1d(row, dec) for row in rows]


def ds0_gen(m):
    """Generates the 2 budgets, 2 bid vectors, and 2m queries
    for dataset ds0.

    Keyword arguments:
    n -- number of advertisers, an integer
    B -- budget for all advertisers, an integer
    """

    # one advertiser pays 1 for both keywords
    #   and one pays 0.5 for only one
    bids = [[1, 1], [0.5, 0]]

    # suppose the queries alternate keywords (unrealistic)
    stream = [0, 1] * m

    # first advertiser can afford all queries with first keyword,
    #   and second advertiser can afford all queries with second keyword.
    budgets = [m, m / 2]

    return (budgets, bids, stream)


def ds1_gen(n, B):
    """Generates the n budgets, n bids vectors, and Bn queries
    for dataset ds1.

    Keyword arguments:
    n -- number of advertisers, an integer
    B -- budget for all advertisers, an integer
    """
    budgets = [B] * n

    # i-th adveriser bids 1 on first B*(i+1) keywords, 0 on rest
    bids = [[1] * B * (i + 1) + [0] * B * (n - i - 1) for i in range(n)]

    # shuffle the advertisers due to deterministic construction
    random.shuffle(bids)

    # add small gaussian noise to bids
    bids = noisy_nz_2d(bids)

    # the queries are just each keyword once
    m = n * B
    stream = list(range(m))

    return (budgets, bids, stream)


def ds2_gen(n, B):
    """Generates the n budgets, n bid vectors, and Bn queries
    for dataset ds2.

    Keyword arguments:
    n -- number of advertisers, an integer
    B -- budget for all advertisers, an integer
    """

    # all budgets are B
    budgets = [B] * n

    mid = n // 2

    prefix = [1] * B * mid
    ones = [1] * B
    zeros = [0] * B

    bids = []
    for i in range(n):
        if i < mid:
            # if i < n/2, then i-th adveriser
            #   bids 1 on keywords {Bi, ..., B(i+1)-1}
            #   and bids on no other keywords
            bid = zeros * i + ones + zeros * (n - i - 1)

        else:
            # if i >= n/2, then i-th adveriser
            #   bids 1 on first Bn/2 keywords and {Bi, ..., B(i+1)-1}
            #   and bids on no other keywords
            i = i - mid
            bid = ones * mid + zeros * i + ones + zeros * (n - mid - i - 1)

        bids.append(bid)

    # the queries are just each keyword once
    m = n * B
    stream = list(range(m))

    # shuffle the advertisers due to deterministic construction
    random.shuffle(bids)

    # randomly perturb the 1's
    bids = noisy_nz_2d(bids)

    return (budgets, bids, stream)


def ds3_gen(n, expo=1, f=1):
    """Generates the n bid vectors and f*n^2 queries
    for dataset ds3.

    Keyword arguments:
    n -- number of advertisers
    expo -- exponent to be used when generating the degrees
            of the queries (default 1.0)
    f -- a scale factor for the size of the stream w.r.t
            the n^2 keywords (default 1)
    """

    m = round(n ** 2)

    # initialize all bids to 0
    bids = [[0] * m for _ in range(n)]

    advertisers = list(range(n))

    # initialize degrees (# of bids) given by each advertiser to 0
    ad_degrees = [0] * n
    total_degree = 0

    for j in range(m):

        # get random degree d, 0 < d <= n
        while True:
            g = random.gauss(1, 1)
            d = min(n, math.floor(math.exp(g)))
            break
            if d != 0:
                break

        # get weight distribution
        if total_degree == 0:
            # all degrees are 0, so weight all advertisers equally
            P = [1] * n
        else:
            # weight advertisers roughly proportional to their degree
            P = [(degree + 1) ** expo for degree in ad_degrees]

        # scale P so that it is a probability distribution
        sum_P = sum(P)
        P = [p / sum_P for p in P]

        # using P, sample d neighbors
        neighbors = np.random.choice(advertisers, d, replace=False, p=P)

        # get random value for j-th keyword to center bids around
        value = random.random()

        # generate the neighbors' bids
        for i in neighbors:
            bid = round(random.gauss(value, 0.1), 2)
            bids[i][j] = max(bid, 0)
            ad_degrees[i] += 1
            total_degree += 1

    stream = list(range(m))

    bids = noisy_nz_2d(bids)

    # if scale factor f > 1, scale the query stream by f and
    #   randomly sample with replacement.
    if f > 1:
        stream = random.choices(stream, k=round(f * m))

    return (bids, stream)


def dump(name, obj):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


# ds0 = ds0_gen(100)
# ds1 = ds1_gen(20,20)
# ds2 = ds2_gen(20,20)
# ds3 = ds3_gen(20,1.3,10)

# dump("ds0", ds0)
# dump("ds1", ds1)
# dump("ds2", ds2)
# dump("ds3", ds3)

'''
Get budget vector for ds3:
1) pretend each advertiser has unlimited budget
2) set their budget equal to the amount spent during
    the online greedy algorithm with unlimited budget
'''

# pretend_budgets = [float('inf')]*20
# ds3_budgets = greedy(pretend_budgets,ds3[0],ds3[1])