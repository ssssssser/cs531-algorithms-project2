import numpy as np


def online_greedy_fair(ds, thres):
    # Input: dataset ds and threshold for selecting similar bids
    # Output: revenue and instance pair

    # Get parameters
    n = len(ds[0])  # the number of advertisers:
    r = len(ds[1][0])  # the number of keywords
    m = len(ds[2])  # the length of queries
    B = np.array(ds[0])  # bugets
    W = np.array(ds[1])  # bid price matrix
    queries = np.array(ds[2])  # query

    # Initialization
    M = np.array([0.0] * n)
    selected_bids = np.array([-1] * m)

    # Online matching as t increases
    for t in range(m):
        # Find keyword j
        j = queries[t]

        # Find max bid
        max_bid = 0
        for i in range(n):
            if (B[i] - M[i]) >= W[i, j]:
                if W[i, j] > max_bid:
                    max_bid = W[i, j]

        if max_bid > 0:
            # Find potential winners
            similar_bid = max(max_bid - thres, 0)
            ad = np.where((W[:, j] > similar_bid) & (W[:, j] <= B - M))[0]
            bid = W[ad, j]

            # Draw one winner proportional to its bid
            p = bid / sum(bid)
            winner = np.random.choice(ad, size=1, p=p)
            winner_bid = W[winner, j]

            # Update M and selected_bids
            M[winner] += winner_bid
            selected_bids[t] = winner

    revenue = sum(M)
    return revenue, selected_bids

