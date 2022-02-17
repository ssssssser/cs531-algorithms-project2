import numpy as np


def offline_greedy(ds):
    # Input: dataset ds
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
    select = np.array([-1] * m)
    t_sold = []

    # Select from the largest bid
    w = np.sort(W, axis=None)[::-1]
    for bid in w:
        if bid > 0:
            # Multiplicity
            sel = np.array(np.where(W == bid))

            for i in range(sel.shape[1]):
                # Select by order of advertiser and keyword
                ad = sel[0, i]
                qu = sel[1, i]

                # Queries with that keyword, select by order of query
                t_fea = np.where(queries == qu)
                t_rem = np.setdiff1d(t_fea, t_sold)

                for t in t_rem:
                    if B[ad] - M[ad] >= bid:
                        M[ad] += bid
                        select[t] = ad
                        t_sold.append(t)

    revenue = sum(M)
    return revenue, select

