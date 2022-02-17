import numpy as np
import matplotlib.pyplot as plt
import pickle


def load(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def generate_ctr(num_keywords):
    rng = np.random.default_rng(42)
    return rng.random(size=num_keywords, dtype=np.float32)


def online_greedy_extension1(dataset, slots_discount_ratio, ctr):
    budgets = np.array(dataset[0], dtype=np.float32)
    bids = np.array(dataset[1], dtype=np.float32)
    queries = np.array(dataset[2], dtype=np.int32)
    slots_discount_ratio = np.array(slots_discount_ratio, dtype=np.float32)
    ctr = np.array(ctr, dtype=np.float32)
    # consider ctr
    bids = ctr[np.newaxis, :] * bids

    num_advertisers = budgets.shape[0]
    num_quries = queries.shape[0]
    num_slots = slots_discount_ratio.shape[0]
    M = np.zeros(num_advertisers, dtype=np.float32)
    selected_bids = -1 * np.zeros((num_quries, num_slots), dtype=np.int32)

    for t, keyword_idx in enumerate(queries):
        for s in range(num_slots):
            discounted_bids = slots_discount_ratio[s] * bids
            affordable_advertisers = np.where(budgets - M >= discounted_bids[:, keyword_idx])[0]
            if len(affordable_advertisers) > 0:
                bids_to_consider = discounted_bids[:, keyword_idx][affordable_advertisers]
                highest_bidder_idx = affordable_advertisers[np.argmax(bids_to_consider)]
                highest_bid = np.max(bids_to_consider)
                M[highest_bidder_idx] += highest_bid
                selected_bids[t, s] = highest_bidder_idx

    revenue = np.sum(M)
    return selected_bids, revenue


def online_greedy_weighted_extension1(dataset, slots_discount_ratio, ctr):
    budgets = np.array(dataset[0], dtype=np.float32)
    bids = np.array(dataset[1], dtype=np.float32)
    queries = np.array(dataset[2], dtype=np.int32)
    slots_discount_ratio = np.array(slots_discount_ratio, dtype=np.float32)
    ctr = np.array(ctr, dtype=np.float32)
    # consider ctr
    bids = ctr[np.newaxis, :] * bids
    num_advertisers = budgets.shape[0]
    num_quries = queries.shape[0]
    num_slots = slots_discount_ratio.shape[0]

    M = np.zeros(num_advertisers, dtype=np.float32)
    selected_bids = -1 * np.zeros((num_quries, num_slots), dtype=np.int32)
    phi = np.ones(num_advertisers, dtype=np.float32)

    for t, keyword_idx in enumerate(queries):
        for s in range(num_slots):
            discounted_bids = slots_discount_ratio[s] * bids
            affordable_advertisers = np.where(budgets - M >= discounted_bids[:, keyword_idx])[0]
            if len(affordable_advertisers) > 0:
                bids_to_consider = discounted_bids[:, keyword_idx][affordable_advertisers]
                highest_bidder_idx = np.argmax(phi[affordable_advertisers] * bids_to_consider)
                highest_bid = bids_to_consider[highest_bidder_idx]
                highest_bidder_idx = affordable_advertisers[highest_bidder_idx]
                M[highest_bidder_idx] += highest_bid
                selected_bids[t, s] = highest_bidder_idx
                phi[highest_bidder_idx] = 1 - np.exp((M[highest_bidder_idx] / budgets[highest_bidder_idx]) - 1)
    revenue = np.sum(M)
    return selected_bids, revenue


if __name__ == "__main__":
    i = 5
    slots = np.linspace(0.5, 1.0, num=i)
    ds0 = load("ds0")
    ds1 = load("ds1")
    ds2 = load("ds2")
    ds3 = load("ds3")

    ctr0 = generate_ctr(len(ds0[1][0]))
    ctr1 = generate_ctr(len(ds1[1][0]))
    ctr2 = generate_ctr(len(ds2[1][0]))
    ctr3 = generate_ctr(len(ds3[1][0]))

    rev_0 = online_greedy_extension1(ds0, slots, ctr0)[1]
    rev_1 = online_greedy_extension1(ds1, slots, ctr1)[1]
    rev_2 = online_greedy_extension1(ds2, slots, ctr2)[1]
    rev_3 = online_greedy_extension1(ds3, slots, ctr3)[1]

    print(rev_0)
    print(rev_1)
    print(rev_2)
    print(rev_3)