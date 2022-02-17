import numpy as np
import pulp as pl
import pickle


def load_pkl(ds):
    f = open(ds, 'rb')
    data = pickle.load(f)
    return data


def offline_lp(ds):
    # load file(outside) and read data
    data = ds
    budgets = data[0]
    bids = data[1]
    queries = data[2]

    # get length of important variables
    n = len(budgets)
    r = len(bids)
    m = len(queries)

    # create (n+m)*(n*m) constraints matrix A

    # Firstly, for \sum_{i=1}^{n}x_ij <= 1, j \in [m]
    A = np.zeros((n + m, n * m))
    count = 0
    for i in range(m):
        tmp = count
        while (count < m * n):
            A[i, count] = 1
            count += m
        count = tmp + 1

        # Then for \sum_{j=1}^{m}w_ij*x_ij <= 1, i \in [n]
    for i in range(n):
        for j in range(m):
            query = queries[j]
            A[m + i, i * m + j] = bids[i][query]

    # Next create (n+m)*1 constant vector B
    B = []
    i = 0
    while (i < m + n):
        while (i < m):
            B.append(1)
            i += 1
        B.append(budgets[i - m])
        i += 1
    # And create (n+m)*2 bounds b
    b = []
    for i in range(n * m):
        b.append([0, 1.0])

    # bids_queries=np.take(bids,queries,1).ravel()

    # Get a contiguous flattened 1-D array of combined bids and queries
    bids_queries = np.take(bids, queries, 1).ravel()
    bq_length = len(bids_queries)

    # Get a list of variable:x1,x2,x3,...
    variables = []
    xs = ['x'] * bq_length
    ns = list(range(1, bq_length + 1))
    for i in range(bq_length):
        variables.append(pl.LpVariable(name=(xs[i] + str(ns[i])), lowBound=b[i][0], upBound=b[i][1]))

    model = pl.LpProblem("OptimizeModel", pl.LpMaximize)

    # constraints=[]
    # for i in range(len(A)):
    for i in (range(len(A))):
        e = pl.LpAffineExpression([(variables[j], A[i][j]) for j in range(bq_length) if A[i][j] != 0])
        bools = (e <= B[i])
        model += bools
        # Add constraints to the LP model
        name = str(i)
        LPconstraints = pl.LpConstraint(e, pl.LpConstraintLE, name, B[i])
        model.addConstraint(LPconstraints)

    # solve the LP model
    objects = pl.lpDot(bids_queries, variables)
    model += objects
    model.solve(pl.getSolver('PULP_CBC_CMD'))

    values = [pl.value(variable) for variable in variables]
    obj_value = pl.value(objects)

    # print(obj_value, values)
    print(obj_value)

    return obj_value


def offline_lp_round(ds):
    # load file(outside) and read data
    data = ds
    budgets = data[0]
    bids = data[1]
    queries = data[2]

    # get length of important variables
    n = len(budgets)
    r = len(bids)
    m = len(queries)

    # create (n+m)*(n*m) constraints matrix A

    # Firstly, for \sum_{i=1}^{n}x_ij <= 1, j \in [m]
    A = np.zeros((n + m, n * m))

    count = 0
    for i in range(m):
        tmp = count
        while (count < m * n):
            A[i, count] = 1
            count += m
        count = tmp + 1

        # Then for \sum_{j=1}^{m}w_ij*x_ij <= 1, i \in [n]
    for i in range(n):
        for j in range(m):
            query = queries[j]
            A[m + i, i * m + j] = bids[i][query]

    # Next create (n+m)*1 constant vector B
    B = []
    i = 0
    while (i < m + n):
        while (i < m):
            B.append(1)
            i += 1
        B.append(budgets[i - m])
        i += 1

    # And create (n+m)*2 bounds b
    b = []
    for i in range(n * m):
        b.append([0, 1.0])

    # bids_queries=np.take(bids,queries,1).ravel()

    # Get a contiguous flattened 1-D array of combined bids and queries
    bids_queries = np.take(bids, queries, 1).ravel()
    bq_length = len(bids_queries)

    # Get a list of variable:x1,x2,x3,...
    variables = []
    xs = ['x'] * bq_length
    ns = list(range(1, bq_length + 1))
    for i in range(bq_length):
        variables.append(pl.LpVariable(name=(xs[i] + str(ns[i])), lowBound=b[i][0], upBound=b[i][1]))

    model = pl.LpProblem("OptimizeModel", pl.LpMaximize)
    # constraints=[]
    # for i in range(len(A)):
    for i in (range(len(A))):
        e = pl.LpAffineExpression([(variables[j], A[i][j]) for j in range(bq_length) if A[i][j] != 0])
        bools = (e <= B[i])
        model += bools

        # Add constraints to the LP model
        name = str(i)
        LPconstraints = pl.LpConstraint(e, pl.LpConstraintLE, name, B[i])
        model.addConstraint(LPconstraints)

    # solve the LP model
    objects = pl.lpDot(bids_queries, variables)
    model += objects
    model.solve(pl.getSolver('PULP_CBC_CMD'))

    values = [pl.value(variable) for variable in variables]
    obj_value = pl.value(objects)

    # print(obj_value, values)
    print(obj_value)

    # print optimal revenues and chosen values
    optimal_revenues = pl.value(objects)
    chosen_values = []
    for i in range(len(variables)):
        chosen_values.append(pl.value(variables[i]))

    re = np.array(chosen_values).reshape(n, m)
    select = []
    status = np.zeros(m)

    M = 0
    # print(sum(budgets))

    for i in range(re.shape[1]):
        lst = [re[j][i] for j in range(re.shape[0])]
        max_value_pos = np.argmax(lst)
        while (status[i] == 0):
            if max(lst) == 0:
                select.append(-1)
                status[i] = 1

            if budgets[max_value_pos] >= bids[max_value_pos][queries[i]]:
                budgets[max_value_pos] -= bids[max_value_pos][queries[i]]
                select.append(max_value_pos)
                M += bids[max_value_pos][queries[i]]
                status[i] = 1

            else:
                lst[max_value_pos] = 0
                max_value_pos = np.argmax(lst)

    print(M)
    # print(budgets)

    return M, select


ds0 = load_pkl('ds0.pkl')
M0, select0 = offline_lp_round(ds0)

ds1 = load_pkl('ds1.pkl')
M1, select1 = offline_lp_round(ds1)

ds2 = load_pkl('ds2.pkl')
M2, select2 = offline_lp_round(ds2)

ds3 = load_pkl('ds3.pkl')
M3, select3 = offline_lp_round(ds3)