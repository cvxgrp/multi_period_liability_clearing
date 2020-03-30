import numpy as np
import cvxpy as cp
from scipy import sparse


def get_summers(L):
    """
    Returns matrices that, when applied to sum the rows
    and columns of a sparse matrix L.
    """
    assert isinstance(
        L, sparse.csr.csr_matrix), "L must be in CSR format. Use L.tocsr()."

    rows, cols = L.nonzero()
    n = L.shape[0]
    nliabilities = rows.size
    summer = sparse.lil_matrix((n, nliabilities))
    summerT = sparse.lil_matrix((n, nliabilities))
    summer[rows, np.arange(nliabilities)] = 1.
    summerT[cols, np.arange(nliabilities)] = 1.

    return summer.tocsr(), summerT.tocsr()


def liability_control(c1, L1, T, stage_cost, final_stage_cost, **kwargs):
    if not isinstance(L1, sparse.csr.csr_matrix):
        L1 = L1.tocsc()

    summer, summerT = get_summers(L1)
    nliabilities = L1.data.size
    n = c1.size

    cs = cp.Variable((T, n), nonneg=True)
    Ls = cp.Variable((T, nliabilities), nonneg=True)
    Ps = cp.Variable((T - 1, nliabilities), nonneg=True)

    objective, constraints = final_stage_cost(cs[-1], Ls[-1])
    constraints += [Ls[0] == L1.data, cs[0] == c1]
    for t in range(T - 1):
        constraints += [Ls[t + 1] == Ls[t] - Ps[t]]
        constraints += [cs[t + 1] == cs[t] - summer @ Ps[t] + summerT @ Ps[t]]
        constraints += [summer @ Ps[t] <= cs[t]]
        gt, const = stage_cost(cs[t], Ls[t], Ps[t], t)
        objective += gt
        constraints += const

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(**kwargs)
    if result == np.inf:
        raise Exception("Liability clearing problem infeasible.")
    cs = cs.value
    Ls = [sparse.csr_matrix((Ls.value[t], L1.nonzero())) for t in range(T)]
    Ps = [sparse.csr_matrix((Ps.value[t], L1.nonzero())) for t in range(T - 1)]
    return cs, Ls, Ps


def liability_control_mpc(c1, L1, T, stage_cost, final_stage_cost, dcs, dLs, **kwargs):
    if not isinstance(L1, sparse.csr.csr_matrix):
        L1 = L1.tocsc()

    summer, summerT = get_summers(L1)
    nliabilities = L1.data.size
    n = c1.size

    cs = cp.Variable((T, n), nonneg=True)
    Ls = cp.Variable((T, nliabilities), nonneg=True)
    Ps = cp.Variable((T - 1, nliabilities), nonneg=True)

    objective, constraints = final_stage_cost(cs[-1], Ls[-1])
    constraints += [Ls[0] == L1.data, cs[0] == c1]
    for t in range(T - 1):
        constraints += [Ls[t + 1] == Ls[t] - Ps[t] + dLs[t]]
        constraints += [cs[t + 1] == cs[t] - summer @ Ps[t] + summerT @ Ps[t] + dcs[t]]
        constraints += [summer @ Ps[t] <= cs[t] + dcs[t]]
        gt, const = stage_cost(cs[t], Ls[t], Ps[t], t)
        objective += gt
        constraints += const

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(**kwargs)
    if result == np.inf:
        raise Exception("Liability clearing problem infeasible.")
    cs = cs.value
    Ls = [sparse.csr_matrix((Ls.value[t], L1.nonzero())) for t in range(T)]
    Ps = [sparse.csr_matrix((Ps.value[t], L1.nonzero())) for t in range(T - 1)]
    return cs, Ls, Ps


def pro_rata_baseline(c, L, Pi=None):
    if Pi is None:
        Pi = sparse.diags(np.array(1. / np.maximum(L.sum(axis=1), 1e-10)).flatten()) @ L
    P = L.minimum(sparse.diags(c, format='csr') @ Pi)
    return P
