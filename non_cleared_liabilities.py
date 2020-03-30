from utils import *
from control import *
import numpy as np
import networkx as nx
import cvxpy as cp


np.random.seed(0)
G = nx.gnm_random_graph(40, 400, directed=True, seed=0)
n = len(G.nodes())
L1 = nx.adjacency_matrix(G).tocsr().astype(float)
L1.data *= np.exp(np.random.randn(L1.data.size))
c1 = np.maximum(L1@np.ones(n) - L1.T@np.ones(n) + np.random.uniform(-5, 5, size=n), 0)


def stage_cost(c, L, P, t):
    return cp.sum(L), []


def final_stage_cost(c, L):
    return cp.sum(L), []

T = 10
cs, Ls, Ps = liability_control(
    c1, L1, T, stage_cost, final_stage_cost, verbose=True)
base_noncleared = (Ls[-1] > 1e-1).sum()
base_sum = Ls[-1].sum()

summer, summerT = get_summers(L1)
nliabilities = L1.data.size
n = c1.size

cs = cp.Variable((T, n), nonneg=True)
Ls = cp.Variable((T, nliabilities), nonneg=True)
Ps = cp.Variable((T - 1, nliabilities), nonneg=True)
z = cp.Variable(nliabilities, boolean=True)

objective = cp.sum(z) + 1e-6 * cp.sum(Ls[-1])
constraints = [Ls[0] == L1.data, cs[0] == c1, Ls[-1]
               <= cp.multiply(z, L1.data)]
for t in range(T - 1):
    constraints += [Ls[t + 1] == Ls[t] - Ps[t]]
    constraints += [cs[t + 1] == cs[t] - summer @ Ps[t] + summerT @ Ps[t]]
    constraints += [summer @ Ps[t] <= cs[t]]

prob = cp.Problem(cp.Minimize(objective), constraints)
result = prob.solve(verbose=True)
ncvx_noncleared = (Ls[-1].value > 1e-1).sum()
ncvx_sum = Ls[-1].value.sum()

print("base: %d noncleared, %.3f sum" % (base_noncleared, base_sum))
print("ncvx: %d noncleared, %.3f sum" % (ncvx_noncleared, ncvx_sum))
