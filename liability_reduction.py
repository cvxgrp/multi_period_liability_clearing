from utils import *
from control import *
import numpy as np
import matplotlib.pyplot as plt

n = 200
m = 2000
seed = 10
L1 = get_L1(n, m, seed)
T = 10
np.random.seed(seed)
summer, summerT = get_summers(L1)
c1 = np.maximum(L1@np.ones(n) - L1.T@np.ones(n) + np.random.uniform(-5, 5, size=n), 0)
w1 = c1 - L1@np.ones(n) + L1.T@np.ones(n)
print("Entities with negative initial net worth:", (w1 < 0).sum())


def stage_cost(c, L, P, t):
    return cp.sum(L), [summer @ P <= .5 * c]


def final_stage_cost(c, L):
    return cp.sum(L), [L == cp.multiply((((summer @ L) / (summer @ L1.data)) @ summer), L1.data)]

T = 20
cs, Ls, Ps = liability_control(
    c1, L1, T, stage_cost, final_stage_cost)
print("Entities non-cleared", (Ls[-1] > 1e-1).sum())

cs_baseline = [np.copy(c1)]
Ls_baseline = [L1.copy()]
Ps_baseline = []
for _ in range(T - 1):
    Ps_baseline.append(pro_rata_baseline(
        cs_baseline[-1] * .5, Ls_baseline[-1]))
    Ls_baseline.append(Ls_baseline[-1] - Ps_baseline[-1])
    cs_baseline.append(cs_baseline[-1] - Ps_baseline[-1] @
                       np.ones(n) + Ps_baseline[-1].T @ np.ones(n))

latexify(4)
plt.plot(np.arange(1, T + 1), [L.sum()
                               for L in Ls], c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [L.sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{1}^TL_t\mathbf{1}$')
plt.legend()
plt.tight_layout()
plt.ylim(0)
plt.savefig("figs/reduction_total_gross_sum.pdf")
plt.close()

latexify(4)
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls],
         c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{nnz}(L_t)$')
plt.legend()
plt.ylim(0)
plt.tight_layout()
plt.savefig("figs/reduction_total_gross_nnz.pdf")
plt.close()
