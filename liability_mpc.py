from utils import *
from control import *
import numpy as np
import matplotlib.pyplot as plt

n = 200
m = 2000
seed = 10
L1 = get_L1(n, m, seed)
T = 20
c1 = np.maximum(L1@np.ones(n) - L1.T@np.ones(n) + np.random.uniform(-5, 0, size=n), 0)


def stage_cost(c, L, P, t):
    return cp.sum(L), []


def final_stage_cost(c, L):
    return cp.sum(L), []

np.random.seed(0)
cs = [np.copy(c1)]
Ls = [L1.copy()]
Ps = []

cs_baseline = [np.copy(c1)]
Ls_baseline = [L1.copy().todense()]
Ps_baseline = []

L_running_sum = L1.copy().todense()
for t in range(18):
    Ls[-1].eliminate_zeros()
    dc = np.exp(np.random.randn(n))
    dL = np.exp(np.random.randn(Ls[-1].data.size)) / 10

    cs_temp, Ls_temp, Ps_temp = liability_control_mpc(cs[-1], Ls[-1], T - t,
                                                      lambda c, L, P, t: (
                                                          cp.sum(L), []),
                                                      lambda c, L: (
                                                          cp.sum(L), []),
                                                      dLs=[
                                                          dL] + [np.exp(.5) / 10] * 19,
                                                      dcs=[dc] + [np.exp(.5)] * 19)

    dL_dense = sparse.csr_matrix((dL,
                                  Ls[-1].nonzero()), shape=(n, n)).todense()
    L_running_sum += dL_dense
    Pi = sparse.csr_matrix(np.array(np.diag(1. / np.array(L_running_sum @ np.ones(n)).flatten()) @ L_running_sum))
    Ps_temp_baseline = pro_rata_baseline(cs_baseline[-1] + dc, sparse.csr_matrix(Ls_baseline[-1] + dL_dense),
                                         Pi)

    cs.append(cs_temp[1])
    Ls.append(Ls_temp[1])
    Ps.append(Ps_temp[0])

    cs_baseline.append(cs_baseline[-1] - Ps_temp_baseline @ np.ones(n) + Ps_temp_baseline.T @ np.ones(n) + dc)
    Ls_baseline.append(Ls_baseline[-1] - Ps_temp_baseline + dL_dense)
    Ps_baseline.append(Ps_temp_baseline)

latexify(4)
plt.plot(np.arange(1, 20), [L.sum() for L in Ls], c='black', label='MPC')
plt.plot(np.arange(1, 20), [L.sum()
                            for L in Ls_baseline], '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, 20, 2))
plt.ylabel(r'$\mathbf{1}^TL_t \mathbf{1}$')
plt.legend()
plt.tight_layout()
plt.ylim(0)
plt.savefig("figs/mpc.pdf")
plt.close()
