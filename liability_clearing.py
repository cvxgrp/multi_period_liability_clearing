from utils import *
from control import *
import numpy as np
import matplotlib.pyplot as plt

n = 200
m = 2000
seed = 10
L1 = get_L1(n, m, seed)
T = 10
c1 = np.maximum(L1@np.ones(n) - L1.T@np.ones(n), 0)
summer, summerT = get_summers(L1)

# Total gross liability
cs_tot, Ls_tot, Ps_tot = liability_control(c1, L1, T,
                                           stage_cost=lambda c, L, P, t: (
                                               cp.sum(L), []),
                                           final_stage_cost=lambda c, L: (
                                               0, [L == 0]))

# Risk-weighted liability
w1 = c1 - L1 @ np.ones(n) + L1.T @ np.ones(n)
r = np.exp(-w1)
cs_risk, Ls_risk, Ps_risk = liability_control(c1, L1, T,
                                              stage_cost=lambda c, L, P, t: (
                                                  (summerT @ L) @ r, []),
                                              final_stage_cost=lambda c, L: (
                                                  (summerT @ L) @ r, [L == 0]))

# Total squared payment
lams = np.append(0, np.logspace(0, 2, 3))
Ls_sum = []
Ps_sum = []
for lam in lams:
    def stage_cost(c, L, P, t):
        return cp.sum(L) + lam * cp.sum_squares(P), []

    def final_stage_cost(c, L):
        return cp.sum(L), [L == 0]
    cs_temp, Ls_temp, Ps_temp = liability_control(
        c1, L1, T, stage_cost, final_stage_cost)

    Ls_sum.append([L.sum() for L in Ls_temp])
    Ps_sum.append([P.sum() for P in Ps_temp])

# Baseline
cs_baseline = [np.copy(c1)]
Ls_baseline = [L1.copy()]
Ps_baseline = []
for _ in range(T - 1):
    Ps_baseline.append(pro_rata_baseline(cs_baseline[-1], Ls_baseline[-1]))
    Ls_baseline.append(Ls_baseline[-1] - Ps_baseline[-1])
    cs_baseline.append(cs_baseline[-1] - Ps_baseline[-1] @
                       np.ones(n) + Ps_baseline[-1].T @ np.ones(n))

# Plots
latexify(4)
plt.plot(np.arange(1, T + 1), [L.sum()
                               for L in Ls_tot], c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [L.sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{1}^TL_t\mathbf{1}$')
plt.legend()
plt.tight_layout()
plt.savefig("../figs/total_gross_sum.pdf")
plt.close()

latexify(4)
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls_tot],
         c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{nnz}(L_t)$')
plt.legend()
plt.tight_layout()
plt.savefig("../figs/total_gross_nnz.pdf")
plt.close()

latexify(4)
plt.plot(np.arange(1, T + 1), [(L @ r).sum()
                               for L in Ls_risk], c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [(L @ r).sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{1}^TL_t r$')
plt.legend()
plt.tight_layout()
plt.savefig("../figs/weighted_total_gross_sum.pdf")
plt.close()

latexify(4)
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls_risk],
         c='black', label='optimal')
plt.plot(np.arange(1, T + 1), [(L > 1e-1).sum() for L in Ls_baseline],
         '--', c='black', label='baseline')
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{nnz}(L_t)$')
plt.legend()
plt.tight_layout()
plt.savefig("../figs/weighted_total_gross_nnz.pdf")
plt.close()

latexify(4)
for i, lam in enumerate(lams):
    plt.plot(np.arange(1, T + 1), Ls_sum[i], label="$\lambda=%d$" % lam)
plt.legend()
plt.xlabel('$t$')
plt.xticks(np.arange(1, T + 1, 2))
plt.ylabel(r'$\mathbf{1}^TL_t \mathbf{1}$')
plt.tight_layout()
plt.savefig("../figs/payment_squared_liability.pdf")
plt.close()

latexify(4)
for i, lam in enumerate(lams):
    plt.plot(np.arange(1, T), Ps_sum[i], label="$\lambda=%d$" % lam)
plt.legend()
plt.xlabel('$t$')
plt.xticks(np.arange(1, T, 2))
plt.ylabel(r'$\mathbf{1}^TP_t^2 \mathbf{1}$')
plt.tight_layout()
plt.savefig("../figs/payment_squared_payment.pdf")
plt.close()
