import matplotlib
from math import sqrt
import numpy as np
import networkx as nx


def get_L1(n, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
    G = nx.gnm_random_graph(n, m, directed=True, seed=seed)
    n = len(G.nodes())
    L1 = nx.adjacency_matrix(G).tocsr().astype(float)
    L1.data *= np.exp(np.random.randn(L1.data.size))

    return L1


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 11,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 11,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)
