import argparse
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

LOG = logging.getLogger(__name__)

def source(x: float) -> float:
    """ Source function for the Poisson equation.

    Args:
        x: the x_i value to evaluate the function at.

    Returns:
        double: the function value evaluated at x_i.

    """
    return x**2

def solve(args):

    _n = args.n
    _x_l = args.x_l
    _x_h = args.x_h
    _gamma_l = args.gamma_l
    _gamma_h = args.gamma_h
    _dx = (_x_h - _x_l) / (_n)

    # Generate tridiagonal matrix ----------------------------------------------
    _diagonal_a = np.ones(_n)
    _diagonal_b = -2.0 * np.ones(_n)
    _diagonal_c = np.ones(_n)
    _diagonals = np.vstack((_diagonal_a, _diagonal_b, _diagonal_c))
    _tridiagonal = scipy.sparse.spdiags(_diagonals, (-1, 0, 1), _n, _n)
    _tridiagonal = np.array(_tridiagonal.todense())
    print(_tridiagonal)

    # Generate u-values vector -------------------------------------------------
    _u_values = np.zeros(_n)

    # Generate source vector ---------------------------------------------------
    _source_terms = np.zeros(_n)
    for i in range(0, _n):
        _x_i = _x_l + (i * (_x_h - _x_l)) / (_n+1)
        _source_terms[i] = source(_x_i) * _dx * _dx

    # Include Dirichlet boundary conditions ------------------------------------
    _source_terms[0] -= _gamma_l
    _source_terms[_n-1] -= _gamma_h

    # Solve the system Mu = s --------------------------------------------------
    _tridiagonal_inv = np.linalg.inv(_tridiagonal)
    print(_tridiagonal_inv.size)
    print(_source_terms.shape)
    _u_values = np.matmul(_tridiagonal_inv, _source_terms)

    # Plot ---------------------------------------------------------------------
    plt.plot(np.linspace(_x_l, _x_h, _n), _u_values)
    #plt.ylim((0.0, 1.0))
    plt.xlabel("x_i", fontsize=16)
    plt.ylabel("u", fontsize=16)
    plt.show()

if __name__ == "__main__":

    _PARSER = argparse.ArgumentParser(description="Parameters")
    _PARSER.add_argument('--n', default=100, type=int)
    _PARSER.add_argument('--x_l', default=0.0, type=float)
    _PARSER.add_argument('--x_h', default=1.0, type=float)
    _PARSER.add_argument('--gamma_l', default=0.0, type=float)
    _PARSER.add_argument('--gamma_h', default=1.0, type=float)
    _ARGS = _PARSER.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    solve(_ARGS)