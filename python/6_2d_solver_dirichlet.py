import argparse
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.sparse

LOG = logging.getLogger(__name__)

def source(x: float, y: float) -> float:
    """ Source function for the Poisson equation.

    Args:
        x: the x_i value to evaluate the function at.

    Returns:
        double: the function value evaluated at x_i.

    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def solve(args):

    _n = args.n
    _m = args.m
    _x_l = args.x_l
    _x_h = args.x_h
    _y_l = args.y_l
    _y_h = args.y_h
    _dx = (_x_h - _x_l) / (_n)
    _dy = (_y_h - _y_l) / (_m)

    # Generate tridiagonal matrix ----------------------------------------------
    _diagonal_a = np.ones(_n+1)
    _diagonal_b = -2.0 * np.ones(_n+1)
    _diagonal_c = np.ones(_n+1)
    _diagonals = np.vstack((_diagonal_a, _diagonal_b, _diagonal_c))
    _tridiagonal = scipy.sparse.spdiags(_diagonals, (-1, 0, 1), _n, _n)
    _tridiagonal = -1.0 * np.array(_tridiagonal.todense())
    print(_tridiagonal)

    # Generate block tridiagonal matrix ----------------------------------------
    _block_tridiagonal = np.zeros((_n * _m, _n * _m))
    _diagonal = np.eye(_n)
    _block = _tridiagonal + 2.0 * _diagonal

    print(_block)

    for i in range(0, _m):
        _start_idx = i * _n 
        _end_idx = (i + 1) * _n
        _block_tridiagonal[_start_idx:_end_idx, _start_idx:_end_idx] = _block

    print(_block_tridiagonal)

    # Invert block tridiagonal matrix ------------------------------------------
    _block_tridiagonal_inv = np.linalg.inv(_block_tridiagonal)

    # Generate u-values vector -------------------------------------------------
    _u_values = np.zeros(_n * _m)

    # Generate source vector ---------------------------------------------------
    _source_terms = np.zeros(_n * _m)
    for i in range(0, _n):
        for j in range(0, _m):
            _idx = j * _n + i
            _x_i = _x_l + (i * (_x_h - _x_l)) / (_n + 1)
            _y_i = _y_l + (j * (_y_h - _y_l)) / (_m + 1)
            _source_terms[_idx] = source(_x_i, _y_i) * _dx * _dx * _dy * _dy

    # Include Dirichlet boundary conditions ------------------------------------
    # for i in range(0, _n):
    #     for j in range(0, _m):
    #         if (i == 0 or j == 0 or i == _n-1 or j == _m-1):
    #             _x_i = _x_l + (i * (_x_h - _x_l)) / (_n + 1)
    #             _y_i = _y_l + (j * (_y_h - _y_l)) / (_m + 1)
    #             _source_terms[_idx] = 0

    # Solve the system Mu = s --------------------------------------------------
    _u_values = np.matmul(_block_tridiagonal_inv, _source_terms)

    # Plot ---------------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _x_values = np.linspace(_x_l, _x_h, _n)
    _y_values = np.linspace(_y_l, _y_h, _m)
    X, Y = np.meshgrid(_x_values, _y_values)
    zs = np.array(source(np.ravel(X), np.ravel(Y)))
    zss = np.array(_u_values)
    Z = zs.reshape(X.shape)
    ZSS = zss.reshape(X.shape)

    #ax.plot_surface(X, Y, Z)
    ax.plot_surface(X, Y, ZSS, cmap=cm.coolwarm)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(0, _x_h)
    ax.set_ylim(0, _y_h)
    #ax.set_zlim(0,1)

    plt.show()

if __name__ == "__main__":

    _PARSER = argparse.ArgumentParser(description="Parameters")
    _PARSER.add_argument('--n', default=100, type=int)
    _PARSER.add_argument('--m', default=100, type=int)
    _PARSER.add_argument('--x_l', default=0.0, type=float)
    _PARSER.add_argument('--x_h', default=1.0, type=float)
    _PARSER.add_argument('--y_l', default=0.0, type=float)
    _PARSER.add_argument('--y_h', default=1.0, type=float)
    _ARGS = _PARSER.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    solve(_ARGS)