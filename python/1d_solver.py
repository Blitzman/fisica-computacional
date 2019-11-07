import argparse
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)

def source(x: float) -> float:
    """ Source function for the Poisson equation.

    Args:
        x: the x_i value to evaluate the function at.

    Returns:
        double: the function value evaluated at x_i.

    """
    return 1.0 - 2.0 * x * x

def fast_tridiagonal_solver(a, b, c, w, u):
    """ Alternative solver for tridiagonal matrix.

    This alternative solver makes use of the a,b,c decomposition of the tridiag.
    matrix M to solve for u = M^-1 w. The decomposition avoids the costly M^-1.

    Args:
        a: first diagonal of tridiagonal M.
        b: second diagonal of tridiagonal M.
        c: third diagonal of tridiagonal M.
        w: source terms.
        u: u-values that will be solved.

    """

    assert a.size == b.size
    assert b.size == c.size
    assert c.size == w.size

    _n = a.size-2

    _x = np.zeros(_n)
    _y = np.zeros(_n)

    _x[_n-1] = -a[_n] / b[_n]
    _y[_n-1] = w[_n] / b[_n]

    for i in range(_n - 2, 0, -1):
        _x[i] = -a[i+1] / (b[i+1] + c[i+1] * _x[i+1])
        _y[i] = (w[i+1] - c[i+1] * _y[i+1]) / (b[i+1] + c[i+1] * _x[i+1])

    _x[0] = 0.0
    _y[0] = (w[1] - c[1] * _y[1]) / (b[1] + c[1] * _x[1])

    u[1] = _y[0]
    for i in range(1, _n):
        u[i+1] = _x[i] * u[i] + _y[i]

def solve(args):

    _n = args.n
    _x_l = args.x_l
    _x_h = args.x_h
    _alpha_l = args.alpha_l
    _alpha_h = args.alpha_h
    _beta_l = args.beta_l
    _beta_h = args.beta_h
    _gamma_l = args.gamma_l
    _gamma_h = args.gamma_h
    _dx = (_x_h - _x_l) / (_n + 1)

    # Initialize the tridiagonal matrix decompositions.
    _tridiagonal_a = np.zeros(_n+2)
    _tridiagonal_b = np.zeros(_n+2)
    _tridiagonal_c = np.zeros(_n+2)

    _tridiagonal_a[2:_n+1] = 1.0
    _tridiagonal_b[1:_n+1] = -2.0
    _tridiagonal_b[1] -= _beta_l / (_alpha_l * _dx - _beta_l)
    _tridiagonal_b[_n] += _beta_h / (_alpha_h * _dx + _beta_h) 
    _tridiagonal_c[:_n] = 1.0

    LOG.info("Tridiagonal A: %s", _tridiagonal_a)
    LOG.info("Tridiagonal B: %s", _tridiagonal_b)
    LOG.info("Tridiagonal C: %s", _tridiagonal_c)

    # Initialize source terms vector.
    _source_terms = np.zeros(_n+2)
    for i in range(0, _n+1):
        _x_i = _x_l + (i * (_x_h - _x_l)) / (_n+1)
        _source_terms[i] = source(_x_i) * _dx * _dx
    _source_terms[1] -= _gamma_l * _dx / (_alpha_l * _dx - _beta_l)
    _source_terms[_n] -= _gamma_h * _dx / (_alpha_h * _dx + _beta_h)

    LOG.info("Source terms: %s", _source_terms)

    # Fast solve tridiagonal matrix equation.
    _u = np.zeros(_n+2)
    fast_tridiagonal_solver(_tridiagonal_a,
                            _tridiagonal_b,
                            _tridiagonal_c,
                            _source_terms,
                            _u)

    # Compute u-values.
    _u[0] = (_gamma_l * _dx - _beta_l * _u[1]) / (_alpha_l * _dx - _beta_l)
    _u[_n+1] = (_gamma_h * _dx + _beta_h * _u[_n]) / (_alpha_h * _dx + _beta_h)

    # Plot
    plt.plot(np.linspace(_x_l, _x_h, _n+2), _u)
    plt.ylim((0.7, 0.9))
    plt.xlabel("x_i", fontsize=16)
    plt.ylabel("u", fontsize=16)
    plt.show()


if __name__ == "__main__":

    _PARSER = argparse.ArgumentParser(description="Parameters")
    _PARSER.add_argument('--n', default='100', type=int)
    _PARSER.add_argument('--x_l', default='0.0', type=float)
    _PARSER.add_argument('--x_h', default='1.0', type=float)
    _PARSER.add_argument('--alpha_l', default='1.0', type=float)
    _PARSER.add_argument('--alpha_h', default='1.0', type=float)
    _PARSER.add_argument('--beta_l', default='-1.0', type=float)
    _PARSER.add_argument('--beta_h', default='1.0', type=float)
    _PARSER.add_argument('--gamma_l', default='1.0', type=float)
    _PARSER.add_argument('--gamma_h', default='1.0', type=float)
    _ARGS = _PARSER.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    solve(_ARGS)