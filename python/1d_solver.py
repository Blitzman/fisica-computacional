import argparse
import logging
import sys
import scipy.sparse

import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def source(x):
    return 1.0 - 2.0 * x * x

def fast_tridiagonal_solver(a, b, c, w, u):

    assert(a.size == b.size)
    assert(b.size == c.size)
    assert(c.size == w.size)

    n = a.size-2

    x = np.zeros(n)
    y = np.zeros(n)

    x[n-1] = -a[n] / b[n]
    y[n-1] = w[n] / b[n]

    for i in range (n-2, 0):
        x[i] = -a[i+1] / (b[i+1] + c[i+1] * x[i+1])
        y[i] = (w[i+1] - c[i+1] * y[i+1]) / (b[i+1] + c[i+1] * x[i+1])

    x[0] = 0.0
    y[0] = (w[1] - c[1] * y[1]) / (b[1] + c[1] * x[1])

    u[1] = y[0]
    for i in range(1, n):
        u[i+1] = x[i] * u[i] + y[i]

def solve(args):

    n = 100
    x_l = 0.0
    x_h = 1.0
    alpha_l = 1.0
    alpha_h = 1.0
    beta_l = -1.0
    beta_h = 1.0
    gamma_l = 1.0
    gamma_h = 1.0
    dx = (x_h - x_l) / (n + 1)

    # Initialize the tridiagonal matrix.
    tridiagonal_a = np.zeros(n+2)
    tridiagonal_b = np.zeros(n+2)
    tridiagonal_c = np.zeros(n+2)

    tridiagonal_a[2:n+1] = 1.0
    tridiagonal_b[1:n+1] = -2.0
    tridiagonal_b[1] -= beta_l / (alpha_l * dx - beta_l)
    tridiagonal_b[n] += beta_h / (alpha_h * dx + beta_h) 
    tridiagonal_c[:n] = 1.0

    log.info(tridiagonal_a)
    log.info(tridiagonal_b)
    log.info(tridiagonal_c)

    # Initialize source terms vector.
    source_terms = np.zeros(n+2)
    for i in range(1, n+1):
        source_terms[i] = source(i) * dx * dx
    source_terms[1] -= gamma_l * dx / (alpha_l * dx - beta_l)
    source_terms[n] -= gamma_h * dx / (alpha_h * dx + beta_h)

    log.info(source_terms)

    # Fast invert tridiagonal matrix equation.
    u = np.zeros(n+2)
    fast_tridiagonal_solver(tridiagonal_a,
                            tridiagonal_b,
                            tridiagonal_c,
                            source_terms,
                            u)

    # Compute u-values.
    u[0] = (gamma_l * dx - beta_l * u[1]) / (alpha_l * dx - beta_l)
    u[n+1] = (gamma_h * dx + beta_h * u[n]) / (alpha_h * dx + beta_h)

    # Plot
    plt.plot(np.linspace(x_l, x_h, n+2), u)
    plt.show()


if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    args_ = parser_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    solve(args_)