# -*- coding: utf-8 -*-
""" Fourth-order Runge-Kutta method

    This program applies the fourth-order Runge-Kutta method to solve an specified
    ODE. Both the analytical and the RK4 solved functions are plotted and compared.
    The resulting figure is stored as "rungekutta4_plot.png".

    The analyical function and the ODE must be hard-coded in this script.

    Other parameters are accepted by command line (start/end of the interval,
    time step size, and initial value). Run the script with --help for more
    information about the available parameters.

"""
import argparse
import matplotlib.pyplot as plt
import numpy as np

def function(t):
    """ Analytical function to be solved. """
    return np.sin(t)
def first_derivative(x, t):
    """ First derivative (ODE) of the analyical function. """
    return np.cos(t)
def rungekutta4(x, t, h, f):
    """ Perform one step of fourth-order Runge-Kutta
    x -- dependent variable value at the current time step.
    t -- current time step.
    h -- time step size.
    f -- first derivative of the function to be solved (ODE).
    """
    k1_ = h * f(x, t)
    k2_ = h * f(x + 0.5 * k1_, t + 0.5 * h)
    k3_ = h * f(x + 0.5 * k2_, t + 0.5 * h)
    k4_ = h * f(x + k3_, t + h)
    return x + (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6.0

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description='Fourth-order Runge-Kutta solver for ODE')
    parser_.add_argument('--a', type=float, default=0.0, help='Start of the interval')
    parser_.add_argument('--b', type=float, default=8.0, help='End of the interval')
    parser_.add_argument('--h', type=float, default=1.0, help='Step size')
    parser_.add_argument('--x0', type=float, default=0.0, help='Initial condition')
    args_ = parser_.parse_args()

    a_ = args_.a
    b_ = args_.b
    h_ = args_.h
    x0_ = args_.x0

    print("Interval start: ", a_)
    print("Interval end: ", b_)
    print("Step size: ", h_)

    # Solve with fourth-order Runge-Kutta meth----------------------------------
    # Arrange time step points from the start to the end of the inverval.
    rungekutta_t_ = np.arange(a_, b_, h_)
    rungekutta_x_ = []

    x_ = x0_
    for t in rungekutta_t_:
      rungekutta_x_.append(x_)
      x_ = rungekutta4(x_, t, h_, first_derivative)

    # Solve with analytical ----------------------------------------------------
    analytical_t_ = np.arange(a_, b_, 0.001)
    analytical_x_ = function(analytical_t_)

    # Plot both solutions ------------------------------------------------------
    fig_ = plt.figure()
    ax_ = plt.axes()
    # Draw Euler solution.
    ax_.plot(rungekutta_t_, rungekutta_x_)
    # Draw analytical solution.
    ax_.plot(analytical_t_, analytical_x_)
    # Draw vertical lines for Euler timesteps.
    for t in rungekutta_t_:
      ax_.axvline(x=t, alpha=0.2, color='k', linestyle='--')
    ax_.set_ylabel("x(t)")
    ax_.set_xlabel("t [s]")
    ax_.legend(["Runge-Kutta 4 (h = " + str(h_) + ")", "Analytical"])
    plt.savefig("rungekutta4_plot.png")