#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Monte Carlo integration.

    This program performs a simple Monte Carlo integration of the provided
    function to calculate its integral for a definite interval. The program
    is intented to integrate pathological functions.
"""

import argparse
import logging
import random
import sys

import numexpr as ne
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)

# ##############################################################################

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Argument parsing ---------------------------------------------------------

    PARSER = argparse.ArgumentParser(description="Monte Carlo integration")
    PARSER.add_argument("--f", type=str, default="sin(_x)", help="Function")
    PARSER.add_argument("--a", type=float, default=0.0, help="Low interval")
    PARSER.add_argument("--b", type=float, default=4.0, help="High interval")
    PARSER.add_argument("--ymin", type=float, default=0.0, help="Minimum y")
    PARSER.add_argument("--ymax", type=float, default=1.0, help="Maximum y")
    PARSER.add_argument("--n", type=int, default=1024, help="Number of points")
    ARGS = PARSER.parse_args()

    F = ARGS.f
    A = ARGS.a
    B = ARGS.b
    N = ARGS.n
    Y_MIN = ARGS.ymin
    Y_MAX = ARGS.ymax

    LOG.info("Evaluating function %s", F)

    def _f(_x: float) -> float:
        return ne.evaluate(F)

    # Perform Monte Carlo integration ------------------------------------------

    # Compute the area of the rectangle that bounds our function and integral
    # limits by just multiplying the ranges.
    BOUNDING_RECTANGLE_AREA = (B - A) * (Y_MAX - Y_MIN)

    count_below_function = 0
    points = []
    integrals = []
    samples = []
    colors = []

    for i in range(N):

        # Randomly sample values for x and y inside the defined intervals for
        # integration and the rectangle to consider.
        x = random.uniform(A, B)
        y = random.uniform(Y_MIN, Y_MAX)

        # Check if the point falls inside the curve or not, in which case we
        # add it to the count.
        count = 0
        if y < _f(x):
            count = 1

        # Add the point to the count and compute the integral for this step as
        # the product of the bounding rectangle and the ration of the points
        # that fall inside the function over the total number of samples.
        count_below_function += count
        INTEGRAL = BOUNDING_RECTANGLE_AREA * count_below_function / N

        # Append values to lists to keep track of them over time and plot.
        colors.append(count)
        points.append([x, y])
        integrals.append(INTEGRAL)
        samples.append(i)

    points = np.array(points)

    LOG.info("Integral value Monte Carlo is %f", integrals[N-1])

    # Plot integration ---------------------------------------------------------

    # Set LaTeX font and appropriate sizes.
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    plt.rc('font', size=24) # Controls default text sizes.
    plt.rc('axes', titlesize=32) # Fontsize of the axes title.
    plt.rc('axes', labelsize=24) # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=24) # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=24) # Fontsize of the tick labels.
    plt.rc('legend', fontsize=24) # Legend fontsize.
    plt.rc('figure', titlesize=32) # Fontsize of the figure title.

    # Create figure and arrange axes.
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.set_xlim([A, B])
    ax1.set_ylim([Y_MIN, Y_MAX])
    ax1.set_title("Monte Carlo sampling")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlabel(r"$x$")

    ax2 = fig.add_subplot(122)
    ax2.set_xlim([0, N])
    ax2.set_ylim([0, np.max(integrals) * 1.5])
    ax2.set_title("Integral evolution")
    ax2.set_ylabel(r"Integral value")
    ax2.set_xlabel(r"Step")

    # Plot function.
    x = np.linspace(A, B, 256)
    y = _f(x)
    ax1.plot(x, y)

    # Plot Monte Carlo sampling to animate later.
    sc1 = ax1.scatter(points[:, 0], points[:, 1], c=colors)

    # Plot integral evolution line.
    sc2, = ax2.plot([], [])

    # Animation ----------------------------------------------------------------

    def animate(step):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle("Monte Carlo integration, sample = " + str(step))
        sc1.set_offsets(points[:step])
        sc2.set_data([samples[:step], integrals[:step]])

    anim = animation.FuncAnimation(fig, animate, frames=N, repeat=False)
    plt.show()

    # Write video output of the animation --------------------------------------

    writer = animation.writers['ffmpeg']
    writer = writer(fps=15, metadata=dict(artist="Albert Garcia"), bitrate=1800)
    anim.save("montecarlo.mp4", writer=writer)
