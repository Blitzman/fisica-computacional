#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Schrodinger solver.

    Author: Alberto Garcia Garcia (albert.garcia.ua@gmail.com)

"""

import argparse
import itertools
import logging
import sys

import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)

def generate_potential_barrier(
        points: int,
        thickness: int,
        amplitude: float
) -> np.array:

    _potential = np.zeros(points)
    _START = int(points / 2)
    _END = int(points / 2 + thickness)
    _potential[_START:_END] = amplitude
    return _potential

def generate_gaussian(
        x: np.array,
        t: float,
        sigma: float,
) -> np.array:

    return np.exp(-(x - t)**2 / (2.0 * sigma**2))

# ##############################################################################

if __name__ == "__main__":

    # Logger setup -------------------------------------------------------------

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Argument parsing ---------------------------------------------------------

    PARSER = argparse.ArgumentParser(description="Advection equation solver")
    PARSER.add_argument(
        "--n",
        type=int,
        default=1200,
        help="Number of spatial points")
    PARSER.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Time step size")
    PARSER.add_argument(
        "--t",
        type=float,
        default=48.0,
        help="Total time")
    PARSER.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Spatial resolution")
    PARSER.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Mass")
    PARSER.add_argument(
        "--gaussian_sigma",
        type=float,
        default=40.0,
        help="Gaussian sigma for starting wave function.")
    ARGS = PARSER.parse_args()

    N = ARGS.n
    #T = ARGS.t
    #DT = ARGS.dt
    DX = ARGS.dx
    MASS = ARGS.mass
    GAUSSIAN_SIGMA = ARGS.gaussian_sigma

    # Other constants ----------------------------------------------------------

    # Number of timesteps.
    TIME_STEPS = 5 * N
    # Planck's constant.
    H = 1.0e0
    # Amplitude of the potential barrier.
    POTENTIAL_AMPLITUDE = 1.0e-2
    # Thicnkess of the potential barrier.
    POTENTIAL_THICKNESS = 15
    # Spatial grid of points.
    X = np.linspace(0.0, N, N) * DX
    # Potential barrier.
    POTENTIAL = generate_potential_barrier(
        N,
        POTENTIAL_THICKNESS,
        POTENTIAL_AMPLITUDE
    )
    # Critical time step size.
    DT = H / (2.0 * H**2 / (MASS * DX**2) + POTENTIAL.max())
    # Initial wave funtion.
    INITIAL_RANGE = range(1, int(N / 2))
    X_INITIAL = X[INITIAL_RANGE] / DX

    X0 = round(N / 2.0) - 5.0 * GAUSSIAN_SIGMA
    K0 = np.pi / 20.0

    GAUSSIAN = generate_gaussian(
        X_INITIAL,
        X0,
        GAUSSIAN_SIGMA
    )
    INITIAL_COSINE = np.cos(K0 * X_INITIAL)
    INITIAL_SINE = np.sin(K0 * X_INITIAL)
    # Energy of the wave.
    WAVE_ENERGY = (H**2 / 2.0 / MASS) * (K0**2 + 0.5 / GAUSSIAN_SIGMA**2)

    # Initialization of wave function ------------------------------------------

    _psi_real = np.zeros((N, 3))
    _psi_imaginary = np.zeros((N, 3))
    _psi_probability = np.zeros(N)

    # Fill with initial values in the initial range.
    _psi_real[INITIAL_RANGE, 0] = INITIAL_COSINE * GAUSSIAN
    _psi_imaginary[INITIAL_RANGE, 1] = INITIAL_SINE * GAUSSIAN

    _psi_real[INITIAL_RANGE, 1] = INITIAL_COSINE * GAUSSIAN
    _psi_imaginary[INITIAL_RANGE, 1] = INITIAL_SINE * GAUSSIAN

    # Initial observable probability.
    _psi_probability = _psi_imaginary[:, 1]**2 + _psi_real[:, 1]**2

    # Normalize wave function to keep the total probability equal to one.
    _TOTAL_PROBABILITY = _psi_probability.sum() * DX
    _PROBABILITY_NORM = np.sqrt(_TOTAL_PROBABILITY)
    _psi_real /= _PROBABILITY_NORM
    _psi_imaginary /= _PROBABILITY_NORM
    _psi_probability /= _TOTAL_PROBABILITY

    # Solve it! ----------------------------------------------------------------

    _psi_reals = np.zeros((TIME_STEPS, N))
    _psi_imaginaries = np.zeros((TIME_STEPS, N))
    _psi_probabilities = np.zeros((TIME_STEPS, N))

    # Precompute indices for k+1 accesses.
    _PREVIOUS_INDICES = range(0, N-2)
    # Precompute indices for k accesses.
    _CURRENT_INDICES = range(1, N-1)
    # Precompute indices for k-1 accesses.
    _NEXT_INDICES = range(2, N)

    # Precompute constant coefficients for efficiency.
    _C1 = H * DT / (MASS * DX**2)
    _C2 = 2.0 * DT / H * POTENTIAL

    for t in range(TIME_STEPS):

        #LOG.info("Timestep {}/{}".format(i, TIME_STEPS))

        _psi_real_present = _psi_real[:, 1]
        _psi_imaginary_present = _psi_imaginary[:, 1]

        # Update equations FTCS.

        _psi_imaginary[_CURRENT_INDICES, 2] = (
            _psi_imaginary[_CURRENT_INDICES, 0]
            + _C1 * (
                _psi_real_present[_NEXT_INDICES]
                - 2.0 * _psi_real_present[_CURRENT_INDICES]
                + _psi_real_present[_PREVIOUS_INDICES]
            )
        )

        _psi_imaginary[:, 2] -= _C2 * _psi_real[:, 1]

        _psi_real[_CURRENT_INDICES, 2] = (
            _psi_real[_CURRENT_INDICES, 0]
            - _C1 * (
                _psi_imaginary_present[_NEXT_INDICES]
                - 2.0 * _psi_imaginary_present[_CURRENT_INDICES]
                + _psi_imaginary_present[_PREVIOUS_INDICES]
            )
        )

        _psi_real[:, 2] += _C2 * _psi_imaginary[:, 1]

        # Now future is present, present is past so prepare next time step.

        _psi_real[:, 0] = _psi_real[:, 1]
        _psi_real[:, 1] = _psi_real[:, 2]
        _psi_imaginary[:, 0] = _psi_imaginary[:, 1]
        _psi_imaginary[:, 1] = _psi_imaginary[:, 2]

        # Log current values for later visualization.

        _psi_reals[t] = _psi_real[:, 1]
        _psi_imaginaries[t] = _psi_imaginary[:, 1]
        _psi_probabilities[t] = _psi_imaginary[:, 1]**2 + _psi_real[:, 1]**2

    # Plotting -----------------------------------------------------------------

    # Set LaTeX font and appropriate sizes.
    #matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    #matplotlib.rc('text', usetex=True)
    plt.rc('font', size=24) # Controls default text sizes.
    plt.rc('axes', titlesize=32) # Fontsize of the axes title.
    plt.rc('axes', labelsize=24) # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=24) # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=24) # Fontsize of the tick labels.
    plt.rc('legend', fontsize=24) # Legend fontsize.
    plt.rc('figure', titlesize=32) # Fontsize of the figure title.

    # Create figure and arrange axes.
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xlim([X.min(), X.max()])
    ax1.set_ylim([-_psi_reals.max() * 1.25, _psi_reals.max() * 1.25])
    ax1.set_title("Schrodinger", pad=16)
    ax1.set_ylabel(r"$y$", labelpad=16)
    ax1.set_xlabel(r"$x$", labelpad=16)

    _line_real, = ax1.plot(X, _psi_reals[0], 'red', label="Real")
    _line_imaginary, = ax1.plot(X, _psi_imaginaries[0], 'blue', label="Imaginary")
    _line_probabilities, = ax1.plot(X, _psi_probabilities[0], 'black', label="Probability")

    _wave_energy_factor = _psi_reals.max() * 0.5
    if POTENTIAL.max() > 0.0:
        _wave_energy_factor /= POTENTIAL.max()

    ax1.axhline(
        WAVE_ENERGY * _wave_energy_factor,
        color="green",
        label="Wave Energy"
    )

    ax1.legend(loc="lower left")

    def animate(i):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle(
            "Time step = "
            + str(i)
        )

        _line_real.set_ydata(_psi_reals[i])
        _line_imaginary.set_ydata(_psi_imaginaries[i])
        _line_probabilities.set_ydata(6.0 * _psi_probabilities[i])

    _FRAMES = TIME_STEPS
    _ANIMATION = animation.FuncAnimation(
        fig,
        animate,
        frames=_FRAMES,
        repeat=False)

    plt.show()