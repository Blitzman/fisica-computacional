#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Schrodinger solver.

    Solving Schrödinger's equation using a FDTD(FTCS) scheme. The code spawns a
    wave with certain specified characteristics and a potential barrier with
    a defined extent and amplitude.

    After the simulation, the code produces a video schrodinger.mp4 with the
    plot showing the evolution.

    Run: python3 schrodinger.py --help
        for help on the parameters that can be tuned.

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
        range: list,
        amplitude: float
) -> np.array:

    """ Generate a potential barrier.

    Args:
        points: Number of points in the spatial domain.
        range: Pair of points in the spatial domain (left, right).
        amplitude: Amplitude for the potential barrier.

    Returns:
        An array of the same size as the spatial domain (in points) with all
        zeroes except for the points which fall in the range of the potential
        barrier which contain its amplitude.

    """

    _potential = np.zeros(points)
    _potential[range[0]:range[1]] = amplitude
    return _potential

def generate_gaussian(
        x: np.array,
        t: float,
        sigma: float,
) -> np.array:

    """ Generate a Gaussian.

    Args:
        x: Spatial domain.
        t: Time shift.
        sigma: standard deviation.

    Returns:
        An array the same size as the spatial domain with the values for the
        Gaussian shifted by a determined time shift with the specified
        standard deviation.

    """

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
        default=1024,
        help="Number of spatial points")
    PARSER.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step size")
    PARSER.add_argument(
        "--t",
        type=float,
        default=4096.0,
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
        "--gaussian_shift",
        type=float,
        default=256.0,
        help="Gaussian shift from the origin")
    PARSER.add_argument(
        "--gaussian_sigma",
        type=float,
        default=32.0,
        help="Gaussian sigma for starting wave function")
    PARSER.add_argument(
        "--k",
        type=int,
        default=16,
        help="Wave number")
    PARSER.add_argument(
        "--snapshot",
        type=int,
        default=64,
        help="Snapshot frequency"
    )
    PARSER.add_argument(
        "--v",
        type=float,
        default=1e-2,
        help="Potential amplitude"
    )
    PARSER.add_argument(
        "--v_range",
        nargs='+',
        type=int,
        default=[512, 544],
        help="Potential range"
    )
    ARGS = PARSER.parse_args()

    N = ARGS.n
    T = ARGS.t
    DT = ARGS.dt
    DX = ARGS.dx
    MASS = ARGS.mass
    GAUSSIAN_SIGMA = ARGS.gaussian_sigma
    SNAPSHOT_FREQUENCY = ARGS.snapshot
    POTENTIAL_AMPLITUDE = ARGS.v
    POTENTIAL_RANGE = ARGS.v_range
    GAUSSIAN_SHIFT = ARGS.gaussian_shift
    K = ARGS.k

    # Other constants ----------------------------------------------------------

    # Number of timesteps.
    TIME_STEPS = int(T / DT)
    # Number of snapshots.
    SNAPSHOTS = int(TIME_STEPS / SNAPSHOT_FREQUENCY)
    # Planck's constant.
    # H = 6.626e-34
    H = 1.0 # Assume unit to accelerate simulation.
    # Spatial grid of points.
    X = np.linspace(0.0, N, N) * DX
    # Potential barrier.
    POTENTIAL = generate_potential_barrier(
        N,
        POTENTIAL_RANGE,
        POTENTIAL_AMPLITUDE
    )
    # Initial wave funtion.
    INITIAL_RANGE = range(1, int(N / 2))
    X_INITIAL = X[INITIAL_RANGE] / DX

    K0 = np.pi / K

    GAUSSIAN = generate_gaussian(
        X_INITIAL,
        GAUSSIAN_SHIFT,
        GAUSSIAN_SIGMA
    )
    INITIAL_COSINE = np.cos(K0 * X_INITIAL)
    INITIAL_SINE = np.sin(K0 * X_INITIAL)

    # Initialization of wave function ------------------------------------------

    # Arrays for storing the real, imaginary and probability values for the wave
    # note that the real and imaginary arrays have three dimensions to store the
    # past, current, and future values (n-1, n, n+1 respectively).
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

    _psi_reals = np.zeros((SNAPSHOTS+1, N))
    _psi_imaginaries = np.zeros((SNAPSHOTS+1, N))
    _psi_probabilities = np.zeros((SNAPSHOTS+1, N))

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

        # Log current values for later visualization.

        if (t % SNAPSHOT_FREQUENCY == 0):

            _snapshot = int(t / SNAPSHOT_FREQUENCY)

            LOG.info("Timestep {}/{}".format(t, TIME_STEPS))
            _psi_reals[_snapshot] = _psi_real[:, 1]
            _psi_imaginaries[_snapshot] = _psi_imaginary[:, 1]
            _psi_probabilities[_snapshot] = (_psi_imaginary[:, 1]**2
                                             + _psi_real[:, 1]**2)

        # Pre-fetch some recurrent values for efficiency.

        _psi_real_present = _psi_real[:, 1]
        _psi_imaginary_present = _psi_imaginary[:, 1]

        # Update equations FTCS/FDTD.

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

    # Plotting -----------------------------------------------------------------

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
    fig = plt.figure(figsize=(32, 16))

    ax1 = fig.add_subplot(111)
    ax1.set_xlim([X.min(), X.max()])
    ax1.set_ylim([-_psi_reals.max() * 1.25, _psi_reals.max() * 1.25])
    ax1.set_title("Schrodinger solver with potential barrier", pad=16)
    ax1.set_ylabel(r"$y$", labelpad=16)
    ax1.set_xlabel(r"$x$", labelpad=16)

    _line_real, = ax1.plot(
        X,
        _psi_reals[0],
        "red",
        label="Real"
    )
    _line_imaginary, = ax1.plot(
        X,
        _psi_imaginaries[0],
        "blue",
        label="Imaginary"
    )
    _line_probabilities, = ax1.plot(
        X,
        _psi_probabilities[0],
        "black",
        label="Probability"
    )

    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"V", labelpad=16)
    ax2.fill(
        [
            POTENTIAL_RANGE[0],
            POTENTIAL_RANGE[0],
            POTENTIAL_RANGE[1],
            POTENTIAL_RANGE[1]
        ],
        [0.0, POTENTIAL_AMPLITUDE, POTENTIAL_AMPLITUDE, 0.0],
        color="yellow",
        alpha=0.25
    )

    def animate(i):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle(
            "Time step = "
            + str(i)
        )

        _line_real.set_ydata(_psi_reals[i])
        _line_imaginary.set_ydata(_psi_imaginaries[i])
        _line_probabilities.set_ydata(6.0 * _psi_probabilities[i])

    _FRAMES = SNAPSHOTS
    _ANIMATION = animation.FuncAnimation(
        fig,
        animate,
        frames=_FRAMES,
        repeat=False)

    #plt.show()

    # Write video output of the animation --------------------------------------

    LOG.info("Generating video schrodinger.mp4...")

    _writer = animation.writers['ffmpeg']
    _writer = _writer(
        fps=24,
        metadata=dict(artist="Albert Garcia"),
        bitrate=1800)
    _ANIMATION.save("schrodinger.mp4", writer=_writer)

    LOG.info("Video schrodinger.mp4 generated!")