#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Molecular dynamics.

"""

import argparse
import logging
import sys

import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)


def get_inside_periodic_boundary(
        position: np.array,
        boundary: np.array
) -> np.array:
    """ Get a position inside a defined boundary.
    
    Args:
        position: A position in n-dimensional space.
        boundaries: The boundaries in n-dimensional space.

    Returns:
        An array containing the new position within the boundaries.
    """

    new_position = np.zeros_like(position)

    for i in range(len(position)):
        if position[i] < 0.0:
            new_position[i] = boundary[i] + position[i]
        elif position[i] > boundary[i]:
            new_position[i] = position[i] - boundary[i]
        else:
            new_position[i] = position[i]

    return new_position

def generate_boundary_pseudo_particles(
        particle_positions: np.array,
        boundaries: np.array,
        pseudo_range: float = 3.0,
) -> np.array:
    """ Generates boundary pseudo-particles.

    In order to be able to apply periodic forces, this method spawns the needed
    pseudo-particles at the boundaries.

    Args:
        particle_positions: Array of particle positions in n-dimensional space.
        boundaries: The boundaries in n-dimensional space.
        range: TODO.

    Returns:
        An array containing the original particle positions in n-dimensional
        space plus the positions from the spawned pseudo-particles.
    """

    num_particles = len(particle_positions)
    return np.zeros_like(particle_positions)

def compute_force(
        distance: np.array
) -> np.array:

    return -24.0 * distance * ((distance.dot(distance) ** 3 - 2)) / (distance.dot(distance) ** 7)


# ##############################################################################

if __name__ == "__main__":

    # Logger setup -------------------------------------------------------------

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Argument parsing ---------------------------------------------------------

    PARSER = argparse.ArgumentParser(description="Advection equation solver")
    PARSER.add_argument("--n", type=int, default=4, help="Number of particles")
    PARSER.add_argument("--dt", type=float, default=0.001, help="dt")
    PARSER.add_argument("--t", type=float, default=1.0, help="Simulation time")
    PARSER.add_argument("--width", type=float, default=4.0, help="Box width")
    PARSER.add_argument("--height", type=float, default=4.0, help="Box height")
    PARSER.add_argument("--range", type=float, default=3.0, help="Pot. range")
    ARGS = PARSER.parse_args()

    N = ARGS.n
    DT = ARGS.dt
    T = ARGS.t
    BOUNDARIES = [ARGS.width, ARGS.height]
    POTENTIAL_RANGE = ARGS.range

    # Initialize arrays --------------------------------------------------------

    position = np.zeros((N, 2))
    velocity = np.zeros((N, 2))
    acceleration = np.zeros((N, 2))

    # Randomize initial positions.
    position[:, 0] = np.random.uniform(low=0.0, high=BOUNDARIES[0], size=N)
    position[:, 1] = np.random.uniform(low=0.0, high=BOUNDARIES[1], size=N)
    LOG.info(position)

    # Randomize initial velocities.
    velocity = np.random.uniform(low=0, high=0, size=(N, 2))
    LOG.info(velocity)

    # Simulation ---------------------------------------------------------------

    TIME_STEPS = int(T / DT)

    positions = np.zeros((TIME_STEPS, N, 2))
    velocities = np.zeros((TIME_STEPS, N, 2))
    accelerations = np.zeros((TIME_STEPS, N, 2))

    for i in range(TIME_STEPS):

        LOG.info("Time step: %d", i)

        # TODO: This might be optimizable.
        # Save current time step values for later plotting.
        positions[i] = np.copy(position)
        velocities[i] = np.copy(velocity)
        accelerations[i] = np.copy(acceleration)

        # Compute positions of the current time step using the velocities and
        # accelerations from the previous time step.

        # TODO: Optimize this computation by vectorization.
        for p in range(N):

            position[p] = (positions[i, p]
                          + velocities[i, p] * DT
                          + accelerations[i, p] * 0.5 * DT ** 2)

            # Check periodic boundaries.
            position[p] = get_inside_periodic_boundary(position[p], BOUNDARIES)

        # Compute accelerations of the current time step.
        for p in range(N):

            acceleration[p] = 0.0
            current_position = position[p]

            for j in range(N):

                distance = current_position - position[j]
                distance_sqr = np.sqrt(distance.dot(distance))

                if (distance_sqr < POTENTIAL_RANGE
                    and distance[0] != 0.0
                    and distance[1] != 0.0):

                    acceleration[p] += compute_force(distance)

        # Compute velocities of the current time step.
        for p in range(N):

            velocity[p] = (velocities[i, p]
                           + (accelerations[i, p] + acceleration[p]) * 0.5 * DT)

    # Plotting -----------------------------------------------------------------

    # Plot integration ---------------------------------------------------------

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
    ax1.set_xlim([0.0, BOUNDARIES[0]])
    ax1.set_ylim([0.0, BOUNDARIES[1]])
    ax1.set_title("Particles")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlabel(r"$x$")

    # Particle plot.
    sc1 = ax1.scatter([], [])

    # Animation ----------------------------------------------------------------

    def animate(step):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle("Time step = " + str(step))
        sc1.set_offsets(positions[step, :])

    anim = animation.FuncAnimation(fig, animate, frames=TIME_STEPS, repeat=False)
    plt.show()