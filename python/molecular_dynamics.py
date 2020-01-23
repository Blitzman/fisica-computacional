#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Molecular dynamics.

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

def lennard_jones_potential(
        distance: float,
        epsilon: float,
        sigma: float,
        cutoff: float,
) -> float:
    """ Computes the 12/6 Lennard-Jones potential over a distance.

    Args:
        distance: Radial distance to which compute the potential.
        epsilon: Energy minimum.
        sigma: Distance to zero crossing point.
        cutoff: Maximum range of the potential.

    Returns:
        The 12/6 LJ potential for such distance.
    """

    lj_12_6_potential = 4.0 * epsilon * ((sigma / distance)**12 
                                     - (sigma / distance)**6)

    return lj_12_6_potential

def compute_forces(
        distances: np.array,
        epsilon: float,
        sigma: float,
        cutoff: float,
) -> np.array:

    """ Computes the forces over a distance using the Lennard-Jones 14/8 pot.

    Args:
        distance: Array of per-axis radial distances.
        epsilon: Energy minimum.
        sigma: Distance to zero crossing point.
        cutoff: Maximum range of the potential.

    Returns:
        An array with per-axis forces using the LJ14/8 potential.
    """

    distance = np.sqrt(distances.dot(distances))

    forces = np.zeros_like(distances)

    if distance < cutoff and distance > 0:

        f_lennard_jones_14_8_2 = sigma**2 / distance
        f_lennard_jones_14_8_6 = f_lennard_jones_14_8_2**3
        f_lennard_jones_14_8 = (48.0 * epsilon * f_lennard_jones_14_8_6
                                * (f_lennard_jones_14_8_6 - 0.5) / distance)

        forces = distances * f_lennard_jones_14_8

    return forces


# ##############################################################################

if __name__ == "__main__":

    # Logger setup -------------------------------------------------------------

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Argument parsing ---------------------------------------------------------

    PARSER = argparse.ArgumentParser(description="Advection equation solver")
    PARSER.add_argument("--n", type=int, default=16, help="Number of particles")
    PARSER.add_argument("--dt", type=float, default=0.001, help="dt")
    PARSER.add_argument("--t", type=float, default=1.0, help="Simulation time")
    PARSER.add_argument("--width", type=float, default=4.0, help="Box width")
    PARSER.add_argument("--height", type=float, default=4.0, help="Box height")
    PARSER.add_argument("--cutoff", type=float, default=8.0, help="Lennard-Jones 14/8 cutoff range.")
    PARSER.add_argument("--epsilon", type=float, default=1.0, help="Lennard-Jones 14/8 energy minimum.")
    PARSER.add_argument("--sigma", type=float, default=1.0, help="Lennard-Jones 14/8 distance to zero-crossing point.")
    ARGS = PARSER.parse_args()

    N = ARGS.n
    DT = ARGS.dt
    T = ARGS.t
    BOUNDARIES = [ARGS.width, ARGS.height]
    CUTOFF = ARGS.cutoff
    LJ_EPSILON = ARGS.epsilon
    LJ_SIGMA = ARGS.sigma

    # Initialize arrays --------------------------------------------------------

    position = np.zeros((N, 2))
    velocity = np.zeros((N, 2))
    acceleration = np.zeros((N, 2))

    # Randomize initial positions.
    position[:, 0] = np.random.uniform(low=0.0, high=BOUNDARIES[0], size=N)
    position[:, 1] = np.random.uniform(low=0.0, high=BOUNDARIES[1], size=N)

    x_positions = np.linspace(0.0, BOUNDARIES[0], int(np.sqrt(N)), endpoint=True)
    y_positions = np.linspace(0.0, BOUNDARIES[1], int(np.sqrt(N)), endpoint=True)

    x, y = np.meshgrid(x_positions, y_positions)
    position[:, 0] = x.flatten()
    position[:, 1] = y.flatten()

    # Randomize initial velocities.
    velocity = np.random.uniform(low=-10.0, high=10.0, size=(N, 2))
    LOG.info(velocity)

    # Simulation ---------------------------------------------------------------

    TIME_STEPS = int(T / DT)

    positions = np.zeros((TIME_STEPS, N, 2))
    velocities = np.zeros((TIME_STEPS, N, 2))
    accelerations = np.zeros((TIME_STEPS, N, 2))

    for i in range(TIME_STEPS):

        #LOG.info("Time step: %d", i)

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
                          + velocities[i, p] * DT)
                          #+ accelerations[i, p] * 0.5 * DT ** 2)

            # Check periodic boundaries.
            position[p] = get_inside_periodic_boundary(position[p], BOUNDARIES)

        # Compute accelerations of the current time step.
        for p in range(N):

            acceleration[p] = 0.0
            current_position = position[p]

            for j in range(N):

                distances = current_position - position[j]
                acceleration[p] += compute_forces(
                    distances, LJ_EPSILON, LJ_SIGMA, CUTOFF)

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
    ax1.set_xlim([0.0 - 0.5, BOUNDARIES[0] + 0.5])
    ax1.set_ylim([0.0 - 0.5, BOUNDARIES[1] + 0.5])
    ax1.set_title("Particles")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlabel(r"$x$")

    # Particle plot.
    sc1 = ax1.scatter([], [])

    # Plot boundaries.
    rect = plt.Rectangle((0, 0),
                     BOUNDARIES[0],
                     BOUNDARIES[1],
                     ec='black', lw=2, fc='none')
    ax1.add_patch(rect)

    # Animation ----------------------------------------------------------------

    def animate(step):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle("Time step = " + str(step))
        sc1.set_offsets(positions[step, :])

    anim = animation.FuncAnimation(fig, animate, frames=TIME_STEPS, repeat=False)
    plt.show()