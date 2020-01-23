#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Molecular dynamics.

    This script performs a simulation of molecular dynamics by placing a set of
    particles in a regular grid.
    
    The velocity Verlet algorithm is used to integrate position, velocity,
    and acceleration.
    
    The interaction potential is Lennard-Jones 14/8.

    The simulation is carried out with periodic boundaries and forces which are
    computed by placing pseudo-particles in such boundaries.

    Furthermore, you can set a maximum for the temperature of the system which
    is somewhat achieved through a thermostat.

    Running the code:

        python3 5_molecular_dynamics.py --h

        To obtain help about the different parameters that can be customized.

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
        cutoff: float,
) -> np.array:
    """ Generates boundary pseudo-particles.

    In order to be able to apply periodic forces, this method spawns the needed
    pseudo-particles at the boundaries.

    Args:
        particle_positions: Array of particle positions in n-dimensional space.
        boundaries: The boundaries in n-dimensional space.
        cutoff: Potential cutoff range.

    Returns:
        An array containing the original particle positions in n-dimensional
        space plus the positions from the spawned pseudo-particles.
    """

    num_particles = len(particle_positions)
    pseudo_particles = []

    for i in range(num_particles):

        particle_x = particle_positions[i, 0]
        particle_y = particle_positions[i, 1]

        # Top boundary.
        if (boundaries[1] >= particle_y >= (boundaries[1] - cutoff)):
            pseudo_particles.append(
                [
                    particle_x,
                    particle_y - boundaries[1]
                ]
            )

        # Right boundary.
        if (boundaries[0] >= particle_x >= (boundaries[0] - cutoff)):
            pseudo_particles.append(
                [
                    particle_x - boundaries[0],
                    particle_y
                ]
            )

        # Bottom boundary.
        if (cutoff >= particle_y >= 0):
            pseudo_particles.append(
                [
                    particle_x,
                    particle_y + boundaries[1]
                ]
            )

        # Left boundary.
        if (cutoff >= particle_x >= 0):
            pseudo_particles.append(
                [
                    particle_x + boundaries[0],
                    particle_y
                ]
            )

    return np.array(pseudo_particles)

def compute_temperature(
        velocities: np.array
) -> float:
    """ Calculate system temperature.

    Using the velocity of each particle (and assuming m=1) we can compute the
    kinetic energy for each one of them K = 0.5 * m * v^2. Averaging the
    kinectic energy across the whole system (and assuming 1.0 for coefficients
    for the sake of simplicity) we can compute the temperature as
    T = 2.0 * avg K / dimensions.

    Args:
        velocities: Array of velocities for each particle in each dimension.

    Returns:
        Average temperature of the system.
    """

    kinetic_energy = 0.0

    for i in range(len(velocities)):
      kinetic_energy += 0.5 * np.dot(velocities[i], velocities[i])

    average_kinectic_energy = kinetic_energy / len(velocities)
    temperature = 2.0 * average_kinectic_energy / velocities.shape[1]

    return temperature


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
    PARSER.add_argument(
        "--n",
        type=int,
        default=16,
        help="Number of particles")
    PARSER.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Time step size")
    PARSER.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Simulation time")
    PARSER.add_argument(
        "--width",
        type=float,
        default=4.0,
        help="Box width")
    PARSER.add_argument(
        "--height",
        type=float,
        default=4.0,
        help="Box height")
    PARSER.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Lennard-Jones 14/8 cutoff range.")
    PARSER.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Lennard-Jones 14/8 energy minimum.")
    PARSER.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Lennard-Jones 14/8 distance to zero-crossing point.")
    PARSER.add_argument(
        "--max_vel",
        type=float,
        default=4.0,
        help="Maximum velocity")
    PARSER.add_argument(
        "--max_temperature",
        type=float,
        default=8.0,
        help="Maximum temperature")
    ARGS = PARSER.parse_args()

    N = ARGS.n
    DT = ARGS.dt
    T = ARGS.t
    BOUNDARIES = [ARGS.width, ARGS.height]
    CUTOFF = ARGS.cutoff
    LJ_EPSILON = ARGS.epsilon
    LJ_SIGMA = ARGS.sigma
    MAX_VELOCITY = ARGS.max_vel
    MAX_TEMPERATURE = ARGS.max_temperature

    # Initialize arrays --------------------------------------------------------

    position = np.zeros((N, 2))
    velocity = np.zeros((N, 2))
    acceleration = np.zeros((N, 2))

    # Set up initial positions in a grid.
    x_positions = np.linspace(
        0.25,
        BOUNDARIES[0] - 0.25,
        int(np.sqrt(N)),
        endpoint=True
    )
    y_positions = np.linspace(
        0.25,
        BOUNDARIES[1] - 0.25,
        int(np.sqrt(N)),
        endpoint=True
    )

    x, y = np.meshgrid(x_positions, y_positions)
    position[:, 0] = x.flatten()
    position[:, 1] = y.flatten()

    # Randomize initial velocities.
    velocity = np.random.uniform(low=-10.0, high=10.0, size=(N, 2))

    # Simulation ---------------------------------------------------------------

    TIME_STEPS = int(T / DT)

    timestamps = np.zeros(TIME_STEPS)
    positions = np.zeros((TIME_STEPS, N, 2))
    velocities = np.zeros((TIME_STEPS, N, 2))
    accelerations = np.zeros((TIME_STEPS, N, 2))
    temperatures = np.zeros(TIME_STEPS)

    for i in range(TIME_STEPS):

        timestamps[i] = i * DT

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

        # Add pseudo-particles on each boundary to generate periodic forces.
        pseudo_particles = generate_boundary_pseudo_particles(
              position,
              BOUNDARIES,
              CUTOFF)
        position = np.concatenate((position, pseudo_particles), axis=0)

        # Compute temperature of the system.
        temperatures[i] = compute_temperature(velocity)

        # Rescale velocity to temperature limit and take half step.
        chi = np.sqrt(MAX_TEMPERATURE / temperatures[i])
        for p in range(N):
          velocity[p] = chi * velocity[p] + 0.5 * DT * acceleration[p]

        # Compute accelerations of the current time step.
        for p in range(N):

            acceleration[p] = 0.0
            current_position = position[p]

            for j in range(len(position)):

                distances = current_position - position[j]
                acceleration[p] += compute_forces(
                    distances, LJ_EPSILON, LJ_SIGMA, CUTOFF)

        # Remove pseudo-particles.
        position = position[:N]

        # Complete velocity calculation for another half time step.
        for p in range(N):
            velocity[p] = velocity[p] + 0.5 * DT * acceleration[p]

            # Old velocity calculation.
            #velocity[p] = (velocities[i, p]
            #               + (accelerations[i, p] + acceleration[p]) * 0.5 * DT)

        # Keep velocities within the assumed limits.
        velocity = np.clip(velocity, -MAX_VELOCITY, MAX_VELOCITY)

        # Compute final temperature of the system at the current time step.
        temperatures[i] = compute_temperature(velocity)

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
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.set_xlim([0.0 - 0.5, BOUNDARIES[0] + 0.5])
    ax1.set_ylim([0.0 - 0.5, BOUNDARIES[1] + 0.5])
    ax1.set_title("Particles", pad=16)
    ax1.set_ylabel(r"$y$", labelpad=16)
    ax1.set_xlabel(r"$x$", labelpad=16)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Temperature Evolution", pad=16)
    ax2.set_xlim([0.0, T])
    ax2.set_ylim([0.0, np.max(temperatures) + np.max(temperatures) * 0.1])
    ax2.set_ylabel(r"Temperature $[K]$", labelpad=16)
    ax2.set_xlabel(r"Time $[s]$", labelpad=16)

    # Particle plot.
    sc1 = ax1.scatter([], [])

    # Plot boundaries.
    rect = plt.Rectangle((0, 0),
                     BOUNDARIES[0],
                     BOUNDARIES[1],
                     ec='black', lw=2, fc='none')
    ax1.add_patch(rect)

    # Plot velocity quivers.
    velocity_quivers = []
    for i in range(N):
        quiver = ax1.quiver(
            positions[0, i, 0],
            positions[0, i, 1],
            velocities[0, i, 0],
            velocities[0, i, 1],
            color='r',
            units="width",
            width=0.0025)

        velocity_quivers.append(quiver)

    # Plot temperature evolution.
    sc2, = ax2.plot([], [])

    # Plot maximum temperature.
    ax2.axhline(MAX_TEMPERATURE, color='r')

    # Animation ----------------------------------------------------------------

    def animate(step):
        """ Function for the matplotlib anim. routine to update the plots. """

        fig.suptitle(
            "Time step = "
            + str(step)
            + "("
            + "{:.2E}".format(timestamps[step])
            + " $[s]$)"
        )

        # Update positions.
        sc1.set_offsets(positions[step, :])

        # Update velocities.
        for i in range(N):
            velocity_quivers[i].set_offsets(positions[step, i, :])
            velocity_quivers[i].set_UVC(
                velocities[step, i, 0],
                velocities[step, i, 1]
            )

        # Update temperature evolution.
        sc2.set_data([timestamps[:step], temperatures[:step]])

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=TIME_STEPS,
        repeat=False
    )
    plt.show()

    # Write video output of the animation --------------------------------------

    writer = animation.writers['ffmpeg']
    writer = writer(fps=15, metadata=dict(artist="Albert Garcia"), bitrate=1800)
    anim.save("md.mp4", writer=writer)