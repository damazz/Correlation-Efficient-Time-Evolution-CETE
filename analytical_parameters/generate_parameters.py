

import numpy as np

from sparse_sim.sparse_sim import *
from sparse_sim.fermion.qiskit_wrapper import *

FILE_NAME = "parameters"


def generate_parameters():
    max_small_theta = 2
    theta1 = 3 * np.pi / 16
    theta2 = -1 * np.pi / 16
    d_iters1 = max(int(round(theta1 / max_small_theta)), 1)
    d_iters2 = max(int(round(theta2 / max_small_theta)), 1)
    small_theta1 = theta1 / d_iters1
    small_theta2 = theta2 / d_iters2
    d1 = (np.cos(theta2) - 1) / \
        (2*(np.cos(theta2) - np.cos(theta1)) * np.sin(theta1))
    d2 = (np.cos(theta1) - 1) / \
        (2*(np.cos(theta2) - np.cos(theta1)) * np.sin(theta2))

    parameters = np.array(
        [small_theta1, small_theta2, d1, d2, d_iters1, d_iters2], dtype=object)
    np.save(f'analytical_parameters/{FILE_NAME}.npy', parameters)


if __name__ == "__main__":
    generate_parameters()
    print("Parameters saved to 'analytical_parameters/parameters.npy'")
