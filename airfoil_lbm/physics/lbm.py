import numpy as np
import numba

import typing


# TODO: New name for this file

@numba.jit(nopython=True, cache=True)
def calculate_macros(f, ex, ey) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the macroscopic variables (density, velocity_x, velocity_y) for the given field.
    :param f: The field at a given timestep
    :param e: [ex, ey], velocity vectors
    :return: rho, ux, uy
    """
    # Calculate rho. This is the sum over the contents of each q (velocity vector)
    rho = f.sum(axis=0)

    # The velocity is the mass-averaged velocity for each direction
    # Ux is periodic if fp is also periodic, so we don't have to apply pbc
    ux = np.zeros(rho.shape)
    uy = np.zeros(rho.shape)

    exi = ex.reshape((ex.size, 1, 1))
    eyi = ey.reshape((ex.size, 1, 1))

    fx = (exi * f).sum(axis=0)
    fy = (eyi * f).sum(axis=0)

    for i, j in zip(*np.nonzero(rho)):
        ux[i, j] = fx[i, j] / rho[i, j]
        uy[i, j] = fy[i, j] / rho[i, j]

    # Increment velocities in x-direction to mimic a constant pressure drop
    # ux[rho > 0] += 0.001 / rho[rho > 0]
    return rho, ux, uy


@numba.jit(nopython=True, cache=True)
def equilibrium(rho, ux, uy, ex, ey, w) -> np.ndarray:
    # Calculate feq
    result = np.zeros((ex.size, *ux.shape))
    for i in range(ex.size):
        result[i, :, :] = ex[i] * ux + ey[i] * uy
        result[i, :, :] = (1 + 3 * result[i, :, :] + 9 / 2 * result[i, :, :] ** 2 - 3 / 2 * (ux ** 2 + uy ** 2))
        result[i, :, :] = w[i] * rho * result[i, :, :]
    # evel = np.multiply.outer(ex, ux) + np.multiply.outer(ey, uy)
    # feq = np.multiply.outer(w, rho) * (1 + 3 * evel + 9 / 2 *
    #                                    evel ** 2 - 3 / 2 * (ux ** 2 + uy ** 2))
    return result