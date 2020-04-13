import numpy as np
import typing


# TODO: New name for this file

def calculate_macros(f, e) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the macroscopic variables (density, velocity_x, velocity_y) for the given field.
    :param f: The field at a given timestep
    :param e: [ex, ey], velocity vectors
    :return: rho, ux, uy
    """
    ex, ey = e.T
    # Calculate rho. This is the sum over the contents of each q (velocity vector)
    rho = f.sum(axis=0)

    # The velocity is the mass-averaged velocity for each direction
    # Ux is periodic if fp is also periodic, so we don't have to apply pbc
    ux = np.zeros(rho.shape)
    uy = np.zeros(rho.shape)
    ux = np.divide((ex[:, None, None] * f).sum(axis=0), rho, out=ux, where=rho != 0)
    uy = np.divide((ey[:, None, None] * f).sum(axis=0), rho, out=uy, where=rho != 0)

    # Increment velocities in x-direction to mimic a constant pressure drop
    # ux[mask_obstacle == False] += 0.0007
    return rho, ux, uy
