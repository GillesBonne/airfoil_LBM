import numpy as np


def circle(domain: np.ndarray):
    Nx, Ny = domain.shape
    r = Ny/5
    x_center, y_center = (Ny, 0)
    x = np.linspace(0, Nx, Nx).reshape([Nx, 1])
    y = np.linspace(-Ny/2, Ny/2, Ny)
    mask = ((x-x_center)**2 + (y-y_center)**2 < r**2)
    return mask
