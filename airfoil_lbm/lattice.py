import numpy as np


def opp(e):
    """
    Returns the indices of vectors (e_x_i, e_y_i) that reverse its direction
    :param e: a DxQ numpy array of velocity vectors
    :return: a list of indices
    """
    opp = np.zeros(e.shape[0], dtype=int)
    for i in range(e.shape[0]):
        opp[i] = np.where(np.all(e == -e[i, :], axis=1))[0][0]
    return opp


def D2Q9():
    """
    2-Dimensional lattice configuration with 9 velocities. The velocities are directed from the center of a square to
    its edge centers and its corners.
    :return:
    """
    q = 9
    ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    e = np.array([ex, ey]).T
    e_opposite = opp(e)
    w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]) * 1.0

    # Make sure we didn't make any typing errors
    assert ex.sum() == 0
    assert ey.sum() == 0
    assert w.sum() == 1

    return q, e, e_opposite, ex, ey, w


def calculate_nu(L, u_inf, Re=120):
    nu_lb = u_inf * L / Re
    return nu_lb


def calculate_u_inf(L, Re, tau):
    nu = (tau - 1 / 2) / 3
    u = Re * nu / L
    return u
