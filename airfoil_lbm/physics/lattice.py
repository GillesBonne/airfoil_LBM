import numpy as np


def opp(e):
    """
    Returns the indices of vectors (e_x_i, e_y_i) that reverse its direction
    :param e: a DxQ numpy array of velocity vectors
    :return: a list of indices
    """
    opp = []
    for i in range(e.shape[0]):
        opp.append(np.where(np.all(e == -e[i, :], axis=1))[0][0])
    return opp


def D2Q9():
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


def get_kernels(ex, ey):
    Q = ex.size
    D = 1+2*ex.max()
    center_ind = ex.max()

    kernels = np.zeros((Q, D, D), dtype=np.int8)
    for i in range(Q):
        if ex[i] == 0 and ey[i] == 0:
            continue
        else:
            kernels[i, center_ind+ex[i], center_ind+ey[i]] = 1
    return kernels
