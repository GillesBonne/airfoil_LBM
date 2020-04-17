import numba
import numpy as np

import mask.obstacles

import matplotlib.pyplot as plt

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])


def show(f, i=5):
    if len(f.shape) == 3:
        f = f[i]
        print(f"(ex, ey) = ({ex[i]},{ey[i]})")
    plt.matshow(f.T, origin='lower')
    plt.colorbar()
    plt.show()


@numba.jit(nopython=True, cache=True)
def apply_periodic_boundary(field, left_right=True, top_bottom=True):
    if len(field.shape) == 2:
        # [y, x]
        if left_right:
            field[:, 0] = field[:, -2]
            field[:, -1] = field[:, 1]
        if top_bottom:
            field[0, :] = field[-2, :]
            field[-1, :] = field[1, :]
    elif len(field.shape) == 3:
        # [q, y, x]
        if left_right:
            field[:, :, 0] = field[:, :, -2]
            field[:, :, -1] = field[:, :, 1]
        if top_bottom:
            field[:, 0, :] = field[:, -2, :]
            field[:, -1, :] = field[:, 1, :]
    return field


# @numba.jit(nopython=True, cache=True)
def set_boundary_macro(mask, field, value):
    for i_f, f in enumerate(field):
        # For non-numba, this is faster:
        f[mask] = value[i_f]

        # For numba (which doesn't support boolean masking), use this
        # x, y = np.nonzero(mask)
        # for i, j in zip(x, y):
        #     field[i_f][i, j] = value[i_f]


@numba.jit(nopython=True, cache=True)
def bounce_back(field, mask, opp):
    """
    Ordinary bounce-back implementation. Should be called before the propagation step.
    :param field:
    :param mask:
    :param opp:
    :return:
    """
    q = len(opp)
    # Bounce back
    f2 = field.copy()
    for k in range(q):
        x, y = np.nonzero(mask)
        for i, j in zip(x, y):
            field[k, i, j] = f2[opp[k], i, j]
    # field[:, mask] = f2[opp, mask]
    return field


# @numba.jit
def bounce_back2(field, field_prev, mask, x_mask, x_minus_ck_mask, q, opp):
    """
    Implementation of the bounce-back scheme described by Bouzidi et al, 2001.
    Note that the parameters x_mask, x_minus_ck_mask, q should be acquired by calling prepare_bounceback_interpolated
    in the initialization of the program. Should be called after the propagation step.
    :param field: the field at timestep t + dt
    :param field_prev: the field at timestep t
    :param mask: the obstacle mask
    :param x_mask: describes the points that would be taken into the obstacle mask by vector c_i
    :param x_minus_ck_mask: x_mask, but propagated backwards in time
    :param q: the ratio of the boundary position along c_i and |c_i|
    :param opp: an array of indices that reverses the direction of e_(x,y)
    :return: the corrected field
    # """
    # field_prev[5, 19, 19] = 0.03
    # field[x_mask] = field_prev[x_mask[opp]]
    # show(field_prev)
    # show(field)
    for i, x, y in zip(*np.nonzero(x_mask)):
        field[i, x, y] = field_prev[opp[i], x, y]
    # field[:, mask] = field[opp][:, mask]
    # field[x_mask[opp]] = field_prev[x_mask]
    # field[:, mask] = 0

    return field


def prepare_bounceback_interpolated(e, opp, shape: mask.obstacles.Shape,
                                    subdomain):
    """
    Preparation for bounce_back_interpolated
    :param e:
    :param opp:
    :param shape:
    :param subdomain:
    :param kernels:
    :return:
    """
    ex, ey = e.T

    x_mask = shape.directional_boundaries(subdomain, e, opp)

    # Get the q factors from the obstacle for this set of kernels
    q = shape.get_kernel_ratios(subdomain, e, x_mask)

    x_minus_ck_mask = np.zeros(x_mask.shape, dtype=np.bool)
    for k in range(ex.size):
        x_minus_ck_mask[k, :, :] = np.roll(x_mask[k, :, :], (-ey[k], -ex[k]), axis=(1, 0))

    return x_mask, x_minus_ck_mask, q


def bounce_back_interpolated(field, field_prev, mask, x_mask, x_minus_ck_mask, q, opp):
    """
    Implementation of the bounce-back scheme described by Bouzidi et al, 2001.
    Note that the parameters x_mask, x_minus_ck_mask, q should be acquired by calling prepare_bounceback_interpolated
    in the initialization of the program. Should be called after the propagation step.
    :param field: the field at timestep t + dt
    :param field_prev: the field at timestep t
    :param mask: the obstacle mask
    :param x_mask: describes the points that would be taken into the obstacle mask by vector c_i
    :param x_minus_ck_mask: x_mask, but propagated backwards in time
    :param q: the ratio of the boundary position along c_i and |c_i|
    :param opp: an array of indices that reverses the direction of e_(x,y)
    :return: the corrected field
    """
    # Get the q-ratio's
    q_masked = q[x_mask]

    # Bouzidi scheme, piecewise for q>1/2 and q<=1/2
    qplus = 2 * q_masked * field_prev[x_mask] \
            + (1 - 2 * q_masked) * field_prev[x_minus_ck_mask]
    qminus = 1 / (2 * q_masked) * field_prev[x_mask] \
             + (2 * q_masked - 1) / (2 * q_masked) * field_prev[x_mask[opp]]

    # For q_masked = 1/2, this should yield the same result as the ordinary bounce-back
    # q_masked[:] = 1/2
    field[opp][x_mask] = qplus * (q_masked < 1 / 2) \
                         + qminus * (q_masked >= 1 / 2)

    return field
