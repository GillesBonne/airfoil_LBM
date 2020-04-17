import numba
import numpy as np

import obstacles


def get_sides_mask(domain, top=True, bottom=True):
    mask = np.zeros(domain.shape)
    if bottom:
        mask[:, -1] = True  # Bottom side
    if top:
        mask[:, 0] = True  # Top side
    return mask


def get_inlet_mask(domain):
    mask = np.zeros(domain.shape)
    mask[0:2, :] = True  # Inlet
    return mask


def get_outlet_mask(domain):
    mask = np.zeros(domain.shape)
    mask[-1, :] = True  # Inlet
    return mask


def get_boundary_mask(domain, inlet=True, outlet=True, top=True, bottom=True):
    return (get_sides_mask(domain, top=top, bottom=bottom)
            + get_inlet_mask(domain) * inlet
            + get_outlet_mask(domain) * outlet) > 0


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


@numba.jit
def bounce_back_simple(field, field_prev, mask, x_mask, x_minus_ck_mask, q, opp):
    """
    Ordinary bounce-back scheme (Half-way BB)
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
    # Loop over all True nodes in x_mask (i.e. nodes that would take a node from the liquid domain to the solid domain
    for i, x, y in zip(*np.nonzero(x_mask)):
        field[opp[i], x, y] = field_prev[i, x, y]

    return field


def prepare_bounceback_interpolated(e, opp, shape: obstacles.Shape,
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
