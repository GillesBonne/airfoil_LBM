import numpy as np


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
