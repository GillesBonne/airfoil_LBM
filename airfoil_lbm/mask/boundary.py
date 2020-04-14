import numpy as np


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
