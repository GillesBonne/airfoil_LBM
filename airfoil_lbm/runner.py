from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numba

import mask.boundary
import mask.obstacles
import visualization

# Flow constants
maxIter = 100000  # amount of cycles
Re = 120  # Reynolds number
Nx = 700  # Lattice points in x-direction
Ny = 200  # Lattice points in y-direction
q = 9  # number of possible directions
U = 0.04  # maximum velocity of Poiseuille flow
U_inf = 0.09  # velocity at a distance far away from the airfoil such that the airfoil does not disturb the velocity there
obstacle_x = Nx / 4  # x location of the cylinder
obstacle_y = Ny / 2  # y location of the cylinder
obstacle_r = Ny / 9  # radius of the cylinder
# tau = 3
# nu = (1.0 / 3.0) * (tau - 0.5)
nu = U_inf * obstacle_r / Re  # kinematic viscosity
tau = 3. * (nu + 0.5)  # relaxation parameter
omega = tau ** -1

print(f"# Parameters")
print(f"tau = {tau:.2f}")
print(f"omega = {omega:.2f}")
print(f"Re = {Re:.2f}")
print(f"U_inf = {U_inf:.2f}")
print("\n")

periodic_x = False
periodic_y = False
periodic = periodic_x or periodic_y

ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]) * 1.0

# Calculate the opposite vectors
opp = []
e = np.array([ex, ey]).T
for i in range(ex.size):
    opp.append(np.where(np.all(e == -e[i, :], axis=1))[0][0])

dims = np.array([Nx, Ny]) + np.array([2 * periodic_x, 2 * periodic_y])
els = dims[0] * dims[1]

# Make sure we didn't make any typing errors
assert ex.sum() == 0
assert ey.sum() == 0
assert w.sum() == 1

mask_boundary = mask.boundary.get_boundary_mask(
    np.zeros(dims), inlet=True, outlet=False, top=True, bottom=True)

# AFOIL = mask.obstacles.Naca00xx(airfoil_size=200, angle=-45, thickness=0.1)
# mask_object = AFOIL.box
#
mask_object = mask.obstacles.circle(np.zeros(dims))

plt.matshow(mask_object)
plt.show()

mask_obstacle = mask_object


def equilibrium(rho, ux, uy) -> np.ndarray:
    # Calculate feq
    evel = np.multiply.outer(ex, ux) + np.multiply.outer(ey, uy)
    feq = np.multiply.outer(w, rho) * (1 + 3 * evel + 9 / 2 *
                                       evel ** 2 - 3 / 2 * (ux ** 2 + uy ** 2))
    return feq


def get_initial_conditions(mask_matrix=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ux = np.zeros(dims)
    uy = np.zeros(dims)

    u0 = U_inf
    rho = np.ones(ux.shape)

    # Start with a uniform initial velocity distribution
    ux = np.broadcast_to(u0, dims).copy()

    if mask_matrix is not None:
        # Set velocity to zero within the boundary
        ux[mask_matrix] = 0
        uy[mask_matrix] = 0

    feq = equilibrium(rho, ux, uy)
    if periodic:
        feq = mask.boundary.apply_periodic_boundary(feq)

    # rho, ux, uy = calculate_macros(feq)

    return feq, rho, ux, uy


def test_fields(f, feq, rho, ux, uy):
    u = np.sqrt(ux ** 2 + uy ** 2)
    # print("Testing values inside the boundary")
    #
    # for (field, name) in [(u, "u"), (ux, "ux"), (uy, "uy")]:
    #     print(f"{name}_max = {field[mask_object].max()}, {name}_min = {field[mask_object].min()}")

    print("Testing values outside the boundary")

    for (field, name) in [(u, "u"), (ux, "ux"), (uy, "uy")]:
        print((f"{name:>3s}_max = {field[~mask_object].max():>12.8f}, "
               f"{name:>3s}_min = {field[~mask_object].min():>12.8f}"))


def calculate_macros(f) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Set velocities within the obstacle to zero
    ux[mask_obstacle] = 0
    uy[mask_obstacle] = 0

    # Set velocity at the
    ux[mask_boundary] = U_inf
    uy[mask_boundary] = 0
    return rho, ux, uy


def main(Nt=1_000_000, tsave=10, debug=True):
    # Initialize PDF
    feq, rho, ux, uy = get_initial_conditions(mask_obstacle)

    test_fields(feq, feq, rho, ux, uy)

    f = feq.copy()
    fp = feq.copy()

    visualization.show_field(ux, title=f"$u_x(t = {0})$")

    ts = np.arange(start=0, stop=Nt, step=tsave)
    m = np.zeros(ts.shape)
    uxmax = np.zeros(ts.shape)

    if debug:
        print(f"Doing {Nt} timesteps")

    for t in range(Nt):
        if debug and t % tsave == 0:
            print(f"\n# t = {t}")

        # Bounce back
        f2 = f.copy()
        for k in range(q):
            f[k, mask_obstacle] = f2[opp[k], mask_obstacle]

        # Do streaming: calculating the densities and velocities after a timestep dt
        # Store them into fp (fprime, f')
        for k in range(q):
            fp[k, :, :] = np.roll(np.roll(f[k, :, :], ey[k], axis=1), ex[k], axis=0)
        if periodic:
            fp = mask.boundary.apply_periodic_boundary(fp, left_right=periodic_x, top_bottom=periodic_y)

        # Calculate macros
        rho, ux, uy = calculate_macros(fp)

        # Calculate feq
        feq = equilibrium(rho, ux, uy)

        # Do collision
        f = omega * feq + (1.0 - omega) * fp

        if t % tsave == 0:
            test_fields(f, feq, rho, ux, uy)
            it = t // tsave
            ts[it] = t
            uxmax[it] = np.max(ux)
            m[it] = rho.sum()

            # Save velocity profile as an image
            # visualization.show_field(ux, mask=mask_obstacle, title=f"velx/{t:d}")
            visualization.save_streamlines_as_image(ux, uy, v=np.sqrt(ux ** 2 + uy ** 2), mask=mask_obstacle,
                                                    filename=f"vel/{t // tsave:08d}")
            # visualization._show_streamlines(ux, uy, v=np.sqrt(ux ** 2 + uy ** 2), mask=mask_obstacle)
            # visualization.save_field_as_image(ux, mask=mask_obstacle, filename=f"velx/{t//tsave:08d}")
            # visualization.save_field_as_image(uy, filename=f"vely/{t:d}")

        # Make sure mass is conserved at every timestep. Due to floating point (in)accuracy, we do need to round
        # the mass at each node.
        # assert rho.round().sum() == els, f"Time = {t}, {rho.sum()}=/={els}"
    # visualization.save_field_as_image(ux)
    visualization.plot_2d(uxmax)


if __name__ == "__main__":
    main()
