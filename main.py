import numpy as np
import boundary
import boundary.obstacles
import visualization
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

# Flow constants
maxIter = 100000  # amount of cycles
Re = 220  # Reynolds number
Nx = 200  # Lattice points in x-direction
Ny = 50  # Lattice points in y-direction
q = 9  # number of possible directions
U = 0.04  # maximum velocity of Poiseuille flow
U_inf = 10  # velocity at a distance far away from the airfoil such that the airfoil does not disturb the velocity there
obstacle_x = Nx / 4  # x location of the cylinder
obstacle_y = Ny / 2  # y location of the cylinder
obstacle_r = Ny / 9  # radius of the cylinder
tau = 1
nu = (1.0 / 3.0) * (tau - 0.5)
# nu = U * obstacle_r / Re  # kinematic viscosity
# tau = 3. * (nu + 0.5)  # relaxation parameter
omega = tau ** -1

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

dims = np.array([Nx, Ny]) + periodic * 2
els = dims[0] * dims[1]

# Make sure we didn't make any typing errors
assert ex.sum() == 0
assert ey.sum() == 0
assert w.sum() == 1

mask_boundary = boundary.get_boundary_mask(np.zeros(dims), inlet=False, outlet=False, top=True, bottom=True)
mask_circle = boundary.obstacles.circle(np.zeros(dims))

plt.matshow(mask_circle)
plt.show()

mask_obstacle = mask_boundary


def equilibrium(rho, ux, uy):
    # Calculate feq
    evel = np.multiply.outer(ex, ux) + np.multiply.outer(ey, uy)
    feq = np.multiply.outer(w, rho) * (1 + 3 * evel + 9 / 2 * evel ** 2 - 3 / 2 * (ux ** 2 + uy ** 2))
    return feq


def get_initial_conditions(mask=None):
    ux = np.zeros(dims)
    uy = np.zeros(dims)

    u0 = 1
    rho = np.ones(ux.shape)

    # Start with a uniform initial velocity distribution
    ux = np.broadcast_to(u0, dims).copy()

    if mask is not None:
        ux[mask] = 0
        uy[mask] = 0

    feq = equilibrium(rho, ux, uy)
    if periodic:
        feq = boundary.apply_periodic_boundary(feq)

    return feq, rho, ux, uy


def main(Nt=1000, tsave=100, debug=True):
    # Initialize PDF
    feq, rho, ux, uy = get_initial_conditions(mask_obstacle)
    f = feq.copy()
    fp = feq.copy()
    visualization.show_field(ux)

    ts = np.arange(start=0, stop=Nt, step=tsave)
    m = np.zeros(ts.shape)
    uxmax = np.zeros(ts.shape)

    if debug:
        print(f"Doing {Nt} timesteps")

    for t in range(Nt):
        if debug and t % tsave == 0:
            print(f"t = {t}")

        # Bounce back
        f2 = f.copy()
        for k in range(q):
            f[k, mask_obstacle] = f2[opp[k], mask_obstacle]

        # Do streaming: calculating the densities and velocities after a timestep dt
        # Store them into fp (fprime, f')
        for k in range(q):
            fp[k, :, :] = np.roll(np.roll(f[k, :, :], ey[k], axis=1), ex[k], axis=0)
        if periodic:
            fp = boundary.apply_periodic_boundary(fp)

        # Calculate rho. This is the sum over the contents of each q (velocity vector)
        rho = fp.sum(axis=0)

        zeros = np.count_nonzero(fp < 0)
        if zeros > 0:
            print(f"Encountered {zeros} 0's at t={t}")

        # The velocity is the mass-averaged velocity for each direction
        # Ux is periodic if fp is also periodic, so we don't have to apply pbc
        ux = (ex[:, None, None] * fp).sum(axis=0) / rho
        uy = (ey[:, None, None] * fp).sum(axis=0) / rho

        # Increment velocities in x-direction to mimic a constant pressure drop
        ux += 0.0001
        # ux[mask_boundary] = U_inf
        # uy[mask_boundary] = 0

        # Calculate feq
        feq = equilibrium(rho, ux, uy)

        # Do collision
        f = omega * feq + (1.0 - omega) * fp

        if t % tsave == 0:
            it = t // tsave
            ts[it] = t
            uxmax[it] = np.max(ux)
            m[it] = rho.sum()
            # visualization.plot_crossection(ux)
            visualization.show_field(ux)

        # Make sure mass is conserved at every timestep. Due to floating point (in)accuracy, we do need to round
        # the mass at each node.
        # assert rho.round().sum() == els, f"Time = {t}, {rho.sum()}=/={els}"
    # visualization.save_field_as_image(ux)
    visualization.plot_2d(uxmax)


if __name__ == "__main__":
    main()
