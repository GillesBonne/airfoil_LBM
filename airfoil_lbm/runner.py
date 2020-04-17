from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numba

import physics
import mask.boundary
import mask.obstacles
import visualization

# Flow constants
maxIter = 100000  # amount of cycles
Re = 200  # Reynolds number
Nx = 700  # Lattice points in x-direction
Ny = 200  # Lattice points in y-direction

# Get lattice parameters
lattice_configuration = physics.lattice.D2Q9
q, e, opp, ex, ey, w = lattice_configuration()

# Obstacle information

obstacle_r = Ny  # radius of the cylinder
my_domain_params = {'x_size': obstacle_r,
                    'y_size': obstacle_r,
                    'x_center': 0.2,
                    'y_center': 0.5}

tau = 1  # relaxation parameter
U_inf = physics.lattice.calculate_u_inf(L=1 / 2 * obstacle_r, Re=Re, tau=tau)
nu = (1.0 / 3.0) * (tau - 0.5)  # kinematic viscosity
omega = tau ** -1

print(f"# Parameters")
print(f"tau = {tau:.2f}")
print(f"omega = {omega:.2f}")
print(f"Re = {Re:.2f}")
print(f"U_inf = {U_inf:.2f}")
print(f"nu = {nu:.2f}")
print("\n")


def show(f, i=5):
    if len(f.shape) == 3:
        f = f[i]
        print(f"(ex, ey) = ({ex[i]},{ey[i]})")
    plt.matshow(f.T, origin='lower')
    plt.colorbar()
    plt.show()


periodic_x = False
periodic_y = False
periodic = periodic_x or periodic_y

dims = np.array([Nx, Ny]) + np.array([2 * periodic_x, 2 * periodic_y])

mask_boundary = mask.boundary.get_boundary_mask(
    np.zeros(dims), inlet=True, outlet=False, top=True, bottom=True)

# shape = mask.obstacles.AirfoilNaca00xx(angle=20, thickness=0.2)
shape = mask.obstacles.Circle(size_fraction=1/5)
# shape = mask.obstacles.Square(size_fraction=1/5)
mask_object = shape.place_on_domain(np.zeros(dims, dtype=np.bool), **my_domain_params)

subdomain = shape.get_subdomain_from_domain(np.zeros(dims), **my_domain_params)

x_mask, x_minus_ck_mask, q_mask = physics.boundary.prepare_bounceback_interpolated(e, opp, shape, subdomain)
x_mask, x_minus_ck_mask, q_mask = [shape.fill_domain_from_subdomain(a, [q, *dims], **my_domain_params)
                                   for a in (x_mask, x_minus_ck_mask, q_mask)]

plt.matshow(mask_object.T)
plt.title("Object mask")
plt.show()

mask_obstacle = mask_object


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

    feq = physics.lbm.equilibrium(rho, ux, uy, ex, ey, w)
    if periodic:
        feq = physics.boundary.apply_periodic_boundary(feq)

    # rho, ux, uy = calculate_macros(feq)

    return feq, rho, ux, uy


def test_fields(f, feq, rho, ux, uy):
    u = np.sqrt(ux ** 2 + uy ** 2)

    fields = [(rho, "rho"), (rho * u, "p"), (u, "u"), (ux, "ux"), (uy, "uy")]
    print("Testing values inside the mask")
    for (field, name) in fields:
        print((f"{name:>3s}_max = {field[mask_obstacle].max():>12.8f}, "
               f"{name:>3s}_min = {field[mask_obstacle].min():>12.8f}"))

    print("Testing values outside the mask")
    for (field, name) in fields:
        print((f"{name:>3s}_max = {field[~mask_obstacle].max():>12.8f}, "
               f"{name:>3s}_min = {field[~mask_obstacle].min():>12.8f}"))


def main(Nt=10000, tsave=10, debug=True):

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
        # f = physics.boundary.bounce_back(f, mask_obstacle, np.array(opp))

        # Do streaming: calculating the densities and velocities after a timestep dt
        # Store them into fp (fprime, f')
        for k in range(q):
            fp[k, :, :] = np.roll(f[k, :, :], (ey[k], ex[k]), axis=(1, 0))
        if periodic:
            fp = physics.boundary.apply_periodic_boundary(fp, left_right=periodic_x, top_bottom=periodic_y)

        # fp = physics.boundary.bounce_back(fp, mask_obstacle, opp)
        # fp = physics.boundary.bounce_back_interpolated(fp, f, mask_obstacle, x_mask, x_minus_ck_mask, q_mask, opp)
        fp = physics.boundary.bounce_back2(fp, f, mask_obstacle, x_mask, x_minus_ck_mask, q_mask, opp)

        # Calculate macros
        rho, ux, uy = physics.lbm.calculate_macros(fp, ex, ey)  # Set velocities within the obstacle to zero
        physics.boundary.set_boundary_macro(mask_obstacle, (rho, ux, uy), (0, 0, 0))
        # physics.boundary.set_boundary_macro(mask_boundary, (ux, uy), (U_inf, 0))
        physics.boundary.set_boundary_macro(mask_boundary, (ux, rho), (U_inf, 1))

        # Calculate feq
        feq = physics.lbm.equilibrium(rho, ux, uy, ex, ey, w)
        # fp[:, mask_obstacle] = 0
        # feq[:, mask_obstacle] = 0

        # Do collision
        f = fp + -(fp - feq) / tau

        if t % tsave == 0:
            test_fields(f, feq, rho, ux, uy)
            it = t // tsave
            ts[it] = t
            uxmax[it] = np.max(rho)
            m[it] = rho.sum()

            # Save velocity profile as an image
            visualization.show_field(ux, mask=mask_obstacle, title=f"t = {t:d}")
            # visualization.save_streamlines_as_image(ux, uy, v=np.sqrt(ux ** 2 + uy ** 2), mask=mask_obstacle,
            #                                         filename=f"vel/{t // tsave:08d}")
            # visualization._show_streamlines(ux, uy, v=np.sqrt(ux ** 2 + uy ** 2), mask=mask_obstacle)
            # visualization._show_streamlines(ux, uy, v=rho, mask=mask_obstacle)
            plt.show()
            # visualization.save_field_as_image(uy, mask=mask_obstacle, filename=f"velx/{t//tsave:08d}")
            # visualization.save_field_as_image(uy, filename=f"vely/{t:d}")

        # Make sure mass is conserved at every timestep. Due to floating point (in)accuracy, we do need to round
        # the mass at each node.
        # assert rho.round().sum() == els, f"Time = {t}, {rho.sum()}=/={els}"
    # visualization.save_field_as_image(ux)
    visualization.plot_2d(uxmax)


if __name__ == "__main__":
    main()
