from typing import Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np

import boundary
import lattice
import lbm
import obstacles
import visualization


def get_initial_conditions(dims, U_inf, ex, ey, w, periodic, mask_matrix=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    feq = lbm.equilibrium(rho, ux, uy, ex, ey, w)
    if periodic:
        feq = boundary.apply_periodic_boundary(feq)

    # rho, ux, uy = calculate_macros(feq)

    return feq, rho, ux, uy


def test_fields(f, feq, rho, ux, uy, mask_obstacle):
    u = np.sqrt(ux ** 2 + uy ** 2)

    fields = [(rho, "rho"), (rho*u, "p"), (u, "u"), (ux, "ux"), (uy, "uy")]

    if np.any(mask_obstacle):
        print("Testing values inside the mask")
        for (field, name) in fields:
            print((f"{name:>3s}_max = {field[mask_obstacle].max():>12.8f}, "
                   f"{name:>3s}_min = {field[mask_obstacle].min():>12.8f}"))

    print("Testing values outside the mask")
    for (field, name) in fields:
        print((f"{name:>3s}_max = {field[~mask_obstacle].max():>12.8f}, "
               f"{name:>3s}_min = {field[~mask_obstacle].min():>12.8f}"))


def run(Nt, tsave, debug, Re, Nx, Ny, tau, periodic_x, periodic_y, simple_bounce, interp_bounce, circle, airfoil, angle=None, thickness=None):
    if circle and naca:
        raise ValueError('Choose either a circle or airfoil obstacle.')
    elif simple_bounce and interp_bounce:
        raise ValueError(
            'Choose either the simple bounce back scheme or using interpolated boundaries.')

    # Get lattice parameters
    lattice_configuration = lattice.D2Q9
    q, e, opp, ex, ey, w = lattice_configuration()

    # Obstacle information
    obstacle_r = Ny // 4  # radius of the cylinder
    my_domain_params = {'x_size': obstacle_r,
                        'y_size': obstacle_r,
                        'x_center': 0.2,
                        'y_center': 0.5}

    U_inf = lattice.calculate_u_inf(L=1 / 2 * obstacle_r, Re=Re, tau=tau)

    # kinematic viscosity
    nu = (1.0 / 3.0) * (tau - 0.5)
    omega = tau ** -1

    print(f"# Parameters")
    print(f"tau = {tau:.2f}")
    print(f"omega = {omega:.2f}")
    print(f"Re = {Re:.2f}")
    print(f"U_inf = {U_inf:.2f}")
    print(f"nu = {nu:.2f}")
    print("\n")

    periodic = periodic_x or periodic_y

    dims = np.array([Nx, Ny]) + np.array([2 * periodic_x, 2 * periodic_y])

    mask_boundary = boundary.get_boundary_mask(
        np.zeros(dims), inlet=True, outlet=False, top=True, bottom=True)

    if circle:
        shape = obstacles.Circle(size_fraction=0.9)
    elif airfoil:
        if angle is not None and thickness is not None:
            shape = obstacles.AirfoilNaca00xx(angle=angle, thickness=thickness)
        else:
            raise ValueError('Naca airfoil properties not specified.')

    if circle or airfoil:
        mask_obstacle = shape.place_on_domain(np.zeros(dims, dtype=np.bool), **my_domain_params)
        subdomain = shape.get_subdomain_from_domain(np.zeros(dims), **my_domain_params)

        if interp_bounce:
            x_mask, x_minus_ck_mask, q_mask = boundary.prepare_bounceback_interpolated(
                e, opp, shape, subdomain)
            x_mask, x_minus_ck_mask, q_mask = [shape.fill_domain_from_subdomain(a, [q, *dims], **my_domain_params)
                                               for a in (x_mask, x_minus_ck_mask, q_mask)]
    else:
        mask_obstacle = np.zeros(dims, dtype=bool)

    plt.matshow(mask_obstacle.T, origin='lower')
    plt.title("Obstacle mask")
    plt.show()

    # Initialize PDF
    feq, rho, ux, uy = get_initial_conditions(dims, U_inf, ex, ey, w, periodic, mask_obstacle)

    test_fields(feq, feq, rho, ux, uy, mask_obstacle)

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
        # f = boundary.bounce_back(f, mask_obstacle, np.array(opp))

        # Do streaming: calculating the densities and velocities after a timestep dt
        # Store them into fp (fprime, f')
        for k in range(q):
            fp[k, :, :] = np.roll(f[k, :, :], (ey[k], ex[k]), axis=(1, 0))
        if periodic:
            fp = boundary.apply_periodic_boundary(
                fp, left_right=periodic_x, top_bottom=periodic_y)

        if circle or airfoil:
            if simple_bounce:
                fp = boundary.bounce_back(fp, mask_obstacle, opp)
            elif interp_bounce:
                fp = boundary.bounce_back_interpolated(
                    fp, f, mask_obstacle, x_mask, x_minus_ck_mask, q_mask, opp)
            else:
                raise ValueError('Choose which bounce back scheme to use.')

        # Calculate macros
        rho, ux, uy = lbm.calculate_macros(fp, ex, ey)

        # Force calculation.
        if t % tsave == 0:
            if simple_bounce:
                fx = 2*ux[mask_obstacle].sum()
                fy = 2*uy[mask_obstacle].sum()
            else:
                fx = None
                fy = None
            print('fx: ', fx)
            print('fy: ', fy)

        # Set velocities within the obstacle to zero
        boundary.set_boundary_macro(mask_obstacle, (rho, ux, uy), (0, 0, 0))
        # boundary.set_boundary_macro(mask_boundary, (ux, uy), (U_inf, 0))
        boundary.set_boundary_macro(mask_boundary, (ux,), (U_inf,))

        # Calculate feq
        feq = lbm.equilibrium(rho, ux, uy, ex, ey, w)
        fp[:, mask_obstacle] = 0
        feq[:, mask_obstacle] = 0

        # Do collision
        f = fp + -(fp - feq) / tau

        if t % tsave == 0:
            test_fields(f, feq, rho, ux, uy, mask_obstacle)
            it = t // tsave
            ts[it] = t
            uxmax[it] = np.max(ux)
            m[it] = rho.sum()

            # Save velocity profile as an image
            # visualization.show_field(ux, mask=mask_obstacle, title=f"velx/{t:d}")
            # visualization.save_streamlines_as_image(ux, uy, v=np.sqrt(ux ** 2 + uy ** 2), mask=mask_obstacle,
            #                                         filename=f"vel/{t // tsave:08d}")
            visualization._show_streamlines(ux, uy, v=np.sqrt(
                ux ** 2 + uy ** 2), mask=mask_obstacle)
            plt.show()
            # visualization.save_field_as_image(ux, mask=mask_obstacle, filename=f"velx/{t//tsave:08d}")
            # visualization.save_field_as_image(uy, filename=f"vely/{t:d}")

        # Make sure mass is conserved at every timestep. Due to floating point (in)accuracy, we do need to round
        # the mass at each node.
        # assert rho.round().sum() == els, f"Time = {t}, {rho.sum()}=/={els}"
    # visualization.save_field_as_image(ux)
    visualization.plot_2d(uxmax)


if __name__ == "__main__":
    Nt = 1_000_000
    tsave = 20
    debug = True
    Re = 20  # Reynolds number
    Nx = 700  # Lattice points in x-direction
    Ny = 200  # Lattice points in y-direction
    tau = 10  # relaxation parameter
    periodic_x = False
    periodic_y = False
    run(Nt, tsave, debug, Re, Nx, Ny, tau, periodic_x, periodic_y,
        simple_bounce=True, interp_bounce=False,
        circle=False, airfoil=True, angle=10, thickness=0.2)
