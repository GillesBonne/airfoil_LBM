import physics.boundary
import mask
import numpy as np
import matplotlib.pyplot as plt


def show(f, i=5):
    if len(f.shape) == 3:
        f = f[i]
        print(f"(ex, ey) = ({ex[i]},{ey[i]})")
    plt.matshow(f.T, origin='lower')
    plt.colorbar()
    plt.show()


Nx = 3  # Lattice points in x-direction
Ny = 3  # Lattice points in y-direction

# Get lattice parameters
lattice_configuration = physics.lattice.D2Q9
q, e, opp, ex, ey, w = lattice_configuration()

# Obstacle information

obstacle_r = Ny  # radius of the cylinder
my_domain_params = {'x_size': obstacle_r,
                    'y_size': obstacle_r,
                    'x_center': 0.5,
                    'y_center': 0.5}

dims = np.array([Nx, Ny])
tau = 1  # relaxation parameter
U_inf = physics.lattice.calculate_u_inf(L=1 / 2 * obstacle_r, Re=20, tau=tau)
nu = (1.0 / 3.0) * (tau - 0.5)  # kinematic viscosity
omega = tau ** -1
shape = mask.obstacles.Square(size_fraction=1 / 3)
mask_object = shape.place_on_domain(np.zeros(dims, dtype=np.bool), **my_domain_params)
subdomain = shape.get_subdomain_from_domain(np.zeros(dims), **my_domain_params)

fp = np.zeros((9, 3, 3))
f = np.zeros((9, 3, 3))

x_mask, x_minus_ck_mask, q_mask = physics.boundary.prepare_bounceback_interpolated(e, opp, shape, subdomain)

fp[5, 1, 1] = 1
show(fp)
f[5, 0, 0] = 1
show(f)
fp = physics.boundary.bounce_back2(fp, f, mask_object, x_mask, x_minus_ck_mask, q_mask, opp)
show(fp)
