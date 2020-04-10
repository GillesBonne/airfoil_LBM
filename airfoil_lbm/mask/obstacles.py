"""Module to create a boolean array mask with an obstacle inside.
"""

import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.spatial
import physics


def rotate_around_point(xy, radians, origin):
    """Rotate a point around a given point.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


class Shape:
    """
    Shape is a super-class for objects that can be placed in a simulation domain.
    """

    # The size of this shape can either be set as a fraction of the domain it is given,
    # or as an absolute value.
    size_fraction = None
    size = None

    def __init__(self, size_fraction=1., size=None):
        self.size = size
        if size is None:
            self.size_fraction = size_fraction

    def get_size(self, domain: np.ndarray):
        if self.size_fraction is not None:
            return min(domain.shape) * self.size_fraction
        else:
            return self.size

    @staticmethod
    def get_center_for_domain(domain: np.ndarray):
        """
        Calculates the center of a domain
        :param domain:
        :return:
        """
        return (np.array(domain.shape) - 1) / 2.

    def place_on_subdomain(self, domain: np.ndarray):
        """
        Places this shape in the center of the supplied (sub)domain
        :param domain: 2D numpy array of type np.bool
        :return: A copy of the domain, with this shape placed on it, centered horizontally and vertically
        """
        raise NotImplementedError("Classes inheriting from Shape should implement _place_on_domain")

    def place_on_domain(self, domain, x_size, y_size, center_x=0.5, center_y=0.5):
        center_x = int(domain.shape[0] * center_x)
        center_y = int(domain.shape[1] * center_y)

        y_start = center_y - y_size // 2
        y_end = y_start + y_size

        x_start = center_x - x_size // 2
        x_end = x_start + x_size

        subdomain = domain[x_start:x_end, y_start:y_end]
        domain[x_start:x_end, y_start:y_end] = self.place_on_subdomain(subdomain)
        return domain

    def get_distance_to_point(self, domain: np.ndarray, x):
        """
        Returns the closest distance to the edge of the shape, analytically.
        :param x: A tuple containing the coordinates
        :return: The closest distance to the shape
        """
        raise NotImplementedError("Classes inheriting from Shape should implement get_distance_to_point")

    def visualize(self, domain: np.ndarray = None):
        if domain is None:
            domain = np.zeros((200, 200), dtype=np.bool)
        domain = self.place_on_subdomain(domain)

        plt.imshow(domain.T, origin='lower')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def directional_boundaries(self, domain, kernels, opp):
        mask = self.place_on_subdomain(domain)

        directional_boundaries = np.zeros((kernels.shape[0], *domain.shape), dtype=int)
        for i, kernel in enumerate(kernels):
            boundary = scipy.ndimage.convolve(
                np.array(mask, dtype=int), kernel, mode='constant')
            boundary[mask] = 0
            directional_boundaries[opp[i], :, :] = boundary
        return directional_boundaries


class Circle(Shape):
    def get_distance_to_point(self, domain: np.ndarray, x):
        return np.linalg.norm(x - self.get_center_for_domain(domain)) - self.get_size(domain) / 2

    def place_on_subdomain(self, domain: np.ndarray):
        Nx, Ny = domain.shape
        r = self.get_size(domain) / 2.
        x_center, y_center = self.get_center_for_domain(domain)
        x = np.arange(Nx).reshape([Nx, 1])
        y = np.arange(Ny)
        mask = ((x - x_center) ** 2 + (y - y_center) ** 2 < r ** 2)
        return mask


class Square(Shape):
    def get_distance_to_point(self, domain: np.ndarray, x):
        return np.linalg.norm(x - self.get_center_for_domain(domain)) - self.get_size(domain) / 2

    def place_on_subdomain(self, domain: np.ndarray):
        Nx, Ny = domain.shape
        r = self.get_size(domain) / 2.
        x_center, y_center = self.get_center_for_domain(domain)
        x = np.arange(Nx).reshape([Nx, 1])
        y = np.arange(Ny)
        mask = (x > x_center - r) * (x < x_center + r) * (y > y_center - r) * (y < y_center + r)
        return mask


class AirfoilNaca00xx(Shape):
    def __init__(self, angle, thickness, *args):
        super().__init__(*args)
        self.angle = angle
        self.thickness = thickness

    def place_on_subdomain(self, domain: np.ndarray):
        # Create points on the edge of the airfoil
        x = np.linspace(0, 0.9906245881926358, max(domain.shape) * 10)
        y = 5 * self.thickness * (0.2926 * np.sqrt(x)
                                  - 0.1260 * x
                                  - 0.3516 * x ** 2
                                  + 0.2843 * x ** 3
                                  - 0.1015 * x ** 4)

        # Create the bottom half by flipping the top half
        x = np.concatenate([x, np.flip(x)])
        y = np.concatenate([y, np.flip(-y)])

        # Rotate to the desired angle of attach
        x, y = rotate_around_point(xy=(x, y), radians=np.deg2rad(self.angle), origin=(0.5, 0))

        # Scale from [0, 1]x[-thickness,thickness] to [0, size] where size is defined in Shape
        size = self.get_size(domain)
        x = x * size
        y = y * size + self.get_center_for_domain(domain)[1]

        # Create hull.
        xy_stacked = np.column_stack((x, y))
        hull = scipy.spatial.ConvexHull(xy_stacked)

        # Map hull onto a boolean array.
        path = matplotlib.path.Path(xy_stacked[hull.vertices])
        x, y = np.meshgrid(np.arange(domain.shape[0]), np.arange(domain.shape[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y, x)).T
        mask = path.contains_points(points)
        mask = mask.reshape(domain.shape)

        return mask

    def get_distance_to_point(self, domain: np.ndarray, x):
        pass


if __name__ == '__main__':
    # A nice and illustrative example of kernels and neighbouring nodes for the D2Q9 implementation of choice
    # is a square object on a (7,7) lattice:
    # domain_test = np.zeros((7, 7))
    # my_obstacle = Square(size_fraction=0.5)

    domain_test = np.zeros((400, 400))
    my_obstacle = AirfoilNaca00xx(angle=-45, thickness=0.1)
    my_obstacle.visualize()

    e, ex, ey, _ = physics.lattice.D2Q9()
    opp = physics.lattice.opp(e)
    kernels = physics.lattice.get_kernels(ex, ey)

    kernelind = 7
    print(f"Showing neighbouring nodes for (e_x, e_y) = ({ex[kernelind]}, {ey[kernelind]})")

    plt.matshow(kernels[kernelind, :, :].T, origin='lower')
    plt.show()
    mask = my_obstacle.place_on_subdomain(domain_test.copy())

    kernel_nodes = my_obstacle.directional_boundaries(domain_test.copy(), kernels, opp)
    plt.matshow((mask * 2 + kernel_nodes[kernelind, :, :]).T, origin='lower')
    plt.show()
