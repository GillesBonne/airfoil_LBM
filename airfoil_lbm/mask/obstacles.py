"""Module to create a boolean array mask with an obstacle inside.
"""

import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.spatial


def circle(domain: np.ndarray):
    Nx, Ny = domain.shape
    r = Ny / 5
    x_center, y_center = (Ny, 0)
    x = np.linspace(0, Nx, Nx).reshape([Nx, 1])
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    mask = ((x - x_center) ** 2 + (y - y_center) ** 2 < r ** 2)
    return mask


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

    def __init__(self, size_fraction=None, size=None):
        if size_fraction is None and size is None:
            raise ValueError("At least one of size_fraction, size should be set.")
        self.size_fraction = size_fraction
        self.size = size

    def get_size(self, domain: np.ndarray):
        if self.size_fraction is not None:
            return min(domain.shape) * self.size_fraction
        else:
            return self.size

    def get_center_for_domain(self, domain: np.ndarray):
        """
        Calculates the center of a domain
        :param domain:
        :return:
        """
        return np.array(domain.shape) / 2.

    def place_on_domain(self, domain: np.ndarray):
        """
        Places this shape on the domain
        :param domain: 2D numpy array of type np.bool
        :return: A copy of the domain, with this shape placed on it, centered horizontally and vertically
        """
        raise NotImplementedError("Classes inheriting from Shape should implement _place_on_domain")

    def get_distance_to_point(self, domain: np.ndarray, x):
        """
        Returns the closest distance to the edge of the shape, analytically.
        :param x: A tuple containing the coordinates
        :return: The closest distance to the shape
        """
        raise NotImplementedError("Classes inheriting from Shape should implement get_distance_to_point")

    def visualize(self, domain: np.ndarray = None):
        if domain is None:
            domain = np.zeros((1000, 1000), dtype=np.bool)
        domain = self.place_on_domain(domain)

        plt.imshow(domain.T, origin='lower')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


class Circle(Shape):
    def get_distance_to_point(self, domain: np.ndarray, x):
        return np.linalg.norm(x - self.get_center_for_domain(domain)) - self.get_size(domain) / 2

    def place_on_domain(self, domain: np.ndarray):
        Nx, Ny = domain.shape
        r = self.get_size(domain) / 2.
        x_center, y_center = self.get_center_for_domain(domain)
        x = np.arange(Nx).reshape([Nx, 1])
        y = np.arange(Ny)
        mask = ((x - x_center) ** 2 + (y - y_center) ** 2 < r ** 2)
        return mask


class Naca:
    """NACA airfoil.
    """
    RESOLUTION = 1000
    BOX_SIZE_MULT_VERT = 2
    BOX_SIZE_MULT_HORI = 9

    # Divisor to shift the location of the center of the airfoil.
    # A divisor of 2 means the airfoil is position in the center.
    DIV_LEFT = 5
    DIV_DOWN = 2
    DIV_AIRFOIL = 2

    airfoil: np.ndarray
    box: np.ndarray

    def __init__(self, airfoil_size, xy, angle):
        self.airfoil_size = airfoil_size
        self.xy = xy
        self.angle = angle

        self.size_airfoil_x = self.airfoil_size
        self.size_airfoil_y = self.airfoil_size

        self.create_airfoil()

        self.size_box_x = self.BOX_SIZE_MULT_HORI * self.airfoil_size
        self.size_box_y = self.BOX_SIZE_MULT_VERT * self.airfoil_size

        self.place_airfoil()

    def create_airfoil(self):
        """Creates the airfoil.
        """
        x, y = rotate_around_point(xy=self.xy, radians=np.deg2rad(self.angle), origin=(0.5, 0))

        # Translate to [0,1], then scale to the size of the airfoil.
        x = self.airfoil_size * x
        y = self.airfoil_size * (y + 0.5)

        # Create hull.
        xy_stacked = np.column_stack((x, y))
        hull = scipy.spatial.ConvexHull(xy_stacked)

        # Map hull onto a boolean array.
        path = matplotlib.path.Path(xy_stacked[hull.vertices])
        x, y = np.meshgrid(np.arange(self.size_airfoil_x), np.arange(self.size_airfoil_y))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y, x)).T
        self.airfoil = path.contains_points(points)
        self.airfoil = self.airfoil.reshape((self.size_airfoil_y, self.size_airfoil_x))

    def place_airfoil(self):
        """Places the airfoil inside the box.
        """
        self.box = np.zeros((self.size_box_x, self.size_box_y), dtype=bool)

        x_lower = self.size_box_x // self.DIV_LEFT - self.size_airfoil_x // self.DIV_AIRFOIL
        x_upper = self.size_box_x // self.DIV_LEFT + self.size_airfoil_x // self.DIV_AIRFOIL
        y_lower = self.size_box_y // self.DIV_DOWN - self.size_airfoil_y // self.DIV_AIRFOIL
        y_upper = self.size_box_y // self.DIV_DOWN + self.size_airfoil_y // self.DIV_AIRFOIL

        self.box[x_lower:x_upper, y_lower:y_upper] = self.airfoil

    @property
    def airfoil_directional_boundaries(self):
        kernels = []
        for i in range(9):
            if i == 4:
                continue
            else:
                kernel = np.zeros(9)
                kernel[i] = 1
                kernels.append(kernel.reshape(3, 3))
        kernels = np.array(kernels, dtype=int)

        directional_boundaries = []
        for kernel in kernels:
            airfoil_boundary = scipy.ndimage.convolve(
                np.array(self.box, dtype=int), kernel, mode='constant')
            airfoil_boundary[self.box] = 0
            directional_boundaries.append(airfoil_boundary)
        directional_boundaries = np.array(directional_boundaries)

        return directional_boundaries

    @property
    def airfoil_boundary(self):
        airfoil_boundary = np.sum(self.airfoil_directional_boundaries, axis=0)
        return airfoil_boundary > 0

    def visualize_airfoil(self):
        """Outputs a visualization of the airfoil.
        """
        plt.imshow(self.airfoil.T, origin='lower')
        plt.show()

    def visualize_box(self):
        """Outputs a visualization of the airfoil inside the box.
        """
        plt.imshow(self.box.T, origin='lower')
        plt.show()

    def visualize_boundary(self):
        """Outputs a visualization of the boundary of the airfoil inside the box.
        """
        plt.imshow(self.airfoil_boundary.T, origin='lower')
        plt.show()


class Naca00xx(Naca):
    """Symmetrical 4-digit NACA airfoil.
    """

    def __init__(self, airfoil_size, angle, thickness):
        x = np.linspace(0, 0.9906245881926358, self.RESOLUTION // 2)
        y = 5 * thickness * (0.2926 * np.sqrt(x)
                             - 0.1260 * x
                             - 0.3516 * x ** 2
                             + 0.2843 * x ** 3
                             - 0.1015 * x ** 4)

        x = np.concatenate([x, np.flip(x)])
        y = np.concatenate([y, np.flip(-y)])

        super().__init__(airfoil_size, (x, y), angle)


if __name__ == '__main__':
    my_obstacle = Circle(size_fraction=0.5)
    my_obstacle.visualize()

    AFOIL = Naca00xx(airfoil_size=100, angle=37, thickness=0.3)

    AFOIL.visualize_boundary()
    # AFOIL.visualize_airfoil()
    # AFOIL.visualize_box()
