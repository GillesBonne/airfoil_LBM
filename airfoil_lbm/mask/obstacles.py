"""Module to create a boolean array mask with an obstacle inside.
"""

import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.spatial
import physics
import shapely.affinity
import shapely.geometry


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
        """
        Calculate the size that this Shape should be when placing itself onto the domain
        :param domain: A numpy array
        :return: An integer value
        """
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

    def get_shape_geometry(self, domain: np.ndarray) -> shapely.geometry.base.BaseGeometry:
        """
        Creates a shapely shape, e.g. by creating a polygon by sequential [x,y] vertices that describes the edge of this
         Shape.
        :param domain: The domain this shape should be placed on
        :return: a BaseGeometry subclass
        """
        raise NotImplementedError("Classes inheriting from Shape should implement _place_on_domain")

    def place_on_subdomain(self, domain: np.ndarray):
        """
        Places this shape in the center of the supplied (sub)domain
        :param domain: 2D numpy array of type np.bool
        :return: A copy of the domain, with this shape placed on it, centered horizontally and vertically
        """
        geometry = self.get_shape_geometry(domain)
        xy_stacked = np.array(geometry.boundary.xy).T
        hull = scipy.spatial.ConvexHull(xy_stacked)

        # Map hull onto a boolean array.
        path = matplotlib.path.Path(xy_stacked[hull.vertices])
        x, y = np.meshgrid(np.arange(domain.shape[0]), np.arange(domain.shape[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y, x)).T
        mask = path.contains_points(points)
        mask = mask.reshape(domain.shape)

        return mask

    def place_on_domain(self, domain, x_size, y_size, center_x=0.5, center_y=0.5):
        """
        Places this Shape in the domain, by first cutting out a small subdomain. This method will then
        call the abstract method place_on_subdomain.
        :param domain: 2D numpy array
        :param x_size: The x size of the subdomain
        :param y_size: The y size of the subdomain
        :param center_x: The center x position of the subdomain, relative to the total x size
        :param center_y: The center y position of the subdomain, relative to the total y size
        :return: A copy of the domain, with this shape placed on it, centered at center_x, center_y
        """
        center_x = int(domain.shape[0] * center_x)
        center_y = int(domain.shape[1] * center_y)

        y_start = center_y - y_size // 2
        y_end = y_start + y_size

        x_start = center_x - x_size // 2
        x_end = x_start + x_size

        subdomain = domain[x_start:x_end, y_start:y_end]
        domain[x_start:x_end, y_start:y_end] = self.place_on_subdomain(subdomain)
        return domain

    def get_kernel_ratios(self, domain: np.ndarray, e, kernels) -> np.ndarray:
        """
        Returns the ratio of the distance along a kernel direction for which it intersects with this Shape
        :param kernels:
        :param domain:
        :return: The closest distance to the shape
        """
        polygon = self.get_shape_geometry(domain)

        kernel_nodes = my_obstacle.directional_boundaries(domain_test.copy(), kernels, opp)

        # Fill q with NaN values and only replace the value when the kernel convolution is nonzero
        q = np.ones((kernels.shape[0], *domain.shape)) * np.nan

        # Loop through all kernels / displacement vectors
        for i, e_i in enumerate(e):
            nodes = kernel_nodes[i]

            # Loop through all coordinates where this kernel has a 1
            for xy1 in np.transpose(np.nonzero(nodes)):
                # Create a line between that coordinate, and the coordinate inside the shape along e_i
                x1, y1 = xy1
                xy2 = xy1 + e_i
                line = shapely.geometry.LineString([xy1, xy2])

                # Intersect that line with the polygon that is this Shape
                intersection = line.intersection(polygon)
                xy3 = np.array(intersection.coords.xy)[:, 0]
                # Calculate the q value
                q[i, x1, y1] = np.linalg.norm(xy3 - xy1) / np.linalg.norm(e_i)
        return q

    def visualize(self, domain: np.ndarray = None):
        if domain is None:
            domain = np.zeros((200, 200), dtype=np.bool)
        domain = self.place_on_subdomain(domain)

        plt.imshow(domain.T, origin='lower')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def directional_boundaries(self, domain, kernels, opp):
        """
        For each kernel, the  points outside this Shape placed in the supplied domain that would be taken
        into this shape will be set to 1.
        :param domain: A numpy array containing the fluid domain in which this Shape should be placed
        :param kernels: The kernels to test for. Should be 2D arrays containing a single 1 value.
        :param opp: An array containing the indices i' of the kernel that mirrors the kernel at index i
        :return: A numpy array of shape (kernels.shape[0], *domain.shape)
        """
        # TODO: Place on domain instead of subdomain. Should probably make a function that cuts and translates subdomain
        #  out of domain
        mask = self.place_on_subdomain(domain)

        directional_boundaries = np.zeros((kernels.shape[0], *domain.shape), dtype=int)
        for i, kernel in enumerate(kernels):
            # Set all points outside the mask to 1 if its kernel is inside the mask
            directional_boundaries[opp[i], ~mask] = scipy.ndimage.convolve(np.array(mask, dtype=int),
                                                                           kernel,
                                                                           mode='constant')[~mask]
        return directional_boundaries


class Circle(Shape):
    def get_shape_geometry(self, domain: np.ndarray) -> shapely.geometry.base.BaseGeometry:
        r = self.get_size(domain) / 2.
        x_center, y_center = self.get_center_for_domain(domain)
        point = shapely.geometry.Point(x_center, y_center).buffer(1)
        return shapely.affinity.scale(point, r, r)


class Square(Shape):
    def get_shape_geometry(self, domain: np.ndarray):
        r = self.get_size(domain) / 2.
        x_center, y_center = self.get_center_for_domain(domain)
        return shapely.geometry.Polygon((
            (x_center - r, y_center - r),
            (x_center - r, y_center + r),
            (x_center + r, y_center + r),
            (x_center + r, y_center - r)
        ))


class AirfoilNaca00xx(Shape):
    def __init__(self, angle, thickness, *args):
        super().__init__(*args)
        self.angle = angle
        self.thickness = thickness

    def get_shape_geometry(self, domain: np.ndarray):
        # Create points on the edge of the airfoil
        x = np.linspace(0, 0.9906245881926358, max(domain.shape) * 100)
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

        xy = np.column_stack((x, y))
        return shapely.geometry.Polygon(xy)


if __name__ == '__main__':
    # A nice and illustrative example of kernels and neighbouring nodes for the D2Q9 implementation of choice
    # is a square object on a (7,7) lattice:
    # domain_test = np.zeros((8, 8))
    # my_obstacle = Square(size_fraction=0.5)

    domain_test = np.zeros((100, 100))
    my_obstacle = AirfoilNaca00xx(angle=-45, thickness=0.1)
    my_obstacle.visualize()

    # Get lattice parameters
    e, ex, ey, _ = physics.lattice.D2Q9()
    opp = physics.lattice.opp(e)
    kernels = physics.lattice.get_kernels(ex, ey)

    # Get the q factors from the obstacle for this set of kernels
    q = my_obstacle.get_kernel_ratios(domain_test, e, kernels)

    # Show neighbouring nodes and q-factors for one kernel
    kernelind = 1

    plt.matshow(q[kernelind, :, :].T, origin='lower')
    plt.title(f"q for kernel {kernelind}, (e_x, e_y) = ({ex[kernelind]}, {ey[kernelind]})")
    plt.colorbar()
    plt.show()

    print(f"Showing neighbouring nodes for (e_x, e_y) = ({ex[kernelind]}, {ey[kernelind]})")

    plt.matshow(kernels[kernelind, :, :].T, origin='lower')
    plt.title(f"Kernel {kernelind}, (e_x, e_y) = ({ex[kernelind]}, {ey[kernelind]})")
    plt.show()
    mask = my_obstacle.place_on_subdomain(domain_test.copy())

    kernel_nodes = my_obstacle.directional_boundaries(domain_test.copy(), kernels, opp)
    plt.matshow((mask * 2 + kernel_nodes[kernelind, :, :]).T, origin='lower')
    plt.show()
