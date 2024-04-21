from distmesh.distance_function import *
from distmesh.distmesh import dist_mesh
import os


def uniform_mesh_on_unit_cirle():
    folder = r".\output"
    if not os.path.exists(folder):
        os.makedirs(folder)

    def plot(count, points, triangles):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.suptitle("DistMesh iteration " + str(count))
        ax.triplot(points[:, 0], points[:, 1], triangles)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        plt.savefig(folder + r"\\plot" + str(count) + ".png", dpi=200)
        plt.close()

    fd = lambda p: d_sphere(p, 0, 0, 1)
    fh = lambda p: h_uniform(p)
    pfix = None
    bbox = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    h0 = 0.2

    dist_mesh(fd, fh, h0, bbox, pfix, plot)


def rectangle_with_circular_hole():
    folder = r".\output"
    if not os.path.exists(folder):
        os.makedirs(folder)

    def plot(count, points, triangles):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.suptitle("DistMesh iteration " + str(count))
        ax.triplot(points[:, 0], points[:, 1], triangles)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        plt.savefig(folder + r"\\plot" + str(count) + ".png", dpi=200)
        plt.close()

    fd = lambda p: d_diff(d_rectangle(p, -1.0, 1.0, -1.0, 1.0), d_sphere(p, 0, 0, 0.5))
    fh = lambda p: 0.05 + 0.3 * d_sphere(p, 0.0, 0.0, 0.5)
    pfix = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    bbox = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    h0 = 0.05

    dist_mesh(fd, fh, h0, bbox, pfix, plot)


def polygon():
    folder = r".\output"
    if not os.path.exists(folder):
        os.makedirs(folder)

    def plot(count, points, triangles):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.suptitle("DistMesh iteration " + str(count))
        ax.triplot(points[:, 0], points[:, 1], triangles)
        ax.set_xlim(-1.1, 2.1)
        ax.set_ylim(-1.1, 1.1)

        plt.savefig(folder + r"\\plot" + str(count) + ".png", dpi=200)
        plt.close()

    pfix = np.array([[-0.4, -0.5], [0.4, -0.2], [0.4, -0.7], [1.5, -0.4], [0.9, 0.1],
                     [1.6, 0.8], [0.5, 0.5], [0.2, 1.0], [0.1, 0.4], [-0.7, 0.7], [-0.4, -0.5]])
    fd = lambda p: d_poly(p, pfix)
    fh = lambda p: h_uniform(p)
    bbox = np.array([[-1.0, 2.0], [-1.0, 1.0]])
    h0 = 0.1

    dist_mesh(fd, fh, h0, bbox, pfix, plot)


def airfoil():
    folder = r".\output"
    if not os.path.exists(folder):
        os.makedirs(folder)

    def plot(count, points, triangles):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("DistMesh iteration " + str(count))
        ax1.triplot(points[:, 0], points[:, 1], triangles)
        ax1.set_xlim(-2.1, 6.1)
        ax1.set_ylim(-4.1, 4.1)

        ax2.triplot(points[:, 0], points[:, 1], triangles)
        ax2.set_xlim(-0.3, 1.3)
        ax2.set_ylim(-0.35, 0.35)

        plt.savefig(folder + r"\\plot" + str(count) + ".png", dpi=200)
        plt.close()

    # Create mesh for airfoil
    h_lead = 0.01
    h_trail = 0.04
    h_max = 2
    circ_x = 2
    circ_r = 4
    a1 = 0.12 / 0.2 * 0.2969
    a2 = 0.12 / 0.2 * np.array([-0.1036, 0.2843, -0.3516, -0.1260, 0.0])

    fd = lambda p: d_diff(d_sphere(p, circ_x, 0, circ_r),
                          (np.abs(p[:, 1]) - np.polyval(a2, p[:, 0])) ** 2 - a1 ** 2 * p[:, 0])
    fh = lambda p: np.minimum(np.minimum(h_lead + 0.3 * d_sphere(p, 0, 0, 0), h_trail + 0.3 * d_sphere(p, 1, 0, 0)),
                              h_max)
    fix_x = 1 - h_trail * np.cumsum(1.3 ** np.arange(0, 5))
    fix_y = a1 * np.sqrt(fix_x) + np.polyval(a2, fix_x)
    fix1 = np.array([(circ_x - circ_r, 0), (circ_x + circ_r, 0), (circ_x, -circ_r), (circ_x, circ_r), (0, 0), (1, 0)])
    fix2 = np.vstack((fix_x, fix_y)).T
    fix3 = np.vstack((fix_x, -fix_y)).T
    pfix = np.vstack((fix1, fix2, fix3))
    bbox = np.array([[circ_x - circ_r, circ_x + circ_r], [-circ_r, circ_r]])
    h0 = min(h_lead, h_trail, h_max)

    dist_mesh(fd, fh, h0, bbox, pfix, plot)


if __name__ == "__main__":
    polygon()
