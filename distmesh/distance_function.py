import numpy as np
from scipy.interpolate import interpn
from matplotlib import path


# Signed distance functions

def d_diff(d1: np.ndarray, d2: np.ndarray):
    """
    Compute signed distance function for set difference of two regions described by signed distance functions d1,d2.
    """
    return np.maximum(d1, -d2)


def d_intersect(d1: np.ndarray, d2: np.ndarray):
    """
    Compute signed distance function for set intersection of two regions described by signed distance functions d1,d2.
    """
    return np.maximum(d1, d2)


def d_matrix(p: np.ndarray, xx: np.ndarray, yy: np.ndarray, dd: np.ndarray):
    """
    Compute signed distance function by interpolation of the values dd on the Cartesian grid xx,yy.
    """
    return interpn((xx, yy), p, dd)


def d_poly(p: np.ndarray, pv: np.ndarray):
    """
    Compute signed distance function for polygon with vertices pv.
    """
    polygon_path = path.Path(pv)
    return (-1.0) ** polygon_path.contains_points(p) * d_segment(p, pv)


def d_rectangle(p: np.ndarray, x1: float, x2: float, y1: float, y2: float):
    """
    Compute signed distance function for rectangle with corners (x1,y1), (x2,y1), (x1,y2), (x2,y2).
    """
    return -np.minimum(np.minimum(np.minimum(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])


def d_segment(p: np.ndarray, pv: np.ndarray):
    """
    Compute distance from points p to the line segments in pv.
    """
    n_p = p.shape[0]
    n_pv = pv.shape[0] - 1

    dist = np.zeros((n_p, n_pv))

    for i in range(n_p):
        point = p[i]
        for j in range(n_pv):
            pv1 = pv[j]
            pv2 = pv[j + 1]
            diff = pv2 - pv1

            norm = np.sum((pv1 - pv2) ** 2)
            u = ((point[0] - pv1[0]) * diff[0] + (point[1] - pv1[1]) * diff[1]) / norm

            u = max(min(1.0, u), 0.0)
            x = pv1[0] + u * diff[0]
            y = pv1[1] + u * diff[1]

            on_line = np.array([x, y])

            dist[i, j] = np.sqrt(np.sum((point - on_line) ** 2))

    return np.min(dist, axis=1)


def d_sphere(p: np.ndarray, xc: float, yc: float, r: float):
    """"
    Compute signed distance function for sphere centered at xc,yc,zc with radius r.
    """
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def d_union(d1: np.ndarray, d2: np.ndarray):
    return np.minimum(d1, d2)


# Mesh size function

def h_matrix(p: np.ndarray, xx: np.ndarray, yy: np.ndarray, hh: np.ndarray):
    """
    Compute mesh size function by interpolation of the values hh on the Cartesian grid xx,yy.
    """
    return interpn((xx, yy), p, hh)


def h_uniform(p: np.ndarray):
    """
    Implements the trivial uniform mesh size function h=1.
    """
    return np.ones(p.shape[0])
