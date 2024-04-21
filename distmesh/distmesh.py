import numpy as np
from scipy.spatial import Delaunay
from typing import Callable


def dist_mesh(fd: Callable[[np.ndarray], np.ndarray], fh: Callable[[np.ndarray], np.ndarray], h0: float,
              bbox: np.ndarray, p_fix: np.ndarray = None,
              plot_func: Callable[[int, np.ndarray, np.ndarray], None] = None):
    """
    :param fd: Signed distance function representing the geometry. This should return the signed distance from each node
    location p to the closest boundary.
    :param fh: The (relative) desired edge length function.
    :param h0: Initial distance between points in the initial distribution.
    :param bbox: Bounding box for the region where the initial points are distributed.
    :param p_fix: Fixed node positions for the generated grid.
    :param plot_func: Possible plotting function which will be called after every iteration.
    :return: Tuple (p,t), where p are the generated points, and t is the triangulation.
    """
    dp_tol = 0.001
    t_tol = 0.1
    F_scale = 1.2
    delta_t = 0.2
    g_eps = 0.001 * h0
    d_eps = np.sqrt(np.finfo(np.double).eps) * h0

    # Bounding box
    x_min = bbox[0][0]
    x_max = bbox[0][1]
    y_min = bbox[1][0]
    y_max = bbox[1][1]

    # 1. Create initial distribution in bounding box (equilateral triangles)
    x, y = np.meshgrid(np.arange(x_min, x_max + h0, h0), np.arange(y_min, y_max + h0, np.sqrt(3) / 2 * h0))
    x[1::2] += h0 / 2
    p = np.vstack((x.flat, y.flat)).T

    #  2. Remove points outside the region, apply the rejection method
    p = p[fd(p) < g_eps]
    r0 = 1.0 / fh(p) ** 2
    p = p[np.random.random(p.shape[0]) < r0 / r0.max()]
    if p_fix is not None:
        p = p[~np.isin(p, p_fix)[:, 0]]
        p = np.vstack((p_fix, p))

        N_fix = p_fix.shape[0]
    else:
        N_fix = 0

    p_old = np.inf
    count = 0
    while True:
        count += 1

        # 3. Retriangulation by the Delaunay algorithm
        if np.sqrt((p - p_old) ** 2).sum(1).max() / h0 > t_tol:
            p_old = p.copy()
            t = Delaunay(p).simplices
            p_mid = p[t].sum(1) / 3.0
            t = t[fd(p_mid) < -g_eps]

            # 4. Describe each bar by a unique pair of nodes
            bars = np.vstack((t[:, [0, 1]],
                              t[:, [1, 2]],
                              t[:, [2, 0]]))
            bars.sort(axis=1)
            bars = np.unique(bars, axis=0)

        # 5. Graphical output of the current mesh
        if plot_func is not None:
            plot_func(count, p, t)

        # 6. Move mesh points based on bar lengths L and forces F
        bar_vec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.sqrt((bar_vec ** 2).sum(1))
        h_bars = fh(0.5 * (p[bars[:, 0]] + p[bars[:, 1]]))
        L0 = h_bars * F_scale * np.sqrt((L ** 2).sum() / (h_bars ** 2).sum())
        F = L0 - L
        F[F < 0] = 0.0
        F_vec = (F[:, np.newaxis] / L[:, np.newaxis]).dot(np.ones((1, 2))) * bar_vec

        F_tot = np.zeros(p.shape)
        for i in range(bars.shape[0]):
            p1 = bars[i, 0]
            p2 = bars[i, 1]
            F_tot[p1, :] += F_vec[i, :]
            F_tot[p2, :] -= F_vec[i, :]

        F_tot[:N_fix] = 0.0
        p += delta_t * F_tot

        # 7. Bring outside points back to the boundary
        d = fd(p)
        ix = d > 0
        if ix.any():
            d_gradx = (fd(p[ix] + [d_eps, 0.0]) - d[ix]) / d_eps
            d_grady = (fd(p[ix] + [0.0, d_eps]) - d[ix]) / d_eps
            d_grad_sqr = d_gradx ** 2 + d_grady ** 2
            p[ix] -= (d[ix] * np.vstack((d_gradx, d_grady)) / d_grad_sqr).T

        # 8. Termination criterion: All interior nodes move less than dp_tol (scaled)
        print("Iteration number: " + str(count))
        print("Convergence number: " + str((np.sqrt((delta_t * F_tot[d < -g_eps] ** 2).sum(1)) / h0).max() / dp_tol))
        print()
        if (np.sqrt((delta_t * F_tot[d < -g_eps] ** 2).sum(1)) / h0).max() < dp_tol:
            break
    return p, t
