"""
Compute Lowner-John style inner and outer ellipsoidal approximations
without external conic solvers.

This version replaces MOSEK Fusion with numerical algorithms:
- Outer ellipsoid: Khachiyan's MVEE algorithm.
- Inner ellipsoid: custom optimization over center and shape with
  a dual solve for each center iterate.
"""

import numpy as np


def _symmetrize(M):
    return 0.5 * (M + M.T)


def _psd_project(M, min_eig=1.0e-10):
    vals, vecs = np.linalg.eigh(_symmetrize(M))
    vals = np.maximum(vals, min_eig)
    return vecs @ np.diag(vals) @ vecs.T


def _find_feasible_center(A, b, margin=1.0e-3, max_iter=5000):
    """Simple projection-style search for a strictly feasible d."""
    m, n = A.shape
    d = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        margins = b - A @ d
        j = int(np.argmin(margins))
        if margins[j] > margin:
            return d

        aj = A[j]
        denom = float(np.dot(aj, aj)) + 1.0e-12
        violation = margin - margins[j]
        d = d - 0.8 * (violation / denom) * aj

    margins = b - A @ d
    if np.min(margins) <= 0.0:
        raise RuntimeError("Failed to find a strictly feasible center for Ax <= b")
    return d


def _sqrt_psd(M):
    vals, vecs = np.linalg.eigh(_symmetrize(M))
    vals = np.maximum(vals, 1.0e-14)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _solve_centered_inner_shape(B, max_iter=1500, tol=1.0e-7):
    """
    Solve max log det(X) subject to b_i^T X b_i <= 1, X >> 0.
    Uses projected gradient descent on the dual variables u >= 0.
    """
    m, n = B.shape
    u = np.ones(m, dtype=float)

    for k in range(max_iter):
        M = np.zeros((n, n), dtype=float)
        for i in range(m):
            bi = B[i][:, None]
            M += u[i] * (bi @ bi.T)

        M = _psd_project(M)
        Minv = np.linalg.inv(M)
        q = np.array([float(B[i] @ Minv @ B[i]) for i in range(m)])

        grad = 1.0 - q
        if np.linalg.norm(grad, ord=np.inf) < tol:
            break

        lr = 0.1 / np.sqrt(1.0 + 0.01 * k)
        u = np.maximum(u - lr * grad, 1.0e-14)

    M = np.zeros((n, n), dtype=float)
    for i in range(m):
        bi = B[i][:, None]
        M += u[i] * (bi @ bi.T)
    M = _psd_project(M)
    X = np.linalg.inv(M)
    return _symmetrize(X)


def lownerjohn_inner(A, b, max_iter=120, tol=1.0e-6):
    """
    Approximate maximum-volume inscribed ellipsoid for polytope Ax <= b.
    Returns (C, d) where ellipse is {x = C u + d : ||u||_2 <= 1}.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    _, n = A.shape

    d = _find_feasible_center(A, b)

    def evaluate(dvec):
        s = b - A @ dvec
        if np.min(s) <= 1.0e-10:
            return -np.inf, None

        B = A / s[:, None]
        X = _solve_centered_inner_shape(B)
        sign, logdetX = np.linalg.slogdet(X)
        if sign <= 0:
            return -np.inf, None
        return 0.5 * logdetX, X

    best_val, best_X = evaluate(d)
    if not np.isfinite(best_val):
        raise RuntimeError("Failed to initialize inner-ellipsoid optimization")

    fd_eps = 1.0e-4
    for _ in range(max_iter):
        grad = np.zeros(n, dtype=float)
        for j in range(n):
            ej = np.zeros(n, dtype=float)
            ej[j] = fd_eps
            vp, _ = evaluate(d + ej)
            vm, _ = evaluate(d - ej)
            grad[j] = (vp - vm) / (2.0 * fd_eps)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        step = 0.5
        improved = False
        while step > 1.0e-6:
            d_try = d + step * grad
            val_try, X_try = evaluate(d_try)
            if np.isfinite(val_try) and val_try > best_val:
                d = d_try
                best_val = val_try
                best_X = X_try
                improved = True
                break
            step *= 0.5

        if not improved:
            break

    C = _sqrt_psd(best_X)

    margins = b - A @ d
    edge_norms = np.linalg.norm(A @ C, axis=1)
    alpha = float(np.min(margins / (edge_norms + 1.0e-14)))
    if alpha < 1.0:
        C *= max(0.0, 0.999 * alpha)

    C = _symmetrize(C)
    return C.tolist(), d.tolist()


def lownerjohn_outer(points, tol=1.0e-7, max_iter=2000):
    """
    Minimum-volume enclosing ellipsoid using Khachiyan's algorithm.
    Returns (P, c) for {x | ||P x - c||_2 <= 1}.
    """
    X = np.asarray(points, dtype=float)
    m, n = X.shape
    if m < n + 1:
        raise RuntimeError("Need at least n+1 points for a full-dimensional ellipsoid")

    Q = np.vstack([X.T, np.ones((1, m))])
    u = np.full(m, 1.0 / m)

    for _ in range(max_iter):
        QuQ = Q @ np.diag(u) @ Q.T
        inv_QuQ = np.linalg.inv(QuQ)
        M = np.einsum("ij,jk,ki->i", Q.T, inv_QuQ, Q)

        j = int(np.argmax(M))
        max_m = float(M[j])
        step = (max_m - (n + 1.0)) / ((n + 1.0) * (max_m - 1.0))
        step = max(0.0, min(1.0, step))

        new_u = (1.0 - step) * u
        new_u[j] += step

        if np.linalg.norm(new_u - u) <= tol:
            u = new_u
            break
        u = new_u

    center = X.T @ u
    S = X.T @ np.diag(u) @ X - np.outer(center, center)
    A = _psd_project(np.linalg.inv(S) / n)

    vals, vecs = np.linalg.eigh(A)
    sqrtA = vecs @ np.diag(np.sqrt(np.maximum(vals, 1.0e-12))) @ vecs.T

    P = sqrtA
    c = P @ center
    return P.tolist(), c.tolist()


if __name__ == "__main__":
    p = [[0.0, 0.0], [1.0, 3.0], [5.5, 4.5], [7.0, 4.0], [7.0, 1.0], [3.0, -2.0]]

    A = [[-p[i][1] + p[i - 1][1], p[i][0] - p[i - 1][0]] for i in range(len(p))]
    b = [A[i][0] * p[i][0] + A[i][1] * p[i][1] for i in range(len(p))]

    Po, co = lownerjohn_outer(p)
    Ci, di = lownerjohn_inner(A, b)

    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        Ci = np.asarray(Ci)
        di = np.asarray(di)
        Po = np.asarray(Po)
        co = np.asarray(co)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(patches.Polygon(p, fill=False, color="red"))

        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        x = Ci[0, 0] * np.cos(theta) + Ci[0, 1] * np.sin(theta) + di[0]
        y = Ci[1, 0] * np.cos(theta) + Ci[1, 1] * np.sin(theta) + di[1]
        ax.plot(x, y)

        xg, yg = np.meshgrid(np.arange(-1.0, 8.0, 0.025), np.arange(-3.0, 6.5, 0.025))
        ax.contour(xg, yg, (Po[0, 0] * xg + Po[0, 1] * yg - co[0]) ** 2 + (Po[1, 0] * xg + Po[1, 1] * yg - co[1]) ** 2, [1.0])

        ax.autoscale_view()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.savefig("ellipsoid.png")
        backend = plt.get_backend().lower()
        if "agg" in backend:
            for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"):
                try:
                    plt.switch_backend(candidate)
                    break
                except Exception:
                    continue

        if "agg" in plt.get_backend().lower():
            print(f"Matplotlib backend '{plt.get_backend()}' is non-interactive; saved ellipsoid.png")
        else:
            plt.show()
    except Exception:
        print("Inner:")
        print("  C =", Ci)
        print("  d =", di)
        print("Outer:")
        print("  P =", Po)
        print("  c =", co)
