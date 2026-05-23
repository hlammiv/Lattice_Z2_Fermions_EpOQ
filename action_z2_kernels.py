"""
action_z2_kernels.py — Numba-jitted numerical kernels for the hot inner
loops of the Z_2+staggered Lagrangian MC.

These take raw numpy arrays (not dataclasses) so Numba can compile them.
Higher-level functions in action_z2_staggered.py and
action_z2_staggered_forwardtime.py wrap these for backward compatibility.

Speedups: typical 5-20× vs pure-Python loops, fully QCD-scalable
(Numba handles arbitrary lattice sizes the same way numpy does).
"""
from __future__ import annotations
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Central-difference Dirac matrix (full V_4 × V_4)
# ---------------------------------------------------------------------------
@njit(cache=True)
def build_dirac_matrix_kernel(Lx: int, Ly: int, N_E: int, m: float,
                              U_x: np.ndarray, U_y: np.ndarray,
                              U_t: np.ndarray) -> np.ndarray:
    """Central-difference staggered Dirac matrix M[σ] on L_x × L_y × N_E.

    Same conventions as action_z2_staggered.build_dirac_matrix:
      mass = m·(-1)^{x+y}
      η_τ = 1, η_x = (-1)^t, η_y = (-1)^{t+x}
      time hops: ±1/2 forward AND backward (central difference)
      spatial hops: ±1/2 forward AND backward
    """
    V3 = Lx * Ly
    V4 = V3 * N_E
    M = np.zeros((V4, V4), dtype=np.float64)

    # Mass term — staggered parity
    for t in range(N_E):
        for x in range(Lx):
            for y in range(Ly):
                i = (t * Lx + x) * Ly + y
                parity = 1.0 if (x + y) % 2 == 0 else -1.0
                M[i, i] = m * parity

    # τ-direction (η_τ = +1, central difference)
    for t in range(N_E):
        for x in range(Lx):
            for y in range(Ly):
                i = (t * Lx + x) * Ly + y
                eta_t = 1.0
                if t + 1 < N_E:
                    j = ((t + 1) * Lx + x) * Ly + y
                    M[i, j] += 0.5 * eta_t * U_t[t, x, y]
                if t - 1 >= 0:
                    j = ((t - 1) * Lx + x) * Ly + y
                    M[i, j] += -0.5 * eta_t * U_t[t - 1, x, y]

    # x-direction
    for t in range(N_E):
        for x in range(Lx):
            for y in range(Ly):
                i = (t * Lx + x) * Ly + y
                eta_x = -1.0 if (t % 2) else 1.0
                if x + 1 < Lx:
                    j = (t * Lx + x + 1) * Ly + y
                    M[i, j] += 0.5 * eta_x * U_x[t, x, y]
                if x - 1 >= 0:
                    j = (t * Lx + x - 1) * Ly + y
                    M[i, j] += -0.5 * eta_x * U_x[t, x - 1, y]

    # y-direction
    for t in range(N_E):
        for x in range(Lx):
            for y in range(Ly):
                i = (t * Lx + x) * Ly + y
                eta_y = -1.0 if ((t + x) % 2) else 1.0
                if y + 1 < Ly:
                    j = (t * Lx + x) * Ly + y + 1
                    M[i, j] += 0.5 * eta_y * U_y[t, x, y]
                if y - 1 >= 0:
                    j = (t * Lx + x) * Ly + y - 1
                    M[i, j] += -0.5 * eta_y * U_y[t, x, y - 1]

    return M


# ---------------------------------------------------------------------------
# Forward-time staggered block A_t (V_3 × V_3 per slice)
# ---------------------------------------------------------------------------
@njit(cache=True)
def build_slice_block_A_kernel(Lx: int, Ly: int, m: float, t: int,
                               U_x_slice: np.ndarray,
                               U_y_slice: np.ndarray) -> np.ndarray:
    """Forward-time spatial block A_t = mass + spatial Dirac at slice t.

    U_x_slice, U_y_slice are slices of U.U_x[t], U.U_y[t] respectively.
    """
    V3 = Lx * Ly
    A = np.zeros((V3, V3), dtype=np.float64)
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            A[i, i] = m * parity
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            eta_x = -1.0 if (t % 2) else 1.0
            if x + 1 < Lx:
                j = (x + 1) * Ly + y
                A[i, j] += 0.5 * eta_x * U_x_slice[x, y]
            if x - 1 >= 0:
                j = (x - 1) * Ly + y
                A[i, j] += -0.5 * eta_x * U_x_slice[x - 1, y]
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            eta_y = -1.0 if ((t + x) % 2) else 1.0
            if y + 1 < Ly:
                j = x * Ly + y + 1
                A[i, j] += 0.5 * eta_y * U_y_slice[x, y]
            if y - 1 >= 0:
                j = x * Ly + y - 1
                A[i, j] += -0.5 * eta_y * U_y_slice[x, y - 1]
    return A


@njit(cache=True)
def compute_det_M_forward_kernel(Lx: int, Ly: int, N_E: int, m: float,
                                  U_x: np.ndarray, U_y: np.ndarray) -> float:
    """det M_forward = ∏_t det(A_t).  O(N_E · V_3³). U_t ignored (does not enter)."""
    det_prod = 1.0
    for t in range(N_E):
        A_t = build_slice_block_A_kernel(Lx, Ly, m, t, U_x[t], U_y[t])
        det_prod *= np.linalg.det(A_t)
    return det_prod


# ---------------------------------------------------------------------------
# Gauge action S_g
# ---------------------------------------------------------------------------
@njit(cache=True)
def gauge_action_kernel(Lx: int, Ly: int, N_E: int,
                        K_E: float, K_M: float,
                        U_x: np.ndarray, U_y: np.ndarray,
                        U_t: np.ndarray) -> float:
    """S_g = -K_M · Σ_xy plaq - K_E · Σ_xτ plaq - K_E · Σ_yτ plaq."""
    total_E = 0.0
    total_M = 0.0
    for t in range(N_E):
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                u1 = U_x[t, x, y]
                u2 = U_y[t, x + 1, y]
                u3 = U_x[t, x, y + 1]
                u4 = U_y[t, x, y]
                total_M += u1 * u2 * u3 * u4
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                u1 = U_x[t, x, y]
                u2 = U_t[t, x + 1, y]
                u3 = U_x[t + 1, x, y]
                u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
    for t in range(N_E - 1):
        for x in range(Lx):
            for y in range(Ly - 1):
                u1 = U_y[t, x, y]
                u2 = U_t[t, x, y + 1]
                u3 = U_y[t + 1, x, y]
                u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
    return -K_E * total_E - K_M * total_M


# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation eagerly so worker processes don't pay
# the compile cost on first MC step.  Call once in each worker.
# ---------------------------------------------------------------------------
def warm_up():
    """Trigger JIT compilation of all kernels with tiny dummy inputs."""
    Lx, Ly, N_E = 2, 2, 3
    U_x = np.ones((N_E, Lx - 1, Ly), dtype=np.float64)
    U_y = np.ones((N_E, Lx, Ly - 1), dtype=np.float64)
    U_t = np.ones((N_E - 1, Lx, Ly), dtype=np.float64)
    _ = build_dirac_matrix_kernel(Lx, Ly, N_E, 0.5, U_x, U_y, U_t)
    _ = compute_det_M_forward_kernel(Lx, Ly, N_E, 0.5, U_x, U_y)
    _ = gauge_action_kernel(Lx, Ly, N_E, 1.0, 0.5, U_x, U_y, U_t)
