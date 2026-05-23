"""
action_z2_staggered_forwardtime.py — forward-time staggered fermion action.

Difference from action_z2_staggered.py (central-difference):
  Time hops are FORWARD ONLY (no backward t-1 hop), giving a block-upper-
  triangular Dirac matrix M_forward.  This is the Susskind/Wilson canonical
  form whose transfer-matrix derivation reproduces H_KS (modulo Trotter
  corrections in a_τ).

Structure:
  M_forward = block-upper-bidiagonal
    diagonal blocks  A_t  = mass + spatial Dirac at slice t  (V_3 × V_3)
    off-diagonal     B_t  = forward time hop coupling t → t+1
  det M_forward = ∏_t det(A_t).

QCD scalability:
  - Compute det M in O(N_E · V_3³) instead of O((N_E·V_3)³).  Linear in N_E.
  - For 2x2 V_3=4 toy: V_3³ = 64, so per-config det cost ≈ 5·64 = 320 FLOPs,
    vs ~8000 FLOPs for the full V_4×V_4 dense det.  ~25× speedup.
  - Scales correctly to larger N_E (linear, vs cubic for full det).

Sign convention:
  S_F = ψ̄ M ψ.  Forward time hop coefficient = +1 (not 1/2 like central-diff).
  Mass = m·(-1)^{x+y} (staggered parity, matching z2_setup).
  Spatial η: STANDARD KS time-dependent (η_x = (-1)^t, η_y = (-1)^{t+x}) —
  the same as action_z2_staggered.py, so spatial fermion physics is identical.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
    gauge_action, count_links_and_plaquettes,
)
from action_z2_kernels import (
    build_slice_block_A_kernel,
    compute_det_M_forward_kernel,
)


def build_slice_block_A(geom: LatticeGeometry, U: Z2GaugeConfig,
                        t: int) -> np.ndarray:
    """Diagonal block A_t = mass + spatial Dirac at slice t (V_3 × V_3).
    Delegates to action_z2_kernels.build_slice_block_A_kernel (Numba-jitted).
    """
    return build_slice_block_A_kernel(
        geom.Lx, geom.Ly, geom.m, t,
        U.U_x[t].astype(np.float64),
        U.U_y[t].astype(np.float64),
    )


def build_dirac_matrix_forward(geom: LatticeGeometry, U: Z2GaugeConfig) -> np.ndarray:
    """Full V_4 × V_4 forward-time Dirac matrix (block-upper-bidiagonal).
    Useful for debugging / cross-validation.  Slow — prefer compute_det_M_forward.
    """
    V4 = geom.V_4
    V3 = geom.V_3
    M = np.zeros((V4, V4), dtype=np.float64)
    for t in range(geom.N_E):
        # Diagonal block A_t at slice t
        A_t = build_slice_block_A(geom, U, t)
        M[t * V3:(t + 1) * V3, t * V3:(t + 1) * V3] = A_t
        # Forward time hop block B_t coupling slice t → t+1 (only if t+1 < N_E)
        if t + 1 < geom.N_E:
            for x in range(geom.Lx):
                for y in range(geom.Ly):
                    i = x * geom.Ly + y
                    # η_τ = 1 for staggered; forward hop coefficient = +1 (not 1/2)
                    M[t * V3 + i, (t + 1) * V3 + i] = +1.0 * U.U_t[t, x, y]
    return M


def compute_det_M_forward(geom: LatticeGeometry, U: Z2GaugeConfig) -> float:
    """det M_forward = ∏_t det(A_t).  Fast: O(N_E · V_3³).

    The off-diagonal forward-time blocks B_t do not contribute to det because
    M is block-upper-triangular.  Delegates to the Numba-jitted kernel.
    """
    return float(compute_det_M_forward_kernel(
        geom.Lx, geom.Ly, geom.N_E, geom.m,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
    ))


def compute_log_abs_det_M_forward(geom: LatticeGeometry,
                                   U: Z2GaugeConfig) -> tuple[float, int]:
    """log |det M_forward| via summing log |det A_t| — numerically stable.
    Returns (log_abs_det, sign).
    """
    log_abs = 0.0
    sign = 1
    for t in range(geom.N_E):
        A_t = build_slice_block_A(geom, U, t)
        s, logabs = np.linalg.slogdet(A_t)
        sign *= int(np.sign(s)) if s != 0 else 0
        log_abs += logabs
    return log_abs, sign


def _self_test():
    """Verify det M_forward via two routes (block product vs full dense det)
    and check that varying gauge changes det M sensibly."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    print(f"Forward-time action self-test: Lx=Ly=2, N_E={geom.N_E}, m={geom.m}")
    print(f"V_3={geom.V_3}, V_4={geom.V_4}")

    U_triv = Z2GaugeConfig.trivial(geom)
    M_full = build_dirac_matrix_forward(geom, U_triv)
    det_full = float(np.linalg.det(M_full))
    det_block = compute_det_M_forward(geom, U_triv)
    print(f"  trivial U: det via full M = {det_full:+.6e}")
    print(f"  trivial U: det via block-product = {det_block:+.6e}")
    print(f"  |Δ| = {abs(det_full - det_block):.2e}")
    assert abs(det_full - det_block) < 1e-10, "block-product det mismatch!"

    # Sanity check: M_full is block-upper-triangular
    V3, V4 = geom.V_3, geom.V_4
    for t in range(geom.N_E):
        for s in range(t):
            block = M_full[t * V3:(t + 1) * V3, s * V3:(s + 1) * V3]
            assert np.allclose(block, 0), (
                f"M_forward not block-upper-triangular: (t={t}, s={s}) nonzero")
    print(f"  block-upper-triangular structure verified.")

    # Random Z_2 config
    rng = np.random.default_rng(2026)
    U_rand = Z2GaugeConfig.random(geom, rng)
    det_full_r = float(np.linalg.det(build_dirac_matrix_forward(geom, U_rand)))
    det_block_r = compute_det_M_forward(geom, U_rand)
    print(f"  random U: det via full = {det_full_r:+.6e}, "
          f"via block = {det_block_r:+.6e}, |Δ| = {abs(det_full_r - det_block_r):.2e}")
    assert abs(det_full_r - det_block_r) < 1e-10

    # Also verify that flipping a U_t link does NOT change det (since B blocks
    # don't enter det).  This is a known feature of forward-time-only action.
    U_flipped = Z2GaugeConfig(
        geom=geom,
        U_x=U_rand.U_x.copy(),
        U_y=U_rand.U_y.copy(),
        U_t=U_rand.U_t.copy(),
    )
    U_flipped.U_t[0, 0, 0] *= -1
    det_flip = compute_det_M_forward(geom, U_flipped)
    print(f"  Flip U_t[0,0,0]: det unchanged? |Δ| = {abs(det_block_r - det_flip):.2e}  "
          f"(expected ~0 for forward-only action)")

    # Spatial link flip should change det
    U_flipped2 = Z2GaugeConfig(
        geom=geom,
        U_x=U_rand.U_x.copy(),
        U_y=U_rand.U_y.copy(),
        U_t=U_rand.U_t.copy(),
    )
    U_flipped2.U_x[0, 0, 0] *= -1
    det_flip2 = compute_det_M_forward(geom, U_flipped2)
    print(f"  Flip U_x[0,0,0]: det = {det_flip2:+.6e}, |Δ_from_orig| = "
          f"{abs(det_block_r - det_flip2):.2e}  (expected nonzero)")


if __name__ == "__main__":
    _self_test()
