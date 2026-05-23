"""
transfer_matrix_baseline.py — sanity diagnostic for the T[U] derivation
milestone.

Purpose: show, with concrete numbers, the discrepancy between
  - G^bdy = (M^{-1})[(τ=0); (τ=N_E−1)]            (Lagrangian propagator,
                                                    currently used in
                                                    Slater-det corner
                                                    weights — WRONG)
  - T^{N_E}                                       (Hamiltonian transfer
                                                    matrix to N_E-th
                                                    power — the right
                                                    object)

Once T_step[U] is derived from the action, plug it into `T_NE_from_derivation`
below and re-run; the partition-function identity
    det M[U] = c · det(1 + T^{N_E}[U])
should hold across the three cases here (1-site free, 2×2 free, 2×2 random
gauge). The prefactor c is what the derivation pins down.

Run:  python /home/hlamm/Desktop/QC/logdet/m1_toy/transfer_matrix_baseline.py
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig,
)
# build_dirac_matrix and compute_G_bdy (central-diff lineage) were removed
# 2026-05-20. This module is legacy; not used by production pipeline.


def site_1d_M(N_E: int, m: float) -> np.ndarray:
    """Free 1+1d staggered Dirac on N_E temporal sites, OBC, V_3=1.

    Direct construction (independent of action_z2_staggered for cross-check):
      M[t, t]   = m
      M[t, t+1] = +1/2     (forward hop, η_τ = 1)
      M[t+1, t] = −1/2     (backward hop)
    OBC: drop hops that exit the lattice.
    """
    M = np.diag([m] * N_E).astype(float)
    for t in range(N_E - 1):
        M[t, t + 1] += 0.5
        M[t + 1, t] += -0.5
    return M


def T_F_single_free_1site(m: float) -> float:
    """Single-particle staggered transfer matrix for free 1+1d, V_3=1, OBC.

    Derivation: block-tridiagonal Schur of M_OBC gives recursion
        D_τ = m · D_{τ-1} + (1/4) · D_{τ-2}
    Eigenvalues  λ_± = (m ± √(m²+1)) / 2.
    Many-body T̂_F is diagonal in |Fock⟩ basis: vacuum eigenvalue = λ_+,
    1-particle eigenvalue = λ_-. So T̂_F = λ_+ · :exp(−â† h_eff â):
    with single-particle T̂_F^single = λ_-/λ_+.

    Returns the V_3×V_3 single-particle transfer matrix (scalar for V_3=1).
    Note: T̂_F^single is negative for free staggered (m > 0).
    """
    s = np.sqrt(m * m + 1)
    return (m - s) / (m + s)


def T_F_single_free(geom: LatticeGeometry) -> np.ndarray:
    """Single-particle staggered transfer matrix for free 2+1d (free spatial+temporal).

    For free temporal (F = (1/2)·I): T̂_F^single = (B − √(B²+4F²))·(B + √(B²+4F²))^{-1}
    where B = m·I + D_spatial (free spatial staggered Dirac, anti-Hermitian).

    Returns: V_3 × V_3 complex matrix.
    """
    V3 = geom.V_3
    # Build free spatial Dirac D_spatial on 2x2 OBC lattice with staggered phases
    # at t = 0 (irrelevant in free case since η_x t-dependence factors out)
    D = np.zeros((V3, V3), dtype=complex)
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            i = x * geom.Ly + y
            # η_x = (-1)^t = 1 at t=0; η_y = (-1)^{t+x} = (-1)^x at t=0
            eta_x = 1.0
            eta_y = -1.0 if (x % 2) else 1.0
            if x + 1 < geom.Lx:
                j = (x + 1) * geom.Ly + y
                D[i, j] += 0.5 * eta_x
                D[j, i] += -0.5 * eta_x
            if y + 1 < geom.Ly:
                j = x * geom.Ly + (y + 1)
                D[i, j] += 0.5 * eta_y
                D[j, i] += -0.5 * eta_y
    B = geom.m * np.eye(V3, dtype=complex) + D
    F2 = 0.25 * np.eye(V3, dtype=complex)  # F² for F = (1/2) I
    # Matrix sqrt of B² + 4F²
    from scipy.linalg import sqrtm
    radicand = B @ B + 4 * F2
    R = sqrtm(radicand)
    T_single = np.linalg.solve(B + R, B - R)
    return T_single


def build_B_slice(geom: LatticeGeometry, U: Z2GaugeConfig, t: int) -> np.ndarray:
    """Diagonal-time block B_t = M_mass + D_spatial(t)[U_x, U_y, η_phases].

    The (t, t) block of M, restricted to spatial site indices a ∈ [V_3].
    Includes:
      - STAGGERED mass m·(-1)^{x+y} on diagonal (matches z2_setup convention)
      - Staggered η_x = (-1)^t, η_y = (-1)^{t+x} on spatial hops
    """
    V3 = geom.V_3
    # Staggered mass: diagonal m·(-1)^{x+y}
    B = np.zeros((V3, V3), dtype=float)
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            i = x * geom.Ly + y
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            B[i, i] = geom.m * parity
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            i = x * geom.Ly + y
            eta_x = -1.0 if (t % 2) else 1.0
            eta_y = -1.0 if ((t + x) % 2) else 1.0
            if x + 1 < geom.Lx:
                j = (x + 1) * geom.Ly + y
                ux = U.U_x[t, x, y]
                B[i, j] += 0.5 * eta_x * ux
                B[j, i] += -0.5 * eta_x * ux
            if y + 1 < geom.Ly:
                j = x * geom.Ly + (y + 1)
                uy = U.U_y[t, x, y]
                B[i, j] += 0.5 * eta_y * uy
                B[j, i] += -0.5 * eta_y * uy
    return B


def build_F_slice(geom: LatticeGeometry, U: Z2GaugeConfig, t: int) -> np.ndarray:
    """Off-diagonal-time block F_t = (1/2) diag(U_t[t, x, y]).

    The (t, t+1) block of M; the (t+1, t) block is -F_t (antisymmetric kinetic).
    Diagonal V_3 × V_3 with η_τ = 1 absorbed.
    """
    V3 = geom.V_3
    diag = np.zeros(V3, dtype=float)
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            i = x * geom.Ly + y
            diag[i] = 0.5 * U.U_t[t, x, y]
    return np.diag(diag)


def block_schur_det(geom: LatticeGeometry, U: Z2GaugeConfig) -> tuple[float, list]:
    """Compute det M[U] via block-tridiagonal Schur recursion.

    With M structured as block-tridiagonal with B_t on diagonal, F_t above,
    -F_t below (for the staggered antisymmetric kinetic):

        G_0 = B_0
        G_t = B_t + F_{t-1} · G_{t-1}^{-1} · F_{t-1}    (t = 1, ..., N_E-1)
        det M = ∏_t det G_t

    Returns: (det M, list of det G_t for each slice).
    """
    G_t = build_B_slice(geom, U, 0)
    dets = [float(np.linalg.det(G_t))]
    for t in range(1, geom.N_E):
        F_prev = build_F_slice(geom, U, t - 1)
        B_t = build_B_slice(geom, U, t)
        G_t = B_t + F_prev @ np.linalg.solve(G_t, F_prev)
        dets.append(float(np.linalg.det(G_t)))
    return float(np.prod(dets)), dets


def T_NE_from_derivation(geom: LatticeGeometry, U: Z2GaugeConfig) -> np.ndarray:
    """Derived staggered single-particle T^{N_E-1}[U] for OBC.

    Currently implements the FREE case (no gauge coupling), V_3 = 1 or 4.
    For Z₂ gauge: use `block_schur_det` to verify identity; full T extraction
    via block-companion eigendecomposition is the next step.
    """
    if not np.all(U.U_x == 1) or not np.all(U.U_y == 1) or not np.all(U.U_t == 1):
        raise NotImplementedError(
            "T_NE single-particle operator for Z₂ gauge config: use slice-wise "
            "T̃_block product (TODO). Use block_schur_det for the det identity."
        )
    T_single = T_F_single_free(geom)
    return np.linalg.matrix_power(T_single, geom.N_E - 1)


def block_companion_X(geom: LatticeGeometry, U: Z2GaugeConfig) -> tuple[np.ndarray, float]:
    """Run the linear X_τ recursion (block-companion form).

    Linearization of G_τ = B_τ + F_{τ-1} G_{τ-1}^{-1} F_{τ-1}:
        X_τ = B_τ · F_{τ-1}^{-1} · X_{τ-1} + F_{τ-1} · F_{τ-2}^{-1} · X_{τ-2}
    with formal initial conditions X_{-1} = I, F_{-1} = I.

    The cumulative determinant identity:
        det M_OBC = det X_{N_E-1} · ∏_{τ=0}^{N_E-2} det F_τ

    Returns: (X_{N_E-1}, ∏ det F_τ)
    """
    V3 = geom.V_3
    F_minus1 = np.eye(V3)            # formal IC: F_{-1} = I
    X_minus1 = np.eye(V3)            # formal IC: X_{-1} = I
    X_0 = build_B_slice(geom, U, 0)  # X_0 = B_0 (matches G_0 = B_0)

    X_prev_prev = X_minus1
    X_prev = X_0
    F_prev_prev = F_minus1                       # F_{-1} (formal)
    F_prev = build_F_slice(geom, U, 0)           # F_0 (physical)

    for t in range(1, geom.N_E):
        B_t = build_B_slice(geom, U, t)
        X_t = (B_t @ np.linalg.solve(F_prev, X_prev)
               + F_prev @ np.linalg.solve(F_prev_prev, X_prev_prev))
        X_prev_prev = X_prev
        X_prev = X_t
        F_prev_prev = F_prev
        if t < geom.N_E - 1:
            F_prev = build_F_slice(geom, U, t)

    # ∏_{τ=0}^{N_E-2} det F_τ
    prod_det_F = 1.0
    for t in range(geom.N_E - 1):
        prod_det_F *= float(np.linalg.det(build_F_slice(geom, U, t)))
    return X_prev, prod_det_F


def verify_companion_identity(n_trials: int = 20, seed: int = 2026) -> dict:
    """Verify det M[U] = det X_{N-1} · ∏ det F_τ from block-companion recursion."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(seed)
    max_rel_err = 0.0
    n_pass = 0
    for trial in range(n_trials):
        U = Z2GaugeConfig.random(geom, rng)
        det_M_direct = float(np.linalg.det(build_dirac_matrix(geom, U)))
        X_NE_1, prod_F = block_companion_X(geom, U)
        det_M_companion = float(np.linalg.det(X_NE_1)) * prod_F
        rel = abs(det_M_companion - det_M_direct) / (abs(det_M_direct) + 1e-30)
        if rel < 1e-10:
            n_pass += 1
        max_rel_err = max(max_rel_err, rel)
    return {
        "n_trials": n_trials,
        "n_pass": n_pass,
        "max_rel_err": max_rel_err,
    }


def verify_schur_identity(n_trials: int = 20, seed: int = 2026) -> dict:
    """Verify det M[U] = ∏ det G_t from block-tridiagonal Schur, for random U.

    Tests on 2×2×4 Z₂ + staggered with m = 0.5. Returns max relative error.
    """
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(seed)
    max_rel_err = 0.0
    n_pass = 0
    for trial in range(n_trials):
        U = Z2GaugeConfig.random(geom, rng)
        M = build_dirac_matrix(geom, U)
        det_M_direct = float(np.linalg.det(M))
        det_M_schur, _ = block_schur_det(geom, U)
        rel = abs(det_M_schur - det_M_direct) / (abs(det_M_direct) + 1e-30)
        if rel < 1e-10:
            n_pass += 1
        max_rel_err = max(max_rel_err, rel)
    return {
        "n_trials": n_trials,
        "n_pass": n_pass,
        "max_rel_err": max_rel_err,
    }


# ---------------------------------------------------------------------------
def report_case1_1site_free():
    print("=" * 72)
    print("CASE 1: free 1+1d staggered, V_3 = 1, N_E = 4, m = 0.5")
    print("=" * 72)
    N_E, m = 4, 0.5

    # Direct construction
    M_direct = site_1d_M(N_E, m)

    # Cross-check via action_z2_staggered with Lx = Ly = 1
    geom_1 = LatticeGeometry(Lx=1, Ly=1, N_E=N_E, m=m)
    U_1 = Z2GaugeConfig.trivial(geom_1)
    M_module = build_dirac_matrix(geom_1, U_1)
    assert np.allclose(M_direct, M_module), \
        "1-site M mismatch between direct and action_z2_staggered!"
    print("  cross-check: direct M == action_z2_staggered M  ✓")

    det_M = np.linalg.det(M_direct)
    M_inv = np.linalg.inv(M_direct)
    G_bdy = M_inv[0, N_E - 1]
    beta = N_E * 1.0
    naive_T_NE = np.exp(-beta * m)

    # Derived T̂_F^single
    T_single_derived = T_F_single_free_1site(m)
    lam_plus = (m + np.sqrt(m * m + 1)) / 2
    lam_minus = (m - np.sqrt(m * m + 1)) / 2

    print(f"\n  M (4×4):")
    for row in M_direct:
        print("    [" + "  ".join(f"{v:+.2f}" for v in row) + "]")
    print(f"\n  det M (OBC, 4×4)                 = {det_M:+.6f}")
    print(f"  G^bdy = M^(-1)[0, N_E-1]          = {G_bdy:+.6f}  (Lagrangian, WRONG)")
    print(f"  naive continuum e^(-m·β)          = {naive_T_NE:+.6f}")
    print()
    print(f"  Derived T̂_F many-body eigenvalues:")
    print(f"    λ_+ = (m + √(m²+1))/2          = {lam_plus:+.6f}  (vacuum)")
    print(f"    λ_- = (m − √(m²+1))/2          = {lam_minus:+.6f}  (1-particle)")
    print(f"  Derived T̂_F^single = λ_-/λ_+    = {T_single_derived:+.6f}")
    print(f"  Pipeline error:  G^bdy / T̂_F^single = {G_bdy / T_single_derived:+.4f}")

    print("\n  Identity check: det M = ⟨BC_top|T̂_F^{N-1}|BC_bot⟩")
    print("  with boundary states determined by OBC half-hop structure.")
    # D_n = A λ_+^n + B λ_-^n with A + B = 1, A λ_+ + B λ_- = m
    A = (1 + m / np.sqrt(m * m + 1)) / 2
    B = 1 - A
    detM_predicted = A * lam_plus**N_E + B * lam_minus**N_E
    print(f"    A λ_+^N + B λ_-^N (closed form) = {detM_predicted:+.6f}")
    print(f"    A = {A:.6f},  B = {B:.6f}")
    print(f"    det M (matches?)                = {det_M:+.6f}  "
          f"({'✓' if abs(detM_predicted - det_M) < 1e-12 else 'MISMATCH'})")

    # Test Slater-det weights for ⟨n_0(t=0)⟩
    print(f"\n  Fock-state matrix elements ⟨Ψ|T̂_F^{{N_E-1}}|Ψ⟩ (use in Slater dets):")
    print(f"    ⟨0|T̂_F^{{N-1}}|0⟩  = λ_+^{{N-1}}            = {lam_plus**(N_E-1):+.6f}")
    print(f"    ⟨1|T̂_F^{{N-1}}|1⟩  = λ_-^{{N-1}}            = {lam_minus**(N_E-1):+.6f}")
    print(f"  Vacuum prefactor N[U] = λ_+ = {lam_plus:.6f}")
    print(f"  Single-particle T̂_F^single^{{N-1}} = {T_single_derived**(N_E-1):+.6f}")


def report_case2_v3_4_free():
    print()
    print("=" * 72)
    print("CASE 2: free 2+1d staggered, 2×2 spatial, V_3 = 4, N_E = 4, m = 0.5")
    print("=" * 72)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    U = Z2GaugeConfig.trivial(geom)
    M = build_dirac_matrix(geom, U)
    det_M = np.linalg.det(M)
    G_bdy = compute_G_bdy(geom, U)
    G_bdy = np.real_if_close(G_bdy)
    if np.iscomplexobj(G_bdy):
        G_bdy = G_bdy.real

    print(f"  det M (16×16, trivial gauge)            = {det_M:+.6f}")
    print(f"\n  G^bdy  (4×4, real part):")
    print("    " + np.array2string(G_bdy, precision=4, suppress_small=True,
                                   prefix="    "))
    print(f"\n  det(I + G^bdy)                          = "
          f"{np.linalg.det(np.eye(4) + G_bdy):+.6f}")

    print("\n  Identity to verify once T derived:")
    print("      det M  =  c · det(1 + T^{N_E})    [T is 4×4 single-particle]")
    print("\n  Spatial-mode-naive guess:  free 1-mode-per-site Hamiltonian")
    print("    h = m · I_4 → T^{N_E} = e^(−N_E·m) · I_4")
    print(f"    det(1 + e^(−2)·I_4) = {(1 + np.exp(-2))**4:+.6f}")
    print("  Real spatial dispersion will modify h; derivation will pin h down.")


def report_case3_v3_4_random_gauge():
    print()
    print("=" * 72)
    print("CASE 3: 2+1d Z₂-coupled staggered, 2×2 spatial, random gauge config")
    print("=" * 72)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    rng = np.random.default_rng(2026)
    U = Z2GaugeConfig.random(geom, rng)
    M = build_dirac_matrix(geom, U)
    det_M = np.linalg.det(M)
    G_bdy = compute_G_bdy(geom, U)
    G_bdy = np.real_if_close(G_bdy)
    if np.iscomplexobj(G_bdy):
        G_bdy = G_bdy.real
    print(f"  det M[U_random]                         = {det_M:+.6f}")
    print(f"  ||G^bdy[U_random]||_F                   = {np.linalg.norm(G_bdy):.4f}")
    print(f"  det(I + G^bdy[U_random])                = "
          f"{np.linalg.det(np.eye(4) + G_bdy):+.6f}")
    print("\n  Plug T^{N_E}[U_random] into the identity once derived and verify\n"
          "  for THIS config + several more from action MC.")


def attempt_with_derived_T():
    """If T_NE_from_derivation is implemented, run the identity check."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    U = Z2GaugeConfig.trivial(geom)
    try:
        T_NE = T_NE_from_derivation(geom, U)
    except NotImplementedError as e:
        print(f"\n[T_NE_from_derivation not yet implemented — {e}]")
        return

    M = build_dirac_matrix(geom, U)
    det_M = np.linalg.det(M)
    det_identity_rhs = np.linalg.det(np.eye(T_NE.shape[0]) + T_NE)
    print("\n" + "=" * 72)
    print("DERIVATION CHECK — identity det M = c · det(1 + T^{N_E})")
    print("=" * 72)
    print(f"  det M                          = {det_M:+.6f}")
    print(f"  det(1 + T^{{N_E}})              = {det_identity_rhs:+.6f}")
    print(f"  ratio (should equal prefactor) = {det_M / det_identity_rhs:+.6f}")


def main():
    report_case1_1site_free()
    report_case2_v3_4_free()
    report_case3_v3_4_random_gauge()
    attempt_with_derived_T()

    print("\n" + "=" * 72)
    print("Next: derive T_step from the Lagrangian action via Grassmann coherent-")
    print("state integration on time-slice pairs. Then:")
    print("  1. Replace T_NE_from_derivation() above with the derived formula.")
    print("  2. Verify det M = c · det(1 + T^{N_E}) across cases 1, 2, 3.")
    print("  3. Replace G^bdy with T^{N_E} in action_corner_direct.py.")
    print("See memory: project-transfer-matrix-derivation")
    print("=" * 72)


if __name__ == "__main__":
    main()
