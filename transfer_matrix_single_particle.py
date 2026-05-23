"""
transfer_matrix_single_particle.py — Phase 4 of the P2 pipeline.

After M-K auxiliary-field integration, the effective T̂_F on χ̂-only Fock space
factorizes as
    T̂_F^single_step[U_t] = e^{-a · H_lat[U_t]}
where H_lat[U_t] is the single-particle V_3 × V_3 Kogut-Susskind Hamiltonian
at time slice t:
    diagonal:    m · (-1)^{x+y}                    (staggered mass)
    hopping:     (1/2) · η'_k(n) · U_k(n)          (gauge-coupled NN hop)

For free V_3=1: H_lat = m → T̂_F^single = e^{-am}, T̂_F^4 = e^{-2} = 0.135 ✓
matches the V_3=1 closed form exactly.

For Z₂ gauge: H_lat[U] is U-dependent.  Cumulative
    T̂_F^{N-1, single}[U] = T_step[U_{N-1}] · ... · T_step[U_1]
where each T_step = e^{-a·H_lat[U_t]}.

This is the SCALABLE form:
  - H_lat is V_3 × V_3 (small even at QCD scale: ~10^4 × 10^4 sparse).
  - At QCD scale, T̂_F^single_step is applied to vectors via Krylov / Chebyshev
    matrix exponential algorithms (no need to build matrix explicitly).
  - Slater-det matrix elements ⟨Ψ_top|T̂_F^{N-1}|Ψ_bot⟩ are V_3 × V_3 sub-dets.
  - Pseudofermion stochastic estimator: standard lattice QCD HMC framework
    applied to single-particle operator.

Toy-scale validation: verify Slater-det formulation gives same result as
M-K's exact 256-dim T̂_F^4 computation.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
from action_z2_staggered import LatticeGeometry, Z2GaugeConfig


# ----------------------------------------------------------------------------
# Build single-particle KS Hamiltonian per slice
# ----------------------------------------------------------------------------
def build_H_lat_slice(geom: LatticeGeometry, U: Z2GaugeConfig,
                      t: int) -> np.ndarray:
    """Build V_3 × V_3 single-particle KS Hamiltonian at time slice t.

    H_lat = m · diag((-1)^{x+y}) + (1/2) Σ_k η'_k · U_k · NN_hop
    with M-K's t-INDEPENDENT η':
        η'_x(x,y) = (-1)^{x+y}
        η'_y(x,y) = (-1)^y

    For consistency with our action's M matrix (which uses std KS t-dep η),
    we use the K-S transformation implicit in M-K's auxiliary-field reduction.
    """
    V3 = geom.V_3
    H = np.zeros((V3, V3), dtype=complex)

    def site(x, y):
        return x * geom.Ly + y

    # Staggered mass on diagonal
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            i = site(x, y)
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            H[i, i] = geom.m * parity

    # Spatial hopping (symmetric, Hermitian)
    for x in range(geom.Lx):
        for y in range(geom.Ly):
            n = site(x, y)
            # x-direction hop
            if x + 1 < geom.Lx:
                n_k = site(x + 1, y)
                eta_kx = -1.0 if ((x + y) % 2) else 1.0  # η'_x = (-1)^{x+y}
                ux = U.U_x[t, x, y]
                H[n, n_k] += 0.5 * eta_kx * ux
                H[n_k, n] += 0.5 * eta_kx * ux        # Hermitian conj (real)
            # y-direction hop
            if y + 1 < geom.Ly:
                n_k = site(x, y + 1)
                eta_ky = -1.0 if (y % 2) else 1.0      # η'_y = (-1)^y
                uy = U.U_y[t, x, y]
                H[n, n_k] += 0.5 * eta_ky * uy
                H[n_k, n] += 0.5 * eta_ky * uy
    return H


def T_F_single_cumul(geom: LatticeGeometry, U: Z2GaugeConfig,
                     a: float = 1.0) -> np.ndarray:
    """Cumulative single-particle transfer matrix:
        T̂_F^{N_E-1, single}[U] = ∏_{t=N_E-1}^{1} e^{-a · H_lat[U_t]}
    (time-ordered product, slice 0 to slice N-1).
    """
    V3 = geom.V_3
    T_total = np.eye(V3, dtype=complex)
    for t in range(geom.N_E - 1):
        H_t = build_H_lat_slice(geom, U, t)
        T_step = expm(-a * H_t)
        T_total = T_step @ T_total
    return T_total


def slater_det_matrix_element(T_single: np.ndarray,
                              occ_top: tuple, occ_bot: tuple) -> complex:
    """Slater-det matrix element ⟨Ψ_top|T̂_F^{N-1}|Ψ_bot⟩ for free fermion.

    For vacuum-to-vacuum: returns the vacuum eigenvalue product = det(I) = 1
    (factored out, conventionally; the actual vacuum factor is separate).

    For 1-particle to 1-particle at modes (a, b): T_single[a, b].

    For N-particle: det(T_single[occ_top, occ_bot]) = N × N submatrix det.
    """
    if len(occ_top) != len(occ_bot):
        return 0j
    if len(occ_top) == 0:
        return 1.0 + 0j
    sub = T_single[np.ix_(list(occ_top), list(occ_bot))]
    return complex(np.linalg.det(sub))


def fock_occupied(state_int: int, V3: int) -> tuple:
    return tuple(i for i in range(V3) if (state_int >> i) & 1)


# ----------------------------------------------------------------------------
# Test: V_3=4 single-particle vs M-K full 256-dim
# ----------------------------------------------------------------------------
def test_v3_4_single_particle_vs_mk():
    """At V_3=4 trivial U: compare Slater-det from T_F^single_cumul to
    M-K's exact T̂_F^4 matrix elements (256-dim restricted to phi=0)."""
    print("=" * 72)
    print("Phase 4 test: Slater-det from T_F^single vs M-K full T̂_F^4")
    print("=" * 72)
    import time
    from transfer_matrix_mk import (
        build_fermion_ops, mk_T_F_single_step, project_to_phi_vacuum,
    )

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    V3 = geom.V_3
    U = Z2GaugeConfig.trivial(geom)

    # M-K full 256-dim
    print(f"  Building M-K T̂_F^4 (256x256)...")
    t0 = time.time()
    chi_ops, phi_ops = build_fermion_ops(V3)
    T = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(geom.N_E - 1):
        T = mk_T_F_single_step((geom.Lx, geom.Ly), chi_ops, phi_ops,
                               U.U_x, U.U_y, t, geom.m) @ T
    T_phi0_mk = project_to_phi_vacuum(T, V3)         # 16×16 chi-Fock
    print(f"    in {time.time() - t0:.1f}s")

    # Single-particle Slater-det
    print(f"  Building T_F^{{N-1, single}} (4x4)...")
    t0 = time.time()
    T_sp = T_F_single_cumul(geom, U, a=1.0)
    print(f"    in {time.time() - t0:.3f}s")
    print(f"    T_F^single eigvals: {sorted(np.linalg.eigvals(T_sp).real)}")

    # Need vacuum factor: ⟨0|T̂_F^4|0⟩ from M-K
    N_vac = T_phi0_mk[0, 0].real
    print(f"  N_vac (from M-K) = {N_vac:.6f}")

    # Compare Slater-det predictions vs M-K direct
    print(f"\n  Compare ⟨Ψ|T̂_F^4|Ψ'⟩ for Fock pairs (M-K | Slater-det · N_vac):")
    print(f"  {'Ψ_top':>8} {'Ψ_bot':>8} {'M-K':>12} {'Slater·N_vac':>15} {'err':>10}")
    print("  " + "-" * 60)
    max_err = 0.0
    n_pairs = 0
    for psi_top in range(1 << V3):
        for psi_bot in range(1 << V3):
            occ_top = fock_occupied(psi_top, V3)
            occ_bot = fock_occupied(psi_bot, V3)
            if len(occ_top) != len(occ_bot):
                continue
            mk_val = T_phi0_mk[psi_top, psi_bot].real
            sd_val = (N_vac * slater_det_matrix_element(T_sp, occ_top, occ_bot)).real
            err = abs(mk_val - sd_val)
            max_err = max(max_err, err)
            n_pairs += 1
            if abs(mk_val) > 0.001 or abs(sd_val) > 0.001:
                print(f"  {psi_top:>8b} {psi_bot:>8b} {mk_val:>+12.6f} "
                      f"{sd_val:>+15.6f} {err:>10.2e}")
    print(f"\n  Total Fock pairs checked: {n_pairs}")
    print(f"  Max |err| across all pairs: {max_err:.2e}")
    if max_err < 1e-6:
        print(f"  PASS: T_F^single Slater-det matches M-K full computation")
    else:
        print(f"  Significant discrepancy — single-particle structure needs refinement")


if __name__ == "__main__":
    test_v3_4_single_particle_vs_mk()
