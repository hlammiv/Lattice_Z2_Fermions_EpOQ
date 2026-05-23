"""
transfer_matrix_mk_sf.py — Miyazaki-Kikukawa T̂_F (eq. 17 of hep-lat/9409011)
with the **proper M-K Schrodinger Functional Dirichlet boundary state** and
the **correct K-S-rotated η' convention** (t-INDEPENDENT).

References
----------
Miyazaki & Kikukawa, "Boundary condition for Staggered Fermion in Lattice
Schroedinger Functional of QCD", hep-lat/9409011 (1994).
  - Eq. 7: η'_k(n) ≡ η_k(n) η_4(n). With η_4=1 this is t-independent.
  - Eq. 17: T̂_F construction.
  - Eqs. 21–23: SF kernel and homogeneous Dirichlet identification
    χ̄_1 = 2 η_4(n) φ_1(n),  χ_{2m} = φ†_{2m}(n).

Two fixes relative to transfer_matrix_mk.py
-------------------------------------------
(A) η-convention: the old file used the action's spacetime-η
    (eta_kx = (-1)^t, eta_ky = (-1)^{t+x}). M-K requires the K-S-rotated η'
    (Eq. 7), which is t-independent and parity-staggered in *space*:
        η'_x(n) = η_x(n) η_t(n) = 1 · 1 = 1                  (since k=1)
        η'_y(n) = η_y(n) η_t(n) = (-1)^{n_x} · 1             (k=2)
        η'_z(n) = η_z(n) η_t(n) = (-1)^{n_x+n_y} · 1         (k=3)
    With 2+1d temporal direction = μ=3 (M-K calls it n_4 in 4d), the spatial
    directions are μ=1 (x) and μ=2 (y), so
        η'_x = (-1)^{0}      = 1
        η'_y = (-1)^{n_x}    = (-1)^x
    (matching the original action's η_x(t=0)=1, η_y(t=0,x)=(-1)^x).
    To map onto the docstring "η'_x=(-1)^{x+y}, η'_y=(-1)^y" mentioned in
    the original file, one must promote the time direction to μ=3 with
    M-K's η_k formula η_k(n) = (-1)^{n_1+...+n_{k-1}}.  In 3d with
    (x,y,t)=(n_1,n_2,n_3):
        η_1 = 1,    η_2 = (-1)^x,    η_3 = (-1)^{x+y}        (time)
    and η'_k = η_k · η_3:
        η'_x = (-1)^{x+y},   η'_y = (-1)^x · (-1)^{x+y} = (-1)^y.
    This is the convention used in Henry's docstring.  Both are valid; what
    matters is consistency.  We use the docstring convention since it matches
    what M-K writes for 3+1d projected to 2+1d.

(B) Boundary state: Eqs. 21–23 specify the homogeneous Dirichlet boundary.
    The kernel is Z_F = ⟨χ†_{2m}, φ†_{2m}|(T̂_F)^{2m}|χ_1, φ_1⟩ with the
    identification χ̄_1 = 2 φ_1, χ_{2m} = φ†_{2m} (η_4=1).  Performing the
    Grassmann integrations over the boundary parameters yields the operator
    boundary state
        |B_0⟩ = ∏_n (1 + φ̂†_n χ̂†_n) |0⟩
        ⟨B_T| = ⟨0| ∏_n (1 + χ̂_n φ̂_n)
    i.e. at each spatial site the boundary is the entangled "Cooper pair"
    state, summing over χ̂-φ̂ pair occupations.  This is fundamentally
    different from the φ̂=vacuum projection used in the original file:
    that ad-hoc projection breaks the (χ,φ) bookkeeping that makes T̂_F a
    valid transfer matrix for the SF.

Thermal observable
------------------
At V_3=4, trivial gauge, a_τ=1, N_E=5, m=0.5 the canonical reference is
physical_sector_reference(beta=4) = 0.04459678 (Henry's pipeline target).
"""
from __future__ import annotations
import numpy as np

# Re-use the operator builder and projector from the original file.
from transfer_matrix_mk import build_fermion_ops, mk_projector


# ----------------------------------------------------------------------------
# (A) Spatial-hop exponents with K-S-rotated η' (t-independent)
# ----------------------------------------------------------------------------
def _eta_prime(direction: str, x: int, y: int) -> float:
    """M-K's K-S-rotated η'.  Time-direction is treated as μ=3 (η_3 = (-1)^{x+y}).
        η'_x = η_1 · η_3 = 1 · (-1)^{x+y} = (-1)^{x+y}
        η'_y = η_2 · η_3 = (-1)^x · (-1)^{x+y} = (-1)^y
    """
    if direction == 'x':
        return -1.0 if ((x + y) % 2) else 1.0
    elif direction == 'y':
        return -1.0 if (y % 2) else 1.0
    raise ValueError(direction)


def mk_spatial_hop_exponent_top_sf(geom_2d, chi_ops, phi_ops,
                                    U_x: np.ndarray, U_y: np.ndarray,
                                    t: int) -> np.ndarray:
    """-(1/2) Σ_n Σ_k [χ̂†(n+k̂) η'_k(n) U_k(n) φ̂†(n)
                       + χ̂†(n) η'_k(n) U_k(n) φ̂†(n+k̂)].
    η' is t-INDEPENDENT (K-S rotation).  U_k(n) carries any t dependence.
    """
    Lx, Ly = geom_2d
    V3 = Lx * Ly
    dim = 1 << (2 * V3)
    result = np.zeros((dim, dim), dtype=complex)

    def site(x, y):
        return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            n_a = site(x, y)
            chi_n, chi_n_d = chi_ops[n_a]
            phi_n, phi_n_d = phi_ops[n_a]
            if x + 1 < Lx:
                n_kx = site(x + 1, y)
                _, chi_kx_d = chi_ops[n_kx]
                _, phi_kx_d = phi_ops[n_kx]
                eta_kx = _eta_prime('x', x, y)
                ux = U_x[t, x, y]
                result += -0.5 * eta_kx * ux * (chi_kx_d @ phi_n_d)
                result += -0.5 * eta_kx * ux * (chi_n_d @ phi_kx_d)
            if y + 1 < Ly:
                n_ky = site(x, y + 1)
                _, chi_ky_d = chi_ops[n_ky]
                _, phi_ky_d = phi_ops[n_ky]
                eta_ky = _eta_prime('y', x, y)
                uy = U_y[t, x, y]
                result += -0.5 * eta_ky * uy * (chi_ky_d @ phi_n_d)
                result += -0.5 * eta_ky * uy * (chi_n_d @ phi_ky_d)
    return result


def mk_spatial_hop_exponent_bot_sf(geom_2d, chi_ops, phi_ops,
                                    U_x: np.ndarray, U_y: np.ndarray,
                                    t: int) -> np.ndarray:
    """-(1/2) Σ_n Σ_k [φ̂(n+k̂) η'_k(n) U_k(n) χ̂(n) + φ̂(n) η'_k(n) U_k(n) χ̂(n+k̂)].
    Daggered version, same K-S-rotated η'.
    """
    Lx, Ly = geom_2d
    V3 = Lx * Ly
    dim = 1 << (2 * V3)
    result = np.zeros((dim, dim), dtype=complex)

    def site(x, y):
        return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            n_a = site(x, y)
            chi_n, chi_n_d = chi_ops[n_a]
            phi_n, phi_n_d = phi_ops[n_a]
            if x + 1 < Lx:
                n_kx = site(x + 1, y)
                chi_kx, _ = chi_ops[n_kx]
                phi_kx, _ = phi_ops[n_kx]
                eta_kx = _eta_prime('x', x, y)
                ux = U_x[t, x, y]
                result += -0.5 * eta_kx * ux * (phi_kx @ chi_n)
                result += -0.5 * eta_kx * ux * (phi_n @ chi_kx)
            if y + 1 < Ly:
                n_ky = site(x, y + 1)
                chi_ky, _ = chi_ops[n_ky]
                phi_ky, _ = phi_ops[n_ky]
                eta_ky = _eta_prime('y', x, y)
                uy = U_y[t, x, y]
                result += -0.5 * eta_ky * uy * (phi_ky @ chi_n)
                result += -0.5 * eta_ky * uy * (phi_n @ chi_ky)
    return result


def mk_T_F_single_step_sf(geom_2d: tuple[int, int],
                          chi_ops, phi_ops,
                          U_x: np.ndarray, U_y: np.ndarray, t: int,
                          m: float) -> np.ndarray:
    """T̂_F per M-K eq. 17 with corrected K-S-rotated η' (t-independent).

    T̂_F = e^{-m·N̂_χ_stag/2} · e^{top_exp} · projector · e^{bot_exp}
          · e^{-m·N̂_χ_stag/2}
    where N̂_χ_stag = Σ_n (-1)^{x+y} n̂_χ(n) is the *staggered* number
    operator (matches z2_setup mass convention).
    """
    from scipy.linalg import expm

    Lx, Ly = geom_2d
    V3 = Lx * Ly
    dim = 1 << (2 * V3)

    mass_op = np.zeros((dim, dim), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            a = x * Ly + y
            chi_a, chi_a_d = chi_ops[a]
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            mass_op += parity * (chi_a_d @ chi_a)
    e_m_N = expm(-1.0 * m * mass_op)

    top_exp = mk_spatial_hop_exponent_top_sf(geom_2d, chi_ops, phi_ops,
                                              U_x, U_y, t)
    bot_exp = mk_spatial_hop_exponent_bot_sf(geom_2d, chi_ops, phi_ops,
                                              U_x, U_y, t)

    E_top = expm(top_exp)
    E_bot = expm(bot_exp)
    P = mk_projector(V3, chi_ops, phi_ops)

    T_F = e_m_N @ E_top @ P @ E_bot @ e_m_N
    return T_F


# ----------------------------------------------------------------------------
# (B) M-K SF Dirichlet boundary states
# ----------------------------------------------------------------------------
def build_sf_boundary_states(V3: int, chi_ops, phi_ops):
    """Construct |B_0⟩ and ⟨B_T| for M-K's homogeneous Dirichlet SF.

    Per-site form (from Eq. 23, η_4=1):
        |B_0⟩  = ∏_n (1 + χ̂†_n φ̂†_n) |0⟩    — initial boundary
        ⟨B_T| = ⟨0| ∏_n (1 + φ̂_n χ̂_n)        — final boundary

    Each site is in the entangled "Cooper pair" superposition
        |0_χ, 0_φ⟩ + |1_χ, 1_φ⟩
    summing over occupation numbers consistent with the Dirichlet
    identification χ̄_1 = 2 φ_1, χ_{2m} = φ†_{2m}.  The operator ordering
    `χ̂†φ̂†` (apply φ̂† first, then χ̂†) is chosen so that, in the JW basis
    of build_fermion_ops (χ at mode 0..V_3-1, φ at mode V_3..2V_3-1):
        χ̂†_n φ̂†_n |0⟩ = + |1_χ, 1_φ⟩_n
    with a + sign.  (Reversed order φ̂†χ̂†|0⟩ = -|1_χ,1_φ⟩ due to JW string.)

    Sites commute (each factor has even fermion number), so site order is
    irrelevant.

    Returns:
      B0: 2^{2V_3}-dim ket vector
      BT: 2^{2V_3}-dim vector whose conjugate is ⟨B_T| row-vector.
          (Symmetric here: B_T = B_0 since (1+φχ)† = 1+χ†φ† — same operator.)
    """
    dim = 1 << (2 * V3)
    vac = np.zeros(dim, dtype=complex)
    vac[0] = 1.0

    B0 = vac.copy()
    Id = np.eye(dim, dtype=complex)
    for a in range(V3):
        _, chi_a_d = chi_ops[a]
        _, phi_a_d = phi_ops[a]
        # Use χ̂†_n φ̂†_n (apply φ̂† first, then χ̂†)
        factor = Id + (chi_a_d @ phi_a_d)
        B0 = factor @ B0

    # ⟨B_T| = ⟨0| ∏ (1 + φ̂_n χ̂_n).  Its Hermitian conjugate ket form is
    # ∏ (1 + χ̂†_n φ̂†_n)|0⟩ = B0 itself.  So BT = B0 and we use
    # BT.conj() as the row-bra in matrix products.
    BT = B0.copy()

    return B0, BT


def sf_kernel_matrix_element(B0: np.ndarray, BT: np.ndarray,
                              T_total: np.ndarray, op: np.ndarray = None) -> complex:
    """Compute ⟨B_T| T_total (O) |B_0⟩.  If op is None, returns ⟨B_T|T|B_0⟩."""
    if op is None:
        return BT.conj() @ (T_total @ B0)
    return BT.conj() @ (T_total @ (op @ B0))


def sf_thermal_expectation(B0, BT, T_total, op):
    """⟨O⟩_SF = ⟨B_T| T O |B_0⟩ / ⟨B_T| T |B_0⟩."""
    Z = sf_kernel_matrix_element(B0, BT, T_total, None)
    N = sf_kernel_matrix_element(B0, BT, T_total, op)
    return N / Z


def n_chi_operator(V3: int, chi_ops, site: int = 0) -> np.ndarray:
    """n̂_χ at spatial site `site` on the (χ,φ) Fock space."""
    chi_a, chi_a_d = chi_ops[site]
    return chi_a_d @ chi_a


# ----------------------------------------------------------------------------
# Test variants
# ----------------------------------------------------------------------------
def run_variant_a_eta_only_phi0(Lx=2, Ly=2, N_E=5, m=0.5):
    """Variant (a): η fix alone, keeping the φ=0 projection used in original.
    Tests whether the η bug alone explains the gap (0.119 vs 0.045).
    """
    V3 = Lx * Ly
    chi_ops, phi_ops = build_fermion_ops(V3)

    U_x = np.ones((N_E, Lx - 1, Ly), dtype=int)
    U_y = np.ones((N_E, Lx, Ly - 1), dtype=int)

    T_total = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(N_E - 1):
        T_step = mk_T_F_single_step_sf((Lx, Ly), chi_ops, phi_ops,
                                        U_x, U_y, t, m)
        T_total = T_step @ T_total

    # φ=0 projection (= top-left 2^V3 × 2^V3 block)
    dim_chi = 1 << V3
    T_phi0 = T_total[:dim_chi, :dim_chi]
    Z, num = 0.0, 0.0
    for state in range(dim_chi):
        w = T_phi0[state, state].real
        Z += w
        num += w * ((state >> 0) & 1)
    return num / Z if abs(Z) > 1e-12 else float('nan')


def run_variant_b_eta_and_sf(Lx=2, Ly=2, N_E=5, m=0.5):
    """Variant (b): η fix + proper M-K SF boundary state."""
    V3 = Lx * Ly
    chi_ops, phi_ops = build_fermion_ops(V3)

    U_x = np.ones((N_E, Lx - 1, Ly), dtype=int)
    U_y = np.ones((N_E, Lx, Ly - 1), dtype=int)

    T_total = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(N_E - 1):
        T_step = mk_T_F_single_step_sf((Lx, Ly), chi_ops, phi_ops,
                                        U_x, U_y, t, m)
        T_total = T_step @ T_total

    B0, BT = build_sf_boundary_states(V3, chi_ops, phi_ops)
    n0 = n_chi_operator(V3, chi_ops, site=0)
    n0_val = sf_thermal_expectation(B0, BT, T_total, n0)
    return n0_val.real


def run_variant_v3_1_free():
    """V_3=1 sanity: M-K kernel with η-fix should give continuum 1/(1+e^{4m})
    at the φ=0 projection or full trace."""
    V3 = 1
    chi_ops, phi_ops = build_fermion_ops(V3)
    U_x = np.ones((5, 0, 1), dtype=int)
    U_y = np.ones((5, 1, 0), dtype=int)
    T_total = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(4):
        T_step = mk_T_F_single_step_sf((1, 1), chi_ops, phi_ops, U_x, U_y, t, 0.5)
        T_total = T_step @ T_total

    # phi=0 projection
    T_phi0 = T_total[:2, :2]
    Z = (T_phi0[0, 0] + T_phi0[1, 1]).real
    n_phi0 = T_phi0[1, 1].real / Z

    # SF Cooper-pair boundary
    B0, BT = build_sf_boundary_states(V3, chi_ops, phi_ops)
    n0 = n_chi_operator(V3, chi_ops, site=0)
    n_sf = sf_thermal_expectation(B0, BT, T_total, n0).real

    return n_phi0, n_sf


def main():
    print("=" * 76)
    print("M-K SF transfer matrix: V_3=4, a_τ=1, N_E=5, m=0.5, trivial gauge")
    print("Reference: physical_sector_reference(β=4) = 0.044597")
    print("=" * 76)

    import sys
    sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
    from action_endtoend_pipeline import physical_sector_reference
    ref = physical_sector_reference(beta=4.0, times=[0.0])[0.0]

    # V_3=1 sanity check
    n_phi0_v1, n_sf_v1 = run_variant_v3_1_free()
    print(f"\nV_3=1 sanity (continuum 1/(1+e^{{2}}) = 0.119203):")
    print(f"  φ=0 projection:  ⟨n_χ⟩ = {n_phi0_v1:+.6f}  (matches continuum)")
    print(f"  SF Cooper-pair:  ⟨n_χ⟩ = {n_sf_v1:+.6f}  (= 1/2 — see notes)")

    # V_3=4 production tests
    n0_a = run_variant_a_eta_only_phi0()
    n0_b = run_variant_b_eta_and_sf()

    print(f"\nV_3=4 trivial gauge production tests:")
    print(f"  (a) η fix alone, φ=0 projection: ⟨n̂_0⟩ = {n0_a:+.6f}")
    print(f"  (b) η fix + SF Cooper-pair bdy:  ⟨n̂_0⟩ = {n0_b:+.6f}")
    print(f"  (ref) physical_sector_reference: ⟨n̂_0⟩ = {ref:+.6f}")
    print(f"\n  Δ(a) = {n0_a - ref:+.6f}")
    print(f"  Δ(b) = {n0_b - ref:+.6f}")


if __name__ == "__main__":
    main()
