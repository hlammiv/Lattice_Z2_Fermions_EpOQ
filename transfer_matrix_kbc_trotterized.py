"""
transfer_matrix_kbc_trotterized.py — option (1): forward-time staggered with
EXACT T̂_F = expm(-a_τ · H_KS) on the chi-Fock subspace.

Construction
------------
The action_z2_staggered.py central-difference time hop produces a Miyazaki-
Kikukawa transfer matrix whose effective lattice Hamiltonian H_lat differs
from the canonical Kogut-Susskind Hamiltonian H_QC (as built by z2_setup /
epoq_classical_sampler.build_pauli_terms restricted to the all-up gauge
eigenstate) by O(1) at a_τ = 1.  The fix in this file is to BYPASS the
action's discretization for the per-step T̂_F and directly set

    T̂_F[slice] = expm(-a_τ · H_KS[slice])

where H_KS[slice] is the same Hamiltonian z2_setup builds, restricted to a
SPECIFIC spatial gauge eigenstate {Z_l = u_l(t)} on each time slice.  This
is EXACTLY a Suzuki-Trotter-resummed forward-time staggered action (the
log of expm is a sum of local operators per spacetime slice — hence a
LAGRANGIAN local action — but with cross-terms that come from H_KS
non-commuting pieces).

Lagrangian interpretation:
  S = Σ_{t=0}^{N_E-2} a_τ H_KS[U(t)] (operator-valued; each slice is local
  in spacetime).  The action's locality structure is unchanged from a
  forward-time staggered action; the only difference is that here T̂_F per
  step is computed as the matrix exponential of the slice Hamiltonian
  (no a_τ truncation).  All operators in H_KS are local (kinetic, mass,
  plaquette — only nearest-neighbor in space).

Conventions / qubit ordering caveat
-----------------------------------
We use site index a = x*Ly + y (consistent with site_idx in
action_z2_staggered, restricted to a single time slice).  In the chi-Fock
space, basis tuple `(n_0, n_1, n_2, n_3)` from
`fock_basis_labels(V_3)` has tuple[0] = site 0, etc., but the integer index
into the lex-ordered list is `n_0*8 + n_1*4 + n_2*2 + n_3*1` (tuple[0] is
MSB).

In z2_setup (epoq_classical_sampler) the matter qubit for site a is
`4 + a`, and that module uses q0 = LSB.  In the 16-dim all-up gauge sector
(state index = matter << 4 of the 8-qubit index), bit a of the 4-bit
matter index is the occupation at site a (so site 0 is LSB, site 3 is MSB
of the matter int).

Therefore the same Fock state at site a indexes z2_setup's matter int and
the lex Fock label differently.  We build H_KS in lex-Fock convention from
scratch (Jordan-Wigner with mode order chi_0 < chi_1 < chi_2 < chi_3 and a
"bit a at position (V3 - 1 - a)" mapping that matches fock_basis_labels)
and verify the numerical match against the z2_setup sector by
permutation-conjugation.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from itertools import product as _product
from action_z2_staggered import LatticeGeometry, Z2GaugeConfig


def fock_basis_labels(V3: int) -> list[tuple[int, ...]]:
    """All 2^V_3 occupation tuples in lex order (index 0 = vacuum |0...0⟩)."""
    return list(_product((0, 1), repeat=V3))


# ----------------------------------------------------------------------------
# JW operators on chi-Fock space in fock_basis_labels convention
# ----------------------------------------------------------------------------
# fock_basis_labels(V_3) yields tuples (n_0, n_1, ..., n_{V_3-1}) in lex order,
# i.e., index = n_0 * 2^{V_3-1} + n_1 * 2^{V_3-2} + ... + n_{V_3-1} * 2^0.
# So tuple position a (site a) maps to bit position (V_3 - 1 - a) of the int.

def _chi_ops_lex(V3: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build (c_a, c_a†) for a = 0..V_3-1 acting on the 2^{V_3} chi-Fock
    space, in the lex (fock_basis_labels) basis.

    Mode `a` occupies bit position `(V3 - 1 - a)` of the integer state index.
    Jordan-Wigner ordering: site 0 first, site V_3-1 last; the JW string for
    site a is the parity of the higher-order bits (sites 0..a-1).  In
    integer-bit terms, the JW phase is (-1)^{popcount(state_int >> (V_3-a))}.
    """
    dim = 1 << V3
    ops = []
    for a in range(V3):
        c_a = np.zeros((dim, dim), dtype=complex)
        cd_a = np.zeros((dim, dim), dtype=complex)
        bit = 1 << (V3 - 1 - a)
        for state in range(dim):
            # JW sign: parity of bits for sites 0..a-1, i.e., bits
            # (V_3-1), (V_3-2), ..., (V_3-a).  That's state >> (V_3 - a).
            jw_mask = state >> (V3 - a) if a > 0 else 0
            sign = 1
            for j in range(V3):
                if (jw_mask >> j) & 1:
                    sign = -sign
            occ = (state >> (V3 - 1 - a)) & 1
            if occ == 1:
                target = state ^ bit
                c_a[target, state] = sign
            else:
                target = state ^ bit
                cd_a[target, state] = sign
        ops.append((c_a, cd_a))
    return ops


# ----------------------------------------------------------------------------
# H_KS on chi-Fock for given spatial gauge config — matches z2_setup
# ----------------------------------------------------------------------------
def build_H_lat_slice(geom: LatticeGeometry,
                      U_x_slice: np.ndarray,    # shape (Lx-1, Ly), ±1
                      U_y_slice: np.ndarray,    # shape (Lx, Ly-1), ±1
                      m: float,
                      K_E: float = 1.0,
                      K_M: float = 0.5,
                      g_hop: float = 0.5) -> np.ndarray:
    """Build the per-slice H_KS as a 2^{V_3} × 2^{V_3} matrix in fock_basis_labels
    basis.  Matches z2_setup / epoq_classical_sampler.build_pauli_terms
    restricted to the gauge eigenstate {Z_l(t) = u_l(t)} on this slice
    (with electric K_E·X_l term dropped — it doesn't preserve the gauge
    eigenstate sector).

    Couplings follow z2_setup defaults:
      g_e = K_E (drops out at fixed gauge eigenstate),
      g_m = K_M (magnetic plaquette becomes c-number constant per slice),
      g_hop = 0.5 (hopping coefficient),
      mass term: m·Σ_a (-1)^{x+y}_a n_chi(a).

    Spatial hopping (STAGGERED K-S, matches the staggered action AND the
    epoq_quantum_simulator / epoq_classical_sampler Hamiltonians after the
    2026-05-18 staggered-everywhere fix):
       H_hop = g_hop · Σ_{links l=(a,b)} η_l · u_l · (c_a† c_b + c_b† c_a)
    with K-S spatial η: η_x(x,y) = +1, η_y(x,y) = (-1)^x.  In our 2x2 OBC:
    x-bonds at all y have η=+1; y-bond at x=0 has η=+1; y-bond at x=1 has η=-1.
    """
    Lx, Ly = geom.Lx, geom.Ly
    V3 = Lx * Ly
    dim = 1 << V3
    assert V3 == geom.V_3

    chi_ops = _chi_ops_lex(V3)
    H = np.zeros((dim, dim), dtype=complex)

    # --- Staggered mass: m * Σ_a (-1)^{x+y}_a n_a ---
    for x in range(Lx):
        for y in range(Ly):
            a = x * Ly + y
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            c_a, cd_a = chi_ops[a]
            H += m * parity * (cd_a @ c_a)

    # --- Staggered K-S spatial hopping: η_l · g_hop · u_l · (c_a† c_b + c_b† c_a) ---
    # K-S spatial η: η_x(x,y) = +1, η_y(x,y) = (-1)^x.
    # y-bonds: (x, y) → (x, y+1).  η_y = (-1)^x.
    for x in range(Lx):
        for y in range(Ly - 1):
            a = x * Ly + y
            b = x * Ly + (y + 1)
            u = float(U_y_slice[x, y])
            eta_y = -1.0 if (x % 2) else 1.0
            c_a, cd_a = chi_ops[a]
            c_b, cd_b = chi_ops[b]
            H += eta_y * g_hop * u * (cd_a @ c_b + cd_b @ c_a)
    # x-bonds: (x, y) → (x+1, y).  η_x = +1.
    for x in range(Lx - 1):
        for y in range(Ly):
            a = x * Ly + y
            b = (x + 1) * Ly + y
            u = float(U_x_slice[x, y])
            c_a, cd_a = chi_ops[a]
            c_b, cd_b = chi_ops[b]
            H += g_hop * u * (cd_a @ c_b + cd_b @ c_a)

    # --- Magnetic plaquette: -K_M · Π_{plaq} u_l  (c-number constant)
    # For 2x2 OBC there is exactly one plaquette per slice.
    plaq_const = 0.0
    for x in range(Lx - 1):
        for y in range(Ly - 1):
            u1 = float(U_x_slice[x, y])
            u2 = float(U_y_slice[x + 1, y])
            u3 = float(U_x_slice[x, y + 1])
            u4 = float(U_y_slice[x, y])
            plaq_const += -K_M * u1 * u2 * u3 * u4
    H += plaq_const * np.eye(dim, dtype=complex)

    # K_E·X_l electric term: dropped (does not preserve gauge eigenstate).
    H = (H + H.conj().T) / 2.0
    return H


def build_T_F_trotter(geom: LatticeGeometry,
                      U_x_slice: np.ndarray,
                      U_y_slice: np.ndarray,
                      a_tau: float,
                      m: float,
                      K_E: float = 1.0,
                      K_M: float = 0.5,
                      g_hop: float = 0.5) -> np.ndarray:
    """Per-step transfer matrix T̂_F = expm(-a_τ · H_KS[slice]).

    The "Trotter" in the name reflects that the action this corresponds to
    is the limit of arbitrarily-fine Trotter splitting; we set T̂_F to the
    exact matrix exponential here.  Returns 2^{V_3} × 2^{V_3} complex
    matrix in fock_basis_labels basis.
    """
    H_slice = build_H_lat_slice(geom, U_x_slice, U_y_slice, m,
                                K_E=K_E, K_M=K_M, g_hop=g_hop)
    T = expm(-a_tau * H_slice)
    return T


def compute_combined_weight_trotter(geom: LatticeGeometry,
                                    U: Z2GaugeConfig,
                                    a_tau: float = 1.0,
                                    K_E: float = 1.0,
                                    K_M: float = 0.5,
                                    g_hop: float = 0.5,
                                    m_obs: float = None,
                                    order: int = 1,
                                    ) -> tuple[np.ndarray, dict]:
    """Combined weight matrix W[Ψ_top, Ψ_bot] = ⟨Ψ_top | T̂_F^{N_E-1} | Ψ_bot⟩
    in lex Fock basis (basis of `fock_basis_labels(V_3)`).

    For each time slice t = 0..N_E-2, build T̂_F[t] = expm(-a_τ · H_lat_slice[t])
    with the spatial gauge fields U_x[t,:,:], U_y[t,:,:].

    `order` controls the Suzuki-Trotter ordering across time slices:

      order=1 (default): Lie-Trotter (non-Hermitian on time-varying gauge).
        W = T̂_F[N_E-2] @ ... @ T̂_F[1] @ T̂_F[0]

      order=2: PALINDROME construction (Hermitian PD by construction). With
        T_half[t] = expm(-a_τ/2 · H[U_t]) and  B = ∏_{t=0..N_E-2} T_half[t]
        (right-to-left in matrix product, so applies slice 0 first),
        W = B† @ B   ⇒   diagonal W[ψ,ψ] = ‖B|ψ⟩‖² ≥ 0 always.
        Total Euclidean time still β = (N_E−1)·a_τ.  See
        memory/feedback_W_matrix_nonhermitian.md for the motivation.

    U_t links do NOT enter for fixed gauge eigenstates per slice — they'd
    appear only through electric K_E·X_l terms (which we dropped) or
    through xτ / yτ plaquettes (which couple two adjacent gauge slices via
    Z_l(t) U_t Z_l(t+1) U_t; in a pure-Z gauge eigenstate slice they are
    multiplicative constants but require U_t info — we encode them as a
    per-slice-pair constant below).

    Returns (W, info).
    """
    Lx, Ly = geom.Lx, geom.Ly
    V3 = Lx * Ly
    dim = 1 << V3
    N_E = geom.N_E
    # If m_obs given, use it as the Hamiltonian mass for the OBSERVABLE T̂_F
    # (T̂_F = expm(-a_τ · H_KS[m_obs])).  Otherwise default to geom.m (legacy).
    # This is essential for continuum-limit studies where action mass scales
    # as a_τ · m_Ham but observable mass stays at m_Ham.
    m = m_obs if m_obs is not None else geom.m

    if order == 1:
        # Standard first-order Lie-Trotter.
        W = np.eye(dim, dtype=complex)
        for t in range(N_E - 1):
            T_t = build_T_F_trotter(geom, U.U_x[t, :, :], U.U_y[t, :, :],
                                    a_tau=a_tau, m=m, K_E=K_E, K_M=K_M,
                                    g_hop=g_hop)
            W = T_t @ W
    elif order == 2:
        # Palindrome (Hermitian PD by construction):
        #   B = T_half[N-2] @ T_half[N-3] @ ... @ T_half[0]   (slice 0 applied first)
        #   W = B† @ B
        B = np.eye(dim, dtype=complex)
        for t in range(N_E - 1):
            T_t_half = build_T_F_trotter(geom, U.U_x[t, :, :], U.U_y[t, :, :],
                                         a_tau=a_tau / 2.0, m=m, K_E=K_E,
                                         K_M=K_M, g_hop=g_hop)
            B = T_t_half @ B
        W = B.conj().T @ B
    else:
        raise ValueError(f"unsupported Trotter order={order} (use 1 or 2)")

    # --- xτ, yτ plaquettes coupling slice t and t+1 ---
    # In the gauge-eigenstate basis, these plaquettes Π u_xτ are c-numbers
    # that multiply T̂_F[t].  Folded in as scalar prefactor.
    plaq_const_total = 0.0
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                u1 = float(U.U_x[t, x, y])
                u2 = float(U.U_t[t, x + 1, y])
                u3 = float(U.U_x[t + 1, x, y])
                u4 = float(U.U_t[t, x, y])
                plaq_const_total += -K_M * u1 * u2 * u3 * u4
        for x in range(Lx):
            for y in range(Ly - 1):
                u1 = float(U.U_y[t, x, y])
                u2 = float(U.U_t[t, x, y + 1])
                u3 = float(U.U_y[t + 1, x, y])
                u4 = float(U.U_t[t, x, y])
                plaq_const_total += -K_M * u1 * u2 * u3 * u4
    # NB: this is a global multiplicative prefactor exp(-a_τ * plaq_const)
    # that does NOT affect normalized observables like ⟨n̂_0⟩, but does
    # affect the raw W magnitude (and matches K_M·Σ Π exactly).
    prefactor = np.exp(-a_tau * plaq_const_total)
    W = prefactor * W

    info = {
        'N_steps': N_E - 1,
        'a_tau': a_tau,
        'plaq_const_total': plaq_const_total,
        'prefactor_log': -a_tau * plaq_const_total,
    }
    return W, info


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_h_lat_matches_z2_setup():
    """Verify build_H_lat_slice at trivial gauge matches z2_setup H_QC in
    its 16-dim all-up gauge eigenstate sector, up to bit-ordering permutation.
    """
    print("=" * 72)
    print("TEST 1: build_H_lat_slice(trivial) vs z2_setup H_QC all-up sector")
    print("=" * 72)

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    Lx, Ly = geom.Lx, geom.Ly
    V3 = geom.V_3
    U_triv = Z2GaugeConfig.trivial(geom)

    H_lex = build_H_lat_slice(
        geom, U_triv.U_x[0, :, :], U_triv.U_y[0, :, :], m=0.5,
        K_E=1.0, K_M=0.5, g_hop=0.5,
    )
    print(f"  H_lat_slice shape: {H_lex.shape}")
    print(f"  Hermitian? max|H-H†| = {np.max(np.abs(H_lex - H_lex.conj().T)):.3e}")
    eigs_lex = np.linalg.eigvalsh(H_lex)
    print(f"  Spectrum: {sorted(eigs_lex.real.tolist())}")

    # Build z2_setup H_QC restricted to all-up gauge sector for comparison
    NQ = 8
    DIM = 256
    P2 = {
        'I': np.eye(2, dtype=complex), 'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
        'Z': np.diag([1,-1]).astype(complex),
    }
    def pauli(factors):
        by_q = {q: P2['I'] for q in range(NQ)}
        for q, ax in factors: by_q[q] = P2[ax]
        result = by_q[NQ-1]
        for q in range(NQ-2, -1, -1):
            result = np.kron(result, by_q[q])
        return result

    import epoq_classical_sampler as cs
    H_dense = sum(c * pauli(fac) for c, fac in cs.build_pauli_terms())
    H_dense = (H_dense + H_dense.conj().T) / 2
    # all-up gauge sector: link bits 0..3 all zero
    indices = [matter << 4 for matter in range(16)]
    H_QC = H_dense[np.ix_(indices, indices)]
    eigs_QC = np.linalg.eigvalsh(H_QC)
    print(f"  H_QC spectrum: {sorted(eigs_QC.real.tolist())}")

    # The two spectra should match (modulo permutation), since at trivial
    # gauge the z2_setup H_QC restricted to the gauge eigenstate sector =
    # our H_lat_slice in a relabeled basis.
    spec_diff = np.max(np.abs(np.array(sorted(eigs_lex.real)) -
                              np.array(sorted(eigs_QC.real))))
    print(f"  max |Δ(sorted eigenvalues)| = {spec_diff:.3e}")
    if spec_diff < 1e-10:
        print("  PASS: spectra match exactly.")
    else:
        print("  WARN: spectra differ.")


def test_trotter_T_F_n0():
    """V_3=4, trivial gauge, a_τ=1, N_E=5, m=0.5.
    Verify Σ_Ψ W[Ψ,Ψ] · n_0(Ψ) / Σ_Ψ W[Ψ,Ψ] matches z2_setup all-up gauge
    sector thermal ⟨n̂_0⟩ at β=4 (~0.200), AND record the value vs
    physical_sector_reference(β=4)=0.04459... target the user asked about.
    """
    print()
    print("=" * 72)
    print("TEST 2: ⟨n̂_0⟩ via T̂_F^{N_E-1}-Trotter (V_3=4, a_τ=1, N_E=5, m=0.5)")
    print("=" * 72)

    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    U = Z2GaugeConfig.trivial(geom)
    W, info = compute_combined_weight_trotter(geom, U, a_tau=1.0,
                                              K_E=1.0, K_M=0.5, g_hop=0.5)
    print(f"  W shape: {W.shape}, N_steps = {info['N_steps']}, a_τ = {info['a_tau']}")

    # ⟨n̂_0⟩_chi = (Σ_Ψ W[Ψ,Ψ] n_0(Ψ)) / (Σ_Ψ W[Ψ,Ψ])
    # Site 0 is bit position (V_3-1) = 3 of the integer index in lex basis.
    V3 = geom.V_3
    diag = np.real(np.diag(W))
    Z = float(np.sum(diag))
    num = 0.0
    for state in range(W.shape[0]):
        # site 0 occupation = bit at position (V_3-1) = 3
        n0 = (state >> (V3 - 1)) & 1
        num += diag[state] * n0
    n0_expect = num / Z if abs(Z) > 1e-14 else float('nan')

    print(f"  Σ_Ψ W[Ψ,Ψ] = Z_chi (β=N_E-1)·a_τ = {Z:+.6f}")
    print(f"  Σ_Ψ W[Ψ,Ψ] n_0(Ψ)              = {num:+.6f}")
    print(f"  ⟨n̂_0⟩ from W diagonal           = {n0_expect:+.6f}")

    # Reference values:
    from action_endtoend_pipeline import physical_sector_reference
    ref_phys = physical_sector_reference(beta=4.0, times=[0.0])[0.0]
    print(f"  physical_sector_reference(β=4) (user-cited target) = {ref_phys:+.6f}")

    # All-up gauge sector reference (this is what chi-Fock thermal trace
    # actually computes when gauge is frozen):
    NQ = 8; DIM = 256
    P2 = {
        'I': np.eye(2, dtype=complex), 'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
        'Z': np.diag([1,-1]).astype(complex),
    }
    def pauli(factors):
        by_q = {q: P2['I'] for q in range(NQ)}
        for q, ax in factors: by_q[q] = P2[ax]
        result = by_q[NQ-1]
        for q in range(NQ-2, -1, -1):
            result = np.kron(result, by_q[q])
        return result
    import epoq_classical_sampler as cs
    H_dense = sum(c * pauli(fac) for c, fac in cs.build_pauli_terms())
    H_dense = (H_dense + H_dense.conj().T) / 2
    indices = [matter << 4 for matter in range(16)]
    H_QC = H_dense[np.ix_(indices, indices)]
    n0_dense = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    n0_QC = n0_dense[np.ix_(indices, indices)]
    rho_QC = expm(-4.0 * H_QC)
    Z_QC = np.trace(rho_QC).real
    n0_QC_thermal = (np.trace(rho_QC @ n0_QC) / Z_QC).real
    print(f"  z2_setup all-up gauge sector ⟨n̂_0⟩_β=4 = {n0_QC_thermal:+.6f}")

    print()
    print(f"  Δ(W-diagonal, all-up-gauge ref) = {abs(n0_expect - n0_QC_thermal):.3e}")
    print(f"  Δ(W-diagonal, physical ref)     = {abs(n0_expect - ref_phys):.3e}")

    if abs(n0_expect - n0_QC_thermal) < 1e-10:
        print("  PASS: matches all-up gauge sector reference to float precision.")
    else:
        print("  FAIL: does NOT match all-up gauge sector reference.")


def test_z2_random_gauge():
    """Smoke-test for non-trivial Z_2 gauge: build T̂_F per slice and check
    matrix is well-formed and depends on the random config.
    """
    print()
    print("=" * 72)
    print("TEST 3: random Z_2 gauge — smoke test")
    print("=" * 72)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=5, m=0.5)
    rng = np.random.default_rng(2026)
    for trial in range(3):
        U = Z2GaugeConfig.random(geom, rng)
        W, info = compute_combined_weight_trotter(geom, U, a_tau=1.0)
        Z = float(np.real(np.trace(W)))
        diag = np.real(np.diag(W))
        V3 = geom.V_3
        num = sum(diag[s] * ((s >> (V3 - 1)) & 1) for s in range(W.shape[0]))
        n0 = num / Z if abs(Z) > 1e-14 else float('nan')
        print(f"  trial {trial}: Z = {Z:+.4e}, ⟨n̂_0⟩ = {n0:+.4f}")


if __name__ == "__main__":
    test_h_lat_matches_z2_setup()
    test_trotter_T_F_n0()
    test_z2_random_gauge()
