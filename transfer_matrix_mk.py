"""
transfer_matrix_mk.py — Miyazaki-Kikukawa Hermitian staggered T̂_F (eq. 17).

This is Phase 6 of the P2 pipeline: replace Agent A's per-step (B-R)(B+R)^{-1}
formula with the proper M-K construction with auxiliary field φ̂.  The
M-K T̂_F is Hermitian per step (T̂_F not positive-definite; (T̂_F)² IS).
This gives real-valued thermal expectations for arbitrary Z₂ gauge configs.

For V_3 spatial sites, the Fock space has 2V_3 fermion modes (V_3 χ̂ + V_3 φ̂),
dimension 2^{2V_3}.  For V_3 = 4: 256-dim, 256×256 matrices — tractable.

M-K eq. 17 (massless free):
  T̂_F = exp{ -1/2 Σ_n Σ_{k=1}^d [χ̂†(n+k̂) η'_k(n) φ̂†(n) + χ̂†(n) η'_k(n) φ̂†(n+k̂)] }
        × ∏_n :(φ̂†(n) - χ̂(n))(φ̂(n) - χ̂†(n)):
        × exp{ -1/2 Σ_n Σ_{k=1}^d [φ̂(n+k̂) η'_k(n) χ̂(n) + φ̂(n) η'_k(n) χ̂(n+k̂)] }

Mass: include e^{-m·n̂_χ} split factors (half on each end of T̂_F).
Gauge: dress hops with U_k(n) factors (per Caracciolo-Palumbo p.14:
"no essential difference in constructing the transfer matrix").

OBC convention: kernel = T̂_F^{N_E-1}.  For (T̂_F)² positive-def → need
N_E-1 even, i.e., N_E ODD.

Project boundaries to |φ̂=vacuum⟩ to extract effective χ̂-only matrix elements.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Fermion operators on 2V_3-mode Fock space
# ----------------------------------------------------------------------------
def build_fermion_ops(V3: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build χ̂_a, χ̂_a†, φ̂_a, φ̂_a† for a = 0..V_3-1 as 2^{2V_3} × 2^{2V_3} matrices.

    Mode ordering (Jordan-Wigner): χ_0, χ_1, ..., χ_{V_3-1}, φ_0, ..., φ_{V_3-1}.
    Mode index k = a (for χ̂_a) or V_3 + a (for φ̂_a), 0 ≤ k < 2V_3.

    Returns (chi_ops, phi_ops) where each is a list of (annihilation, creation)
    pairs:  chi_ops[a] = (χ̂_a, χ̂_a†),  phi_ops[a] = (φ̂_a, φ̂_a†).
    """
    n_modes = 2 * V3
    dim = 1 << n_modes
    # Build all c_k, c_k† via Jordan-Wigner
    c_ops = []        # (annihilation, creation) for each mode k
    for k in range(n_modes):
        c_k = np.zeros((dim, dim), dtype=complex)
        cd_k = np.zeros((dim, dim), dtype=complex)
        for state in range(dim):
            occ = (state >> k) & 1
            # JW sign: (-1)^{# fermions in modes 0..k-1}
            sign = 1
            for j in range(k):
                if (state >> j) & 1:
                    sign = -sign
            if occ == 1:
                target = state ^ (1 << k)
                c_k[target, state] = sign       # annihilation
            else:
                target = state ^ (1 << k)
                cd_k[target, state] = sign      # creation
        c_ops.append((c_k, cd_k))

    chi_ops = [c_ops[a] for a in range(V3)]
    phi_ops = [c_ops[V3 + a] for a in range(V3)]
    return chi_ops, phi_ops


# ----------------------------------------------------------------------------
# M-K T̂_F at one time slice
# ----------------------------------------------------------------------------
def mk_projector(V3: int, chi_ops, phi_ops) -> np.ndarray:
    """Build ∏_n :(φ̂†(n) - χ̂(n))(φ̂(n) - χ̂†(n)): for n = 0..V_3-1.

    Per site n, the factor is :(φ̂_n† - χ̂_n)(φ̂_n - χ̂_n†):.
    Expand:  :φ̂_n†φ̂_n - φ̂_n†χ̂_n† - χ̂_nφ̂_n + χ̂_nχ̂_n†:
          = n̂_φ,n - φ̂_n†χ̂_n† - χ̂_nφ̂_n - n̂_χ,n   (normal-ordering :χ̂χ̂†: = -n̂_χ)
    Per-site factors commute as long as operators at different sites
    anticommute consistently (they do, via JW).
    Returns the projector as a 2^{2V_3} × 2^{2V_3} matrix.
    """
    dim = 1 << (2 * V3)
    P_total = np.eye(dim, dtype=complex)
    for a in range(V3):
        chi_a, chi_a_d = chi_ops[a]
        phi_a, phi_a_d = phi_ops[a]
        n_chi_a = chi_a_d @ chi_a
        n_phi_a = phi_a_d @ phi_a
        factor = (n_phi_a - phi_a_d @ chi_a_d - chi_a @ phi_a - n_chi_a)
        P_total = P_total @ factor
    return P_total


def mk_spatial_hop_exponent_top(geom_2d, chi_ops, phi_ops,
                                U_x: np.ndarray, U_y: np.ndarray,
                                t: int) -> np.ndarray:
    """Build (-1/2) Σ_n Σ_k [χ̂†(n+k̂) η'_k(n) U_k(n) φ̂†(n) + χ̂†(n) η'_k(n) U_k(n) φ̂†(n+k̂)].

    η'_k(n) = η_k(n) · η_4(n) per M-K eq. 7.  For our staggered convention
    η_4(n) = 1; η_k(n) depends on the lattice coords at time slice t.

    geom_2d = (Lx, Ly).  Site index a = x·Ly + y.
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
            # M-K's η'_k convention (t-INDEPENDENT for 2+1d):
            #   η'_1 = η_1 · η_3 = 1·(-1)^{x+y} = (-1)^{x+y}
            #   η'_2 = η_2 · η_3 = (-1)^x·(-1)^{x+y} = (-1)^y
            # Our action_z2_staggered uses different η convention (time-major)
            # related by Kawamoto-Smit transformation — same det M, different
            # T̂_F operator structure.  M-K's convention gives Hermitian T̂_F
            # per step that's also identical across t for trivial gauge.
            if x + 1 < Lx:
                n_kx = site(x + 1, y)
                chi_kx, chi_kx_d = chi_ops[n_kx]
                phi_kx, phi_kx_d = phi_ops[n_kx]
                # To match z2_setup (naive fermion hopping, no staggered phases):
                # set η' = 1 for all directions.
                eta_kx = 1.0 if (t % 2 == 0) else -1.0  # η_x = (-1)^t
                ux = U_x[t, x, y]
                result += -0.5 * eta_kx * ux * (chi_kx_d @ phi_n_d)
                result += -0.5 * eta_kx * ux * (chi_n_d @ phi_kx_d)
            if y + 1 < Ly:
                n_ky = site(x, y + 1)
                chi_ky, chi_ky_d = chi_ops[n_ky]
                phi_ky, phi_ky_d = phi_ops[n_ky]
                eta_ky = -1.0 if ((t + x) % 2) else 1.0  # η_y = (-1)^{t+x}
                uy = U_y[t, x, y]
                result += -0.5 * eta_ky * uy * (chi_ky_d @ phi_n_d)
                result += -0.5 * eta_ky * uy * (chi_n_d @ phi_ky_d)
    return result


def mk_spatial_hop_exponent_bot(geom_2d, chi_ops, phi_ops,
                                U_x: np.ndarray, U_y: np.ndarray,
                                t: int) -> np.ndarray:
    """Build (-1/2) Σ_n Σ_k [φ̂(n+k̂) η'_k(n) U_k(n) χ̂(n) + φ̂(n) η'_k(n) U_k(n) χ̂(n+k̂)].

    Daggered version of top exponent.
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
            # Same M-K η' convention as in top exponent.
            if x + 1 < Lx:
                n_kx = site(x + 1, y)
                chi_kx, chi_kx_d = chi_ops[n_kx]
                phi_kx, phi_kx_d = phi_ops[n_kx]
                eta_kx = 1.0 if (t % 2 == 0) else -1.0  # η_x = (-1)^t
                ux = U_x[t, x, y]
                result += -0.5 * eta_kx * ux * (phi_kx @ chi_n)
                result += -0.5 * eta_kx * ux * (phi_n @ chi_kx)
            if y + 1 < Ly:
                n_ky = site(x, y + 1)
                chi_ky, chi_ky_d = chi_ops[n_ky]
                phi_ky, phi_ky_d = phi_ops[n_ky]
                eta_ky = -1.0 if ((t + x) % 2) else 1.0  # η_y = (-1)^{t+x}
                uy = U_y[t, x, y]
                result += -0.5 * eta_ky * uy * (phi_ky @ chi_n)
                result += -0.5 * eta_ky * uy * (phi_n @ chi_ky)
    return result


def mk_T_F_single_step(geom_2d: tuple[int, int],
                       chi_ops, phi_ops,
                       U_x: np.ndarray, U_y: np.ndarray, t: int,
                       m: float) -> np.ndarray:
    """Build T̂_F per M-K eq. 17 at time slice t with mass m and Z_2 gauge.

    T̂_F = e^{-m·N̂_χ/2} · e^{top_exp} · projector · e^{bot_exp} · e^{-m·N̂_χ/2}
    where N̂_χ = Σ_n n̂_χ(n) is the total χ̂ number operator.
    Mass split form preserves Hermiticity.
    """
    from scipy.linalg import expm

    Lx, Ly = geom_2d
    V3 = Lx * Ly
    dim = 1 << (2 * V3)

    # STAGGERED mass: m·(-1)^(x+y)·n_chi(a) — matches z2_setup convention.
    # Each spatial site gets parity-dependent sign.
    mass_op = np.zeros((dim, dim), dtype=complex)
    for x in range(Lx):
        for y in range(Ly):
            a = x * Ly + y
            chi_a, chi_a_d = chi_ops[a]
            parity = 1.0 if (x + y) % 2 == 0 else -1.0
            mass_op += parity * (chi_a_d @ chi_a)
    e_m_N = expm(-1.0 * m * mass_op)

    top_exp = mk_spatial_hop_exponent_top(geom_2d, chi_ops, phi_ops, U_x, U_y, t)
    bot_exp = mk_spatial_hop_exponent_bot(geom_2d, chi_ops, phi_ops, U_x, U_y, t)

    E_top = expm(top_exp)
    E_bot = expm(bot_exp)
    P = mk_projector(V3, chi_ops, phi_ops)

    T_F = e_m_N @ E_top @ P @ E_bot @ e_m_N
    return T_F


def project_to_phi_vacuum(matrix: np.ndarray, V3: int) -> np.ndarray:
    """Extract the 2^{V_3} × 2^{V_3} sub-matrix where φ̂-sector is |0⟩.

    In our Jordan-Wigner basis with φ̂ modes at indices V_3..2V_3-1:
    states with φ̂=vacuum have bits V_3..2V_3-1 all zero → indices 0..2^{V_3}-1.
    """
    dim_chi = 1 << V3
    return matrix[:dim_chi, :dim_chi]


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_v3_1_free():
    """V_3 = 1, free (no gauge), m = 0.5, N_E = 5.
    Expected: ⟨n̂_χ⟩ ≈ 1/(1+e^2) ≈ 0.119 (continuum).
    """
    print("=" * 72)
    print("M-K T̂_F test: V_3 = 1 free, m = 0.5, N_E = 5")
    print("=" * 72)

    V3 = 1
    Lx, Ly = 1, 1
    N_E = 5
    m = 0.5

    chi_ops, phi_ops = build_fermion_ops(V3)

    # Dummy gauge (no spatial hops since Lx=Ly=1)
    U_x = np.ones((N_E, 0, 1), dtype=int)
    U_y = np.ones((N_E, 1, 0), dtype=int)

    # Build T̂_F at each time slice and multiply
    T_total = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(N_E - 1):
        T_step = mk_T_F_single_step((Lx, Ly), chi_ops, phi_ops, U_x, U_y, t, m)
        T_total = T_step @ T_total

    print(f"  T̂_F^{N_E-1} dimensions: {T_total.shape}")
    print(f"  Hermitian? max|T - T†| = {np.max(np.abs(T_total - T_total.conj().T)):.4e}")
    eigs = np.linalg.eigvals(T_total)
    print(f"  Eigvals (real parts): {sorted(eigs.real)}")
    print(f"  Max |imag eigval|: {np.max(np.abs(eigs.imag)):.4e}")

    # Project to phi = 0 boundary
    T_phi0 = project_to_phi_vacuum(T_total, V3)
    print(f"  T̂_F^{N_E-1} on χ̂-only (phi=0 BC):")
    print(f"    {T_phi0.real.round(6)}")

    # Compute ⟨n̂_χ⟩ in phi=0 subspace
    # Basis: |chi=0⟩, |chi=1⟩.  n̂_χ = diag(0, 1).
    Z = T_phi0[0, 0] + T_phi0[1, 1]
    num = T_phi0[1, 1]  # only |chi=1⟩ has n_chi=1
    n_chi = (num / Z).real
    print(f"\n  Z = Tr[T̂_F^4]_{{φ=0}} = {Z.real:+.6f}")
    print(f"  Tr[T̂_F^4 n̂_χ]_{{φ=0}} = {num.real:+.6f}")
    print(f"  ⟨n̂_χ⟩_MK = {n_chi:+.6f}")
    continuum = 1.0 / (1.0 + np.exp(4 * m))
    print(f"  continuum 1/(1+e^(β·m)) at β=4 m=0.5 = {continuum:+.6f}")
    print(f"  |M-K − continuum| = {abs(n_chi - continuum):.4e}")


def test_v3_4_free():
    """V_3 = 4 free (Lx=Ly=2, trivial gauge), m = 0.5, N_E = 5.
    Build 256x256 T̂_F, verify Hermitian per step, compute ⟨n̂_0⟩ at β=4.
    Compare to z2_setup reference at K=0 (gauge-decoupled limit).
    """
    print()
    print("=" * 72)
    print("M-K T̂_F test: V_3 = 4 free, m = 0.5, N_E = 5 (Lx=Ly=2)")
    print("=" * 72)

    Lx, Ly = 2, 2
    V3 = Lx * Ly
    N_E = 5
    m = 0.5
    import time

    t0 = time.time()
    chi_ops, phi_ops = build_fermion_ops(V3)
    print(f"  built fermion ops on dim={1 << (2 * V3)} space in {time.time()-t0:.1f}s")

    # Trivial gauge
    U_x = np.ones((N_E, Lx - 1, Ly), dtype=int)
    U_y = np.ones((N_E, Lx, Ly - 1), dtype=int)

    # Build T̂_F^{N-1} (each step has different η_k due to t-alternation)
    t0 = time.time()
    T_total = np.eye(1 << (2 * V3), dtype=complex)
    for t in range(N_E - 1):
        T_step = mk_T_F_single_step((Lx, Ly), chi_ops, phi_ops, U_x, U_y, t, m)
        # Verify per-step Hermiticity
        herm_err = np.max(np.abs(T_step - T_step.conj().T))
        print(f"  T_F step t={t}: |T - T†| = {herm_err:.3e}")
        T_total = T_step @ T_total
    print(f"  built T̂_F^{N_E-1} ({N_E-1} steps) in {time.time()-t0:.1f}s")

    herm_err = np.max(np.abs(T_total - T_total.conj().T))
    print(f"  T̂_F^{N_E-1}: |T - T†| = {herm_err:.3e}")

    eigs = np.linalg.eigvals(T_total)
    print(f"  Max |imag eigval|: {np.max(np.abs(eigs.imag)):.3e}")
    print(f"  Eigvals (real, sorted, top 8): {sorted(eigs.real, reverse=True)[:8]}")
    print(f"  All real >= 0?  {np.all(eigs.real >= -1e-10)}")

    # Project to phi=0 BC: 16x16 sub-matrix
    T_phi0 = project_to_phi_vacuum(T_total, V3)
    print(f"  T̂_F^4 in χ̂-only subspace: shape={T_phi0.shape}")

    # n̂_0 = bit 0 of basis index (chi at spatial site 0)
    dim_chi = 1 << V3
    Z = 0.0
    num = 0.0
    for state in range(dim_chi):
        weight = T_phi0[state, state].real
        Z += weight
        n0 = (state >> 0) & 1
        num += weight * n0
    n0_lat = num / Z if Z > 1e-12 else float('nan')
    print(f"  Z = Σ T̂_F^4[i,i] (φ=0 sector) = {Z:+.6f}")
    print(f"  Tr[T̂_F^4 n̂_0]_{{φ=0}} = {num:+.6f}")
    print(f"  ⟨n̂_0⟩_lattice_MK = {n0_lat:+.6f}")

    # Compare to z2_setup at K=0 (free fermion limit on 2x2 spatial)
    import sys
    sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
    from action_endtoend_pipeline import physical_sector_reference
    # K=0: gauge term zero (only kinetic + mass).
    # Actually physical_sector_reference uses build_pauli_terms which encodes
    # the specific z2_setup Hamiltonian.  Run at β=4.
    ref = physical_sector_reference(beta=4.0, times=[0.0])
    print(f"  physical_sector_reference(β=4)  = {ref[0.0]:+.6f}  (z2_setup Hamiltonian)")


def test_v3_4_gauge_averaged(n_gauge: int = 80, K_E: float = 1.0, K_M: float = 0.5,
                            seed: int = 2026, n_warmup: int = 500):
    """V_3 = 4, gauge-average M-K thermal ⟨n̂_0⟩ via gauge MC.
    Now with separate K_E (electric) and K_M (magnetic) to match z2_setup.
    Tracks sign(det M) for sign-corrected reweighting."""
    print()
    print(f"=" * 72)
    print(f"M-K gauge-averaged ⟨n̂_0⟩: V_3=4, K_E={K_E}, K_M={K_M}, "
          f"N_E=5, m=0.5, n_gauge={n_gauge}")
    print(f"=" * 72)
    import time
    import sys
    sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
    from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
    from action_z2_metropolis import run_metropolis

    Lx, Ly = 2, 2
    V3 = Lx * Ly
    N_E = 5
    m = 0.5
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=m)

    chi_ops, phi_ops = build_fermion_ops(V3)

    # Gauge MC with K_E, K_M and longer warmup
    mc = run_metropolis(geom, K_E=K_E, K_M=K_M, n_sweeps=n_gauge,
                       n_warmup=n_warmup, seed=seed)
    print(f"  gauge MC: {len(mc.configs)} configs, accept {mc.accept_rate:.3f}, "
          f"⟨sign⟩ = {mc.avg_sign:+.3f}")
    n_unique = len(set(tuple(c.U_x.flatten()) + tuple(c.U_y.flatten())
                        + tuple(c.U_t.flatten()) for c in mc.configs))
    print(f"  unique configs: {n_unique}/{len(mc.configs)}")

    Z_num_sum = 0.0          # sign-weighted Σ (n̂_0 part of trace)
    Z_denom_sum = 0.0        # sign-weighted Σ (trace)
    sign_sum = 0.0
    t0 = time.time()
    for U_idx, U in enumerate(mc.configs):
        T_total = np.eye(1 << (2 * V3), dtype=complex)
        for t in range(N_E - 1):
            T_step = mk_T_F_single_step((Lx, Ly), chi_ops, phi_ops,
                                        U.U_x, U.U_y, t, m)
            T_total = T_step @ T_total
        T_phi0 = project_to_phi_vacuum(T_total, V3)
        dim_chi = 1 << V3
        Z_U = 0.0
        num_U = 0.0
        for st in range(dim_chi):
            w = T_phi0[st, st].real
            Z_U += w
            num_U += w * ((st >> 0) & 1)
        sign_U = mc.sign_history[U_idx]
        # Reweighted sum:  ⟨O⟩ = ⟨sign·O⟩ / ⟨sign⟩
        Z_num_sum += sign_U * num_U
        Z_denom_sum += sign_U * Z_U
        sign_sum += sign_U
        if U_idx % 10 == 0:
            n0_U = num_U / Z_U if abs(Z_U) > 1e-12 else float('nan')
            print(f"  config {U_idx}: sign={sign_U:+d}, "
                  f"⟨n̂_0⟩_U = {n0_U:+.5f}, t={time.time()-t0:.0f}s")

    n0_avg = Z_num_sum / Z_denom_sum if abs(Z_denom_sum) > 1e-12 else float('nan')
    print(f"\n  Sign-reweighted ⟨n̂_0⟩ = (Σ sign·num) / (Σ sign·Z) = {n0_avg:+.6f}")
    print(f"  ⟨sign⟩ recorded = {sign_sum / len(mc.configs):+.4f}")

    from action_endtoend_pipeline import physical_sector_reference
    ref = physical_sector_reference(beta=4.0, times=[0.0])
    print(f"  physical_sector_reference(β=4): {ref[0.0]:+.6f}")


def test_v3_4_z2_random(n_trials: int = 3, seed: int = 2026):
    """V_3 = 4 with Z_2 random gauge. Verify Hermiticity preserved per step
    AND in the cumulative T̂_F^4."""
    print()
    print("=" * 72)
    print(f"M-K T̂_F test: V_3 = 4 with Z_2 random gauge, m = 0.5, N_E = 5")
    print("=" * 72)

    Lx, Ly = 2, 2
    V3 = Lx * Ly
    N_E = 5
    m = 0.5

    chi_ops, phi_ops = build_fermion_ops(V3)
    rng = np.random.default_rng(seed)

    for trial in range(n_trials):
        # Random Z_2 gauge config
        U_x = rng.choice([-1, 1], size=(N_E, Lx - 1, Ly)).astype(int)
        U_y = rng.choice([-1, 1], size=(N_E, Lx, Ly - 1)).astype(int)

        # Per-step Hermiticity
        max_step_herm = 0.0
        T_total = np.eye(1 << (2 * V3), dtype=complex)
        for t in range(N_E - 1):
            T_step = mk_T_F_single_step((Lx, Ly), chi_ops, phi_ops, U_x, U_y, t, m)
            herm_err = np.max(np.abs(T_step - T_step.conj().T))
            max_step_herm = max(max_step_herm, herm_err)
            T_total = T_step @ T_total

        # Cumulative Hermiticity
        cum_herm = np.max(np.abs(T_total - T_total.conj().T))
        eigs = np.linalg.eigvals(T_total)
        max_imag = np.max(np.abs(eigs.imag))
        all_real_pos = np.all(eigs.real >= -1e-10) and (max_imag < 1e-10)

        # phi=0 projection → ⟨n̂_0⟩
        T_phi0 = project_to_phi_vacuum(T_total, V3)
        dim_chi = 1 << V3
        Z, num = 0.0, 0.0
        for st in range(dim_chi):
            w = T_phi0[st, st].real
            Z += w
            num += w * ((st >> 0) & 1)
        n0 = num / Z if abs(Z) > 1e-12 else float('nan')

        print(f"  trial {trial}: max step |T-T†| = {max_step_herm:.2e}, "
              f"cumul |T-T†| = {cum_herm:.2e}, "
              f"max |Im eig| = {max_imag:.2e}, "
              f"all positive = {all_real_pos}, "
              f"⟨n̂_0⟩ = {n0:+.6f}")


if __name__ == "__main__":
    test_v3_1_free()
    test_v3_4_free()
    test_v3_4_z2_random()
    test_v3_4_gauge_averaged(n_gauge=30)
