"""Compare lattice ρ̃_det (temporal gauge) to the MATCHED Hamiltonian
Trotter operator that corresponds to the lattice's action structure.

The lattice in temporal gauge applies (reading right-to-left from |g_b, ψ_b⟩):
  M(g_0) · F(g_0) · E(g_0→g_1) · M(g_1) · F(g_1) · E(g_1→g_2) · M(g_2)
i.e., N+1 H_M factors interleaved with N H_E transitions and N H_F applications.

In operator form on 256-dim space:
  T_match = E_M · (E_E · E_F · E_M)^N   for N = N_E − 1

If this matches lattice ρ̃_det in temporal gauge, my earlier
Hamiltonian-Trotter comparison was wrong (used [E_g·E_F]^N which is a
different Trotter ordering than the lattice).

Setup: β=2, m=0.5, a_τ=1, N_E=3 → N=2.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 0.5
import epoq_quantum_simulator as qs
qs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_kernels import warm_up
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
A_TAU = 1.0
N_E = 3
N = N_E - 1
W_ORDER = 2


def pauli_op(factors, nq=8):
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    by_q = {q: P2['I'] for q in range(nq)}
    for q, ax in factors:
        by_q[q] = P2[ax]
    r = by_q[nq - 1]
    for q in range(nq - 2, -1, -1):
        r = np.kron(r, by_q[q])
    return r


def build_HE_HM_HF():
    """Split H_QC into H_E (single-X), H_M (4-Z plaquette), H_F (rest)."""
    terms = cs.build_pauli_terms()
    H = np.zeros((256, 256), dtype=complex)
    H_E = np.zeros((256, 256), dtype=complex)
    H_M = np.zeros((256, 256), dtype=complex)
    H_F = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        op_mat = pauli_op(ops)
        H += coef * op_mat
        # Determine classification
        ops_sorted = sorted(ops)
        if len(ops) == 1 and ops[0][0] < 4 and ops[0][1] == 'X':
            H_E += coef * op_mat
        elif len(ops) == 4 and all(q < 4 and p == 'Z' for (q, p) in ops):
            H_M += coef * op_mat
        else:
            H_F += coef * op_mat
    return (H + H.conj().T)/2, (H_E + H_E.conj().T)/2, (H_M + H_M.conj().T)/2, (H_F + H_F.conj().T)/2


def n_spatial_links(N_E):
    return N_E * 1 * 2 + N_E * 2 * 1


def decode_spatial_only(idx, N_E):
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.ones((N_E - 1, 2, 2), dtype=np.int8)
    b = 0
    for t in range(N_E):
        for x in range(1):
            for y in range(2):
                U_x[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    for t in range(N_E):
        for x in range(2):
            for y in range(1):
                U_y[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


def gauge_action_temporal(N_E, K_E, K_M, U_x, U_y):
    """Temporal gauge: K_E bonds for transitions, K_M plaquette per slice."""
    total_E = 0.0
    total_M = 0.0
    for t in range(N_E - 1):
        for y in range(2):
            total_E += U_x[t, 0, y] * U_x[t + 1, 0, y]
        for x in range(2):
            total_E += U_y[t, x, 0] * U_y[t + 1, x, 0]
    for t in range(N_E):
        u1 = U_x[t, 0, 0]; u2 = U_y[t, 1, 0]
        u3 = U_x[t, 0, 1]; u4 = U_y[t, 0, 0]
        total_M += u1 * u2 * u3 * u4
    return -K_E * total_E - K_M * total_M


def per_config_temporal(args):
    idx, K_E, K_M = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_spatial_only(idx, N_E)
    U.geom = geom
    S_g = gauge_action_temporal(
        N_E, K_E, K_M,
        U.U_x.astype(np.float64), U.U_y.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER)
    idx_to_psi = _fock_index_to_psi_map(4)
    W_psi = np.zeros((16, 16), dtype=complex)
    for li in range(16):
        for lj in range(16):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return boltz, W_psi, g_top, g_bot


def build_rho_det_temporal():
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    n_configs = 1 << n_spatial_links(N_E)
    args = ((i, K_E, K_M) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    with Pool(6, initializer=warm_up) as pool:
        for w, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config_temporal, args, chunksize=max(1, n_configs // 192)):
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    return (rho + rho.conj().T) / 2


def main():
    print("=" * 80)
    print(f"Lattice (temporal gauge) ρ̃_det vs Matched-Trotter Hamiltonian operator")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}, N={N}")
    print("=" * 80, flush=True)

    H, H_E, H_M, H_F = build_HE_HM_HF()
    # Sanity
    assert np.max(np.abs(H - H_E - H_M - H_F)) < 1e-12, "H split mismatch"
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    E_E = expm(-A_TAU * H_E)
    E_M = expm(-A_TAU * H_M)
    E_F = expm(-A_TAU * H_F)
    # Original [E_g · E_F]^N (where E_g = e^{-a_τ(H_E+H_M)})
    H_g = H_E + H_M
    E_g = expm(-A_TAU * H_g)
    rho_T_OP = np.linalg.matrix_power(E_g @ E_F, N)
    rho_T_OP_n = rho_T_OP / np.trace(rho_T_OP).real

    # MATCHED: lattice's M·F·E·M·F·E·M structure
    # rho_T_match = E_M · (E_E · E_F · E_M)^N
    #             = E_M · ∏_{k=1..N} (E_E · E_F · E_M)
    step = E_E @ E_F @ E_M
    rho_T_match = E_M @ np.linalg.matrix_power(step, N)
    rho_T_match_n = rho_T_match / np.trace(rho_T_match).real

    # Variant: skip the leftmost E_M, i.e., (E_E·E_F·E_M)^N
    # corresponds to lattice with K_M dropped at the last slice
    rho_T_match_skip_last = np.linalg.matrix_power(step, N)
    rho_T_match_skip_last_n = rho_T_match_skip_last / np.trace(rho_T_match_skip_last).real

    # Variant: skip the rightmost E_M, i.e., E_M · (E_E·E_F)^N · E_M (wait, that's different)
    # Actually: lattice with K_M at all but slice 0 → drop the LAST E_M in our right-to-left expansion
    # Original: M(g_0) · F(g_0) · E · M(g_1) · F(g_1) · E · M(g_2)  (RTL)
    # operator = M · F · E · M · F · E · M
    # skip_first means drop K_M at slice 0 = drop the RIGHTMOST M in operator product
    # i.e., F · E · M · F · E · M
    rho_T_match_skip_first = E_M @ E_F @ E_E @ E_M @ E_F @ E_E   # for N=2
    # Hmm let me check. Reading right-to-left: E_E first, then E_F at intermediate gauge, M at g_1, ...
    # actually wait — skipping slice 0 K_M means no M(g_0). The leftmost operator we apply to ψ_b is E_F(g_b).
    # operator (RTL): F(g_0) · E · M(g_1) · F(g_1) · E · M(g_2)
    # In matrix product: M · E · F · M · E · F  (read left-to-right, applies F first)
    rho_T_match_skip_first = E_M @ E_E @ E_F @ E_M @ E_E @ E_F
    rho_T_match_skip_first_n = rho_T_match_skip_first / np.trace(rho_T_match_skip_first).real

    # Compute lattice ρ̃_det in temporal gauge
    print("\nBuilding lattice ρ̃_det (temporal gauge, 4096 configs)...", flush=True)
    t0 = time.time()
    rho_det = build_rho_det_temporal()
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    rho_det_n = rho_det / np.trace(rho_det).real

    # Hermitization sanity: is lattice symmetric in particular ways?
    print(f"\n  ‖ρ_det − ρ_det.conj().T‖ = {np.linalg.norm(rho_det - rho_det.conj().T):.4e}")

    # Frobenius distances
    print(f"\n[Frobenius distances ‖ · −·‖_F, normalized]")
    print(f"  ρ_det     vs ρ_ED         = {np.linalg.norm(rho_det_n - rho_ED_n):.5f}")
    print(f"  ρ_det     vs ρ_T_OP       = {np.linalg.norm(rho_det_n - rho_T_OP_n):.5f}")
    print(f"  ρ_det     vs ρ_T_match    = {np.linalg.norm(rho_det_n - rho_T_match_n):.5f}")
    print(f"  ρ_det     vs ρ_T_match_skip_first = {np.linalg.norm(rho_det_n - rho_T_match_skip_first_n):.5f}")
    print(f"  ρ_det     vs ρ_T_match_skip_last  = {np.linalg.norm(rho_det_n - rho_T_match_skip_last_n):.5f}")
    print(f"  ρ_ED      vs ρ_T_match    = {np.linalg.norm(rho_ED_n - rho_T_match_n):.5f}")
    print(f"  ρ_ED      vs ρ_T_OP       = {np.linalg.norm(rho_ED_n - rho_T_OP_n):.5f}")

    # C(t) comparison
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0]
    print(f"\n[C(t) comparison]")
    print(f"  {'t':>5} | {'ED':>10} | {'T_OP':>10} | {'T_match':>10} | {'T_match_skip_first':>18} | {'T_match_skip_last':>18} | {'det':>10}")
    for t in TIMES:
        UOU = expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t)
        C_ED = np.real(np.trace(rho_ED_n @ UOU @ n0_op))
        C_T_OP = np.real(np.trace(rho_T_OP_n @ UOU @ n0_op))
        C_T_match = np.real(np.trace(rho_T_match_n @ UOU @ n0_op))
        C_T_match_skf = np.real(np.trace(rho_T_match_skip_first_n @ UOU @ n0_op))
        C_T_match_skl = np.real(np.trace(rho_T_match_skip_last_n @ UOU @ n0_op))
        C_det = np.real(np.trace(rho_det_n @ UOU @ n0_op))
        print(f"  {t:>5.2f} | {C_ED:>+10.5f} | {C_T_OP:>+10.5f} | {C_T_match:>+10.5f} | {C_T_match_skf:>+18.5f} | {C_T_match_skl:>+18.5f} | {C_det:>+10.5f}")


if __name__ == "__main__":
    main()
