"""Patched lattice ρ̃_det with different spatial-plaquette patterns at the
boundary slices.  Tests whether the "extra magnetic plaquette" hypothesis
explains the lattice/Hamiltonian-Trotter discrepancy.

Patterns tested:
  - 'all':       K_M at every slice (current/Wilson)
  - 'skip_first': K_M at slices 1..N_E-1 (matches Lie [H_M·H_E] ordering)
  - 'skip_last':  K_M at slices 0..N_E-2 (matches Lie [H_E·H_M] ordering)
  - 'half_bdy':   K_M·(1/2) at slices 0 and N_E-1, K_M at intermediates (Strang-within-H_g)

For each, compute ρ̃_det at β=2, m=0.5, a_τ=1, N_E=3 and compare to:
  - ED of e^{-βH_QC}
  - Hamiltonian Trotter (Lie and Strang orderings)
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
W_ORDER = 2

PATTERNS = ['all', 'skip_first', 'skip_last', 'half_bdy']


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


def build_H_components():
    terms = cs.build_pauli_terms()
    H = np.zeros((256, 256), dtype=complex)
    H_g = np.zeros((256, 256), dtype=complex)
    H_F = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        op_mat = pauli_op(ops)
        H += coef * op_mat
        if all(q < 4 for (q, _) in ops):
            H_g += coef * op_mat
        else:
            H_F += coef * op_mat
    return (H + H.conj().T)/2, (H_g + H_g.conj().T)/2, (H_F + H_F.conj().T)/2


def gauge_action_with_pattern(N_E, Lx, Ly, K_E, K_M, U_x, U_y, U_t, pattern):
    """S_g with controllable magnetic plaquette pattern."""
    total_E = 0.0
    total_M = 0.0
    for t in range(N_E):
        # Determine weight for slice t
        if pattern == 'all':
            w = 1.0
        elif pattern == 'skip_first':
            w = 0.0 if t == 0 else 1.0
        elif pattern == 'skip_last':
            w = 0.0 if t == N_E - 1 else 1.0
        elif pattern == 'half_bdy':
            w = 0.5 if (t == 0 or t == N_E - 1) else 1.0
        else:
            raise ValueError(pattern)
        if w == 0.0:
            continue
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                u1 = U_x[t, x, y]; u2 = U_y[t, x + 1, y]
                u3 = U_x[t, x, y + 1]; u4 = U_y[t, x, y]
                total_M += w * u1 * u2 * u3 * u4
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                u1 = U_x[t, x, y]; u2 = U_t[t, x + 1, y]
                u3 = U_x[t + 1, x, y]; u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
        for x in range(Lx):
            for y in range(Ly - 1):
                u1 = U_y[t, x, y]; u2 = U_t[t, x, y + 1]
                u3 = U_y[t + 1, x, y]; u4 = U_t[t, x, y]
                total_E += u1 * u2 * u3 * u4
    return -K_E * total_E - K_M * total_M


def n_links(N_E):
    return N_E * 1 * 2 + N_E * 2 * 1 + (N_E - 1) * 2 * 2


def decode_to_U(idx, N_E):
    Lx = Ly = 2
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.empty((N_E - 1, 2, 2), dtype=np.int8)
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
    for t in range(N_E - 1):
        for x in range(2):
            for y in range(2):
                U_t[t, x, y] = 1 if (idx >> b) & 1 == 0 else -1
                b += 1
    return Z2GaugeConfig(geom=geom, U_x=U_x, U_y=U_y, U_t=U_t)


def per_config(args):
    idx, K_E, K_M, pattern = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_to_U(idx, N_E)
    U.geom = geom
    S_g = gauge_action_with_pattern(
        N_E, 2, 2, K_E, K_M,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
        pattern,
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER)
    idx_to_psi = _fock_index_to_psi_map(4)
    dim_F = 16
    W_psi = np.zeros((dim_F, dim_F), dtype=complex)
    for li in range(dim_F):
        for lj in range(dim_F):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return idx, boltz, W_psi, g_top, g_bot


def build_rho_det(pattern):
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    total = n_links(N_E)
    n_configs = 1 << total
    args = ((i, K_E, K_M, pattern) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    completed = 0
    with Pool(6, initializer=warm_up) as pool:
        for idx, w, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config, args, chunksize=max(1, n_configs // 192)):
            completed += 1
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    print(f"    walltime: {time.time()-t0:.0f}s", flush=True)
    return (rho + rho.conj().T) / 2


def main():
    print("=" * 80)
    print(f"Lattice ρ̃_det with various spatial-plaquette patterns")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}")
    print("=" * 80, flush=True)

    H, H_g, H_F = build_H_components()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    # Hamiltonian Trotter Lie [H_F then H_g] (right-to-left)
    E_g = expm(-A_TAU * H_g)
    E_F = expm(-A_TAU * H_F)
    rho_T_OP = np.linalg.matrix_power(E_g @ E_F, N_E - 1)
    rho_T_OP_n = rho_T_OP / np.trace(rho_T_OP).real

    # Hamiltonian Trotter Strang
    E_g_half = expm(-(A_TAU/2) * H_g)
    rho_T_strang = np.linalg.matrix_power(E_g_half @ E_F @ E_g_half, N_E - 1)
    rho_T_strang_n = rho_T_strang / np.trace(rho_T_strang).real

    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0]
    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}

    def C_of(rho_n):
        return {t: np.real(np.trace(rho_n @ UOU[t] @ n0_op)) for t in TIMES}

    C_ED = C_of(rho_ED_n)
    C_T_OP = C_of(rho_T_OP_n)
    C_T_strang = C_of(rho_T_strang_n)

    print(f"\n  Reference C(t):")
    for t in TIMES:
        print(f"    t={t}: C_ED={C_ED[t]:+.5f}, C_T_OP(Lie)={C_T_OP[t]:+.5f}, "
              f"C_T_strang={C_T_strang[t]:+.5f}")

    pattern_results = {}
    for pattern in PATTERNS:
        print(f"\n[Pattern: {pattern}]")
        rho_det = build_rho_det(pattern)
        rho_det_n = rho_det / np.trace(rho_det).real
        Z_lat = np.trace(rho_det).real
        eigs = np.real(np.linalg.eigvalsh(rho_det_n)); eigs.sort()
        d_ED = np.linalg.norm(rho_det_n - rho_ED_n)
        d_TOP = np.linalg.norm(rho_det_n - rho_T_OP_n)
        d_Tstrang = np.linalg.norm(rho_det_n - rho_T_strang_n)
        print(f"  Z_lat = {Z_lat:.4e}, ‖Δρ ED‖={d_ED:.4f}, ‖Δρ T_OP‖={d_TOP:.4f}, "
              f"‖Δρ T_strang‖={d_Tstrang:.4f}")
        print(f"  min_eig={eigs[0]:+.4e}, max_eig={eigs[-1]:+.4e}, "
              f"#neg={int((eigs<-1e-6).sum())}/256")
        C_lat = C_of(rho_det_n)
        for t in TIMES:
            print(f"  t={t}: C_lat={C_lat[t]:+.5f}, gap_ED={C_lat[t]-C_ED[t]:+.5f}, "
                  f"gap_T_OP={C_lat[t]-C_T_OP[t]:+.5f}, gap_T_strang={C_lat[t]-C_T_strang[t]:+.5f}")
        pattern_results[pattern] = {'C_lat': C_lat, 'd_ED': d_ED, 'd_TOP': d_TOP, 'd_Tstrang': d_Tstrang}

    print("\n" + "=" * 80)
    print("FINAL summary — ‖Δρ‖_F (smallest wins)")
    print("=" * 80)
    print(f"  {'pattern':>15} | {'vs ED':>8} | {'vs T_Lie':>8} | {'vs T_Strang':>11}")
    for p in PATTERNS:
        r = pattern_results[p]
        print(f"  {p:>15} | {r['d_ED']:>8.4f} | {r['d_TOP']:>8.4f} | {r['d_Tstrang']:>11.4f}")


if __name__ == "__main__":
    main()
