"""Deterministic enumeration in TEMPORAL GAUGE (U_t = +1 fixed).

In temporal gauge, the Wilson 4-link temporal plaquette
  u_x(t)·u_t(t,a)·u_x(t+1)·u_t(t,b)
reduces to a 2-link bond σ_x(t)·σ_x(t+1) (since u_t = +1 → u_t² = 1).
This matches the standard Suzuki mapping of e^{-a_τH_E}.

If lattice C(0) in temporal gauge matches Hamiltonian Trotter Lie
(within Trotter-error magnitude), then the prior gap was from
gauge non-invariance in Wilson gauge (free U_t).

If still differs → another bug.

Gauge dof in temporal gauge for N_E=3, 2x2 OBC:
  - U_x: 3·1·2 = 6 links
  - U_y: 3·2·1 = 6 links
  - U_t: fixed to +1, 0 variables
Total: 12 binary dof → 4096 configs.  Fast.
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


def n_spatial_links(N_E):
    return N_E * 1 * 2 + N_E * 2 * 1   # nx + ny


def decode_spatial_only(idx, N_E):
    """Decode bits into U_x, U_y; set U_t = +1 (temporal gauge)."""
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U_x = np.empty((N_E, 1, 2), dtype=np.int8)
    U_y = np.empty((N_E, 2, 1), dtype=np.int8)
    U_t = np.ones((N_E - 1, 2, 2), dtype=np.int8)  # FIXED +1
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


def gauge_action_temporal(N_E, Lx, Ly, K_E, K_M, U_x, U_y, M_pattern='all'):
    """S_g in TEMPORAL GAUGE (U_t = +1): 2-link bonds in time, spatial plaquettes
    according to M_pattern.
    """
    total_E = 0.0
    total_M = 0.0
    # Temporal "bonds" (since U_t=1, plaquette = σ_x(t)·σ_x(t+1)):
    for t in range(N_E - 1):
        for x in range(Lx - 1):
            for y in range(Ly):
                total_E += U_x[t, x, y] * U_x[t + 1, x, y]
        for x in range(Lx):
            for y in range(Ly - 1):
                total_E += U_y[t, x, y] * U_y[t + 1, x, y]
    # Spatial plaquettes (xy):
    for t in range(N_E):
        if M_pattern == 'all':
            w = 1.0
        elif M_pattern == 'skip_first':
            w = 0.0 if t == 0 else 1.0
        elif M_pattern == 'skip_last':
            w = 0.0 if t == N_E - 1 else 1.0
        elif M_pattern == 'half_bdy':
            w = 0.5 if (t == 0 or t == N_E - 1) else 1.0
        else:
            raise ValueError(M_pattern)
        if w == 0.0:
            continue
        for x in range(Lx - 1):
            for y in range(Ly - 1):
                u1 = U_x[t, x, y]; u2 = U_y[t, x + 1, y]
                u3 = U_x[t, x, y + 1]; u4 = U_y[t, x, y]
                total_M += w * u1 * u2 * u3 * u4
    return -K_E * total_E - K_M * total_M


def per_config_temporal(args):
    idx, K_E, K_M, M_pattern = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_spatial_only(idx, N_E)
    U.geom = geom
    S_g = gauge_action_temporal(
        N_E, 2, 2, K_E, K_M,
        U.U_x.astype(np.float64), U.U_y.astype(np.float64),
        M_pattern,
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


def build_rho_det_temporal(M_pattern='all'):
    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    total = n_spatial_links(N_E)
    n_configs = 1 << total
    args = ((i, K_E, K_M, M_pattern) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
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
    print(f"    walltime: {time.time()-t0:.0f}s", flush=True)
    return (rho + rho.conj().T) / 2


def main():
    print("=" * 80)
    print(f"TEMPORAL GAUGE deterministic test: U_t = +1 fixed")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}")
    print(f"  4096 spatial gauge configs (vs 1M in free-U_t case)")
    print("=" * 80, flush=True)

    H, H_g, H_F = build_H_components()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    E_g = expm(-A_TAU * H_g)
    E_F = expm(-A_TAU * H_F)
    rho_T_OP = np.linalg.matrix_power(E_g @ E_F, N_E - 1)
    rho_T_OP_n = rho_T_OP / np.trace(rho_T_OP).real

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

    print(f"\n  References:")
    for t in TIMES:
        print(f"    t={t}: C_ED={C_ED[t]:+.5f}  C_T_OP={C_T_OP[t]:+.5f}  "
              f"C_T_strang={C_T_strang[t]:+.5f}")

    for pattern in ['all', 'skip_first', 'skip_last', 'half_bdy']:
        print(f"\n[Temporal gauge + spatial K_M pattern: {pattern}]")
        rho_det = build_rho_det_temporal(M_pattern=pattern)
        rho_det_n = rho_det / np.trace(rho_det).real
        Z_lat = np.trace(rho_det).real
        eigs = np.real(np.linalg.eigvalsh(rho_det_n)); eigs.sort()
        d_ED = np.linalg.norm(rho_det_n - rho_ED_n)
        d_TOP = np.linalg.norm(rho_det_n - rho_T_OP_n)
        d_Tstrang = np.linalg.norm(rho_det_n - rho_T_strang_n)
        print(f"  Z_lat = {Z_lat:.4e}, Z_lat/Z_ED = {Z_lat/np.trace(rho_ED).real:.4f}")
        print(f"  ‖Δρ‖_F: vs_ED={d_ED:.4f}, vs_T_OP={d_TOP:.4f}, vs_T_strang={d_Tstrang:.4f}")
        print(f"  min_eig={eigs[0]:+.4e}, max_eig={eigs[-1]:+.4e}, "
              f"#neg={int((eigs<-1e-6).sum())}/256")
        C_lat = C_of(rho_det_n)
        for t in TIMES:
            print(f"  t={t}: C_lat={C_lat[t]:+.5f}, gap_ED={C_lat[t]-C_ED[t]:+.5f}, "
                  f"gap_T_OP={C_lat[t]-C_T_OP[t]:+.5f}, gap_T_strang={C_lat[t]-C_T_strang[t]:+.5f}")


if __name__ == "__main__":
    main()
