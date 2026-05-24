"""Phase 37b: Direct full-matrix comparison of lattice (TEMPORAL GAUGE)
ρ̃ vs the matched Hamiltonian Trotter operator T_match = M·F·E·M·F·E·M.

Phase 37 confirmed T_E_lat_temp = (1/c^4) · Hamiltonian e^{-a_τH_E} per
transition (perfect, constant ratio).  Phase 36 confirmed T_F lat exact.
So lattice (temporal) ρ̃ should equal (1/c^8) · T_match by direct
operator-by-operator substitution.

If they DO match → bug was in my analysis of free U_t case only.
If they DON'T match → there's still a missed structural difference.
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

M_HAM = 0.5
G_E_HAM, G_M_HAM, G_HOP = 1.0, 0.5, 0.5
BETA = 2.0
A_TAU = 1.0
N_E = 3
N = N_E - 1


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
    terms = cs.build_pauli_terms()
    H_E = np.zeros((256, 256), dtype=complex)
    H_M = np.zeros((256, 256), dtype=complex)
    H_F = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        op_mat = pauli_op(ops)
        if len(ops) == 1 and ops[0][0] < 4 and ops[0][1] == 'X':
            H_E += coef * op_mat
        elif len(ops) == 4 and all(q < 4 and p == 'Z' for (q, p) in ops):
            H_M += coef * op_mat
        else:
            H_F += coef * op_mat
    return ((H_E + H_E.conj().T) / 2,
            (H_M + H_M.conj().T) / 2,
            (H_F + H_F.conj().T) / 2)


def gauge_action_temporal_inline(N_E, K_E, K_M, U_x, U_y):
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


def per_config_temporal(args):
    idx, K_E, K_M = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=A_TAU * M_HAM)
    U = decode_spatial_only(idx, N_E)
    U.geom = geom
    S_g = gauge_action_temporal_inline(
        N_E, K_E, K_M,
        U.U_x.astype(np.float64), U.U_y.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=1)
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
    n_configs = 1 << (N_E * 1 * 2 + N_E * 2 * 1)
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
    return rho   # NOT hermitized; want raw for direct comparison


def main():
    print("=" * 80)
    print("Phase 37b: lattice (temporal gauge) ρ̃ vs T_match operator")
    print(f"  β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}")
    print("=" * 80, flush=True)

    H_E, H_M, H_F = build_HE_HM_HF()
    E_E = expm(-A_TAU * H_E)
    E_M = expm(-A_TAU * H_M)
    E_F = expm(-A_TAU * H_F)

    # T_match = M · (E·F·M)^N  (using order=1 W: T_F[1]·T_F[0])
    step = E_E @ E_F @ E_M
    rho_T_match = E_M @ np.linalg.matrix_power(step, N)

    print(f"\nT_match diagonals (first 4): {[rho_T_match[i,i].real for i in range(4)]}")
    print(f"T_match trace = {np.trace(rho_T_match).real:.4e}", flush=True)

    print(f"\nBuilding lattice ρ̃_lat in TEMPORAL gauge...", flush=True)
    t0 = time.time()
    rho_lat = build_rho_det_temporal()
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    print(f"  ρ_lat trace = {np.trace(rho_lat).real:.4e}")
    print(f"  Hermiticity: ‖ρ_lat − ρ_lat†‖_F = {np.linalg.norm(rho_lat - rho_lat.conj().T):.4e}")

    # Expected: ρ_lat = (1/c^8) · T_match
    c_per_link = np.sqrt(np.cosh(A_TAU * G_E_HAM) * np.sinh(A_TAU * G_E_HAM))
    expected_ratio = 1.0 / (c_per_link ** 8)
    print(f"\n  c_per_link = {c_per_link:.4f}")
    print(f"  expected (1/c^8) = {expected_ratio:.4e}")
    print(f"  empirical ratio Tr(ρ_lat)/Tr(T_match) = {(np.trace(rho_lat)/np.trace(rho_T_match)).real:.4e}")

    # Direct comparison
    print(f"\n  ‖ρ_lat − (1/c^8) · T_match‖_F = {np.linalg.norm(rho_lat - expected_ratio * rho_T_match):.4e}")
    # Normalized
    rho_lat_n = rho_lat / np.trace(rho_lat).real
    rho_T_match_n = rho_T_match / np.trace(rho_T_match).real
    print(f"  ‖ρ_lat_n − T_match_n‖_F = {np.linalg.norm(rho_lat_n - rho_T_match_n):.4e}")

    # Element-by-element ratio (for entries where T_match is non-tiny)
    mask = np.abs(rho_T_match) > 1e-6
    ratio = np.where(mask, rho_lat / np.maximum(np.abs(rho_T_match), 1e-30) * np.sign(rho_T_match.real), 0.0)
    nonzero_ratios = ratio[mask].real
    print(f"\n  Element-wise ratio rho_lat/T_match (where |T_match|>1e-6):")
    print(f"    n elements: {mask.sum()}")
    print(f"    min ratio: {nonzero_ratios.min():.6e}")
    print(f"    max ratio: {nonzero_ratios.max():.6e}")
    print(f"    std/mean:  {nonzero_ratios.std()/nonzero_ratios.mean():.4e}")
    print(f"    If proportional, ratios should all be ≈ (1/c^8) = {expected_ratio:.4e}")


if __name__ == "__main__":
    main()
