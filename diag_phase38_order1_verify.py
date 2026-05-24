"""Phase 38: verify the order=1 fix.

Re-run the deterministic enumeration at β=2, m=0.5 with W_ORDER=1
(time-ordered Lie) instead of WORDER=2 (palindrome), in TEMPORAL gauge
(U_t = +1).  Two a_τ values to check Trotter convergence:

  a_τ=1, N_E=3: 4096 spatial gauge configs (instant)
  a_τ=0.5, N_E=5: 2^20 = 1M spatial gauge configs (a few minutes)

If order=1 fix is correct:
  - At a_τ=1: gap = T_match Trotter error ≈ -0.006 (from Phase 37b)
  - At a_τ=0.5: gap should be smaller (Trotter convergence)
  - Extrapolation to a_τ → 0: gap → 0
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


def build_H_QC():
    terms = cs.build_pauli_terms()
    H = np.zeros((256, 256), dtype=complex)
    for coef, ops in terms:
        H += coef * pauli_op(ops)
    return (H + H.conj().T) / 2


def n_spatial_links(N_E):
    return N_E * 1 * 2 + N_E * 2 * 1


def decode_spatial_only(idx, N_E):
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=M_HAM)
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


def per_config(args):
    idx, N_E, a_tau, K_E, K_M = args
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=a_tau * M_HAM)
    U = decode_spatial_only(idx, N_E)
    U.geom = geom
    S_g = gauge_action_temporal(
        N_E, K_E, K_M,
        U.U_x.astype(np.float64), U.U_y.astype(np.float64),
    )
    boltz = np.exp(-S_g)
    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=1)   # ORDER=1 — the fix
    idx_to_psi = _fock_index_to_psi_map(4)
    W_psi = np.zeros((16, 16), dtype=complex)
    for li in range(16):
        for lj in range(16):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return boltz, W_psi, g_top, g_bot


def build_rho_temporal_order1(N_E, a_tau, hermitize=True):
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    n_configs = 1 << n_spatial_links(N_E)
    print(f"  N_E={N_E}, a_τ={a_tau}, β={(N_E-1)*a_tau:.2f}, "
          f"{n_configs} configs, K_E={K_E:.4f}, K_M={K_M:.4f}", flush=True)
    args = ((i, N_E, a_tau, K_E, K_M) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    completed = 0
    report_every = max(1, n_configs // 10)
    with Pool(6, initializer=warm_up) as pool:
        for w, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config, args, chunksize=max(1, n_configs // 192)):
            completed += 1
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
            if completed % report_every == 0:
                print(f"    {100*completed/n_configs:.0f}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    if hermitize:
        rho = (rho + rho.conj().T) / 2
    return rho


def main():
    print("=" * 80)
    print(f"Phase 38: order=1 fix verification at β={BETA}, m={M_HAM}")
    print(f"  Two a_τ values to test Trotter convergence")
    print("=" * 80, flush=True)

    H = build_H_QC()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0]
    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op)) for t in TIMES}
    print(f"\nED reference: " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES), flush=True)

    SCAN = [(3, 1.0), (5, 0.5)]
    summaries = {}
    for N_E, a_tau in SCAN:
        print(f"\n[a_τ={a_tau}, N_E={N_E}]")
        rho_lat = build_rho_temporal_order1(N_E=N_E, a_tau=a_tau, hermitize=True)
        rho_lat_n = rho_lat / np.trace(rho_lat).real
        Z_lat = np.trace(rho_lat).real
        eigs = np.real(np.linalg.eigvalsh(rho_lat_n)); eigs.sort()
        d = np.linalg.norm(rho_lat_n - rho_ED_n)
        print(f"  Z_lat = {Z_lat:.4e}")
        print(f"  min_eig = {eigs[0]:+.4e}, max_eig = {eigs[-1]:+.4e}, "
              f"# neg = {int((eigs<-1e-6).sum())}/256")
        print(f"  ‖Δρ‖_F = {d:.5f}")
        gaps = {}
        for t in TIMES:
            C_lat = np.real(np.trace(rho_lat_n @ UOU[t] @ n0_op))
            gap = C_lat - C_ED[t]
            gaps[t] = (C_lat, gap)
            print(f"  t={t}: C_lat={C_lat:+.5f}, C_ED={C_ED[t]:+.5f}, gap={gap:+.5f}")
        summaries[(N_E, a_tau)] = gaps

    print("\n" + "=" * 80)
    print("Trotter convergence (gap = C_lat − C_ED)")
    print("=" * 80)
    print(f"  {'a_τ':>5} | {'N_E':>4} | "
          + " | ".join(f"gap C({t}):>10" for t in TIMES))
    for (N_E, a_tau), gaps in summaries.items():
        print(f"  {a_tau:>5.2f} | {N_E:>4d} | "
              + " | ".join(f"{gaps[t][1]:>+10.5f}" for t in TIMES))
    print(f"\nIf Trotter convergence works:")
    print(f"  - gap at a_τ=0.5 should be smaller (closer to 0) than at a_τ=1")
    print(f"  - Specifically, O(a_τ²) Trotter → roughly 4× smaller")


if __name__ == "__main__":
    main()
