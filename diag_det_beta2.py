"""Deterministic enumeration at β=2, m=0.5, a_τ=1, N_E=3.

Critical diagnostic: if MC at β=2 m=0.5 didn't converge to ED but
deterministic enumeration at the SAME (β, m, a_τ) → N_E=3 setup DOES
match ED → MC bug. If deterministic ALSO fails → construction bug at N_E≥3.

For 2x2 OBC, N_E=3: 8·3−4 = 20 binary gauge dof → 2^20 = 1M configs.
Tractable in ~10 min on 6 cores.
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
import epoq_quantum_simulator as qs
cs.M_MASS = 0.5
qs.M_MASS = 0.5

from action_z2_staggered import LatticeGeometry, Z2GaugeConfig
from action_z2_kernels import gauge_action_kernel, warm_up
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, G_HOP = 0.5, 1.0, 0.5, 0.5
BETA = 2.0
A_TAU = 1.0
N_E = int(round(BETA / A_TAU)) + 1   # 3
W_ORDER = 2


def n_links(N_E):
    Lx = Ly = 2
    nx = N_E * 1 * 2
    ny = N_E * 2 * 1
    nt = (N_E - 1) * 2 * 2
    return nx + ny + nt


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


def per_config(idx_n_E_atau_KE_KM):
    """Compute (idx, weight, W_psi block, g_top, g_bot) for one gauge config."""
    idx, N_E, a_tau, K_E, K_M = idx_n_E_atau_KE_KM
    Lx = Ly = 2
    geom = LatticeGeometry(Lx=Lx, Ly=Ly, N_E=N_E, m=a_tau * M_HAM)
    U = decode_to_U(idx, N_E)
    U.geom = geom

    S_g = gauge_action_kernel(
        Lx, Ly, N_E, K_E, K_M,
        U.U_x.astype(np.float64),
        U.U_y.astype(np.float64),
        U.U_t.astype(np.float64),
    )
    boltz = np.exp(-S_g)

    W_full, _ = compute_combined_weight_trotter(
        geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
        m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER)

    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)
    dim_F = 1 << V3
    W_psi = np.zeros((dim_F, dim_F), dtype=complex)
    for li in range(dim_F):
        for lj in range(dim_F):
            W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
    g_top = gauge_qc_bits_from_slice(U, N_E - 1)
    g_bot = gauge_qc_bits_from_slice(U, 0)
    return idx, boltz, W_psi, g_top, g_bot


def build_rho_det(N_E, a_tau, n_workers=6):
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    total = n_links(N_E)
    n_configs = 1 << total
    print(f"  Enumerating 2^{total} = {n_configs} configs with {n_workers} workers",
          flush=True)
    print(f"  K_E={K_E:.4f}, K_M={K_M:.4f}, a_τ={a_tau}, β={(N_E-1)*a_tau}",
          flush=True)

    args = ((i, N_E, a_tau, K_E, K_M) for i in range(n_configs))
    rho = np.zeros((256, 256), dtype=complex)
    t0 = time.time()
    completed = 0
    report_every = max(1, n_configs // 20)
    with Pool(n_workers, initializer=warm_up) as pool:
        for idx, w, W_psi, g_top, g_bot in pool.imap_unordered(
                per_config, args, chunksize=max(1, n_configs // (n_workers * 32))):
            completed += 1
            if w == 0:
                continue
            for psi_top in range(16):
                for psi_bot in range(16):
                    bit_a = (psi_top << 4) | g_top
                    bit_b = (psi_bot << 4) | g_bot
                    rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
            if completed % report_every == 0:
                elapsed = time.time() - t0
                print(f"    {completed}/{n_configs} ({100*completed/n_configs:.1f}%), "
                      f"elapsed {elapsed:.0f}s", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return (rho + rho.conj().T) / 2


def build_H_QC():
    P2 = {'I': np.eye(2, dtype=complex),
          'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': np.diag([1, -1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors:
            by_q[q] = P2[ax]
        r = by_q[nq - 1]
        for q in range(nq - 2, -1, -1):
            r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in cs.build_pauli_terms())
    return (H + H.conj().T) / 2


def main():
    print("=" * 80)
    print(f"Deterministic enumeration: β={BETA}, m={M_HAM}, a_τ={A_TAU}, N_E={N_E}")
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
    UOU = {}
    C_ED = {}
    for t in TIMES:
        Ut = expm(-1j * H * t)
        UOU[t] = Ut.conj().T @ n0_op @ Ut
        C_ED[t] = np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op))
    print(f"\nED reference: " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES))

    rho_det = build_rho_det(N_E=N_E, a_tau=A_TAU, n_workers=6)
    rho_det_n = rho_det / np.trace(rho_det).real

    print(f"\n  Z_lat (raw) = {np.trace(rho_det).real:.4e}, Z_ED = {np.trace(rho_ED).real:.4e}")
    eigs = np.real(np.linalg.eigvalsh(rho_det_n))
    eigs.sort()
    print(f"  min_eig(ρ̃_norm) = {eigs[0]:+.4e}, max_eig = {eigs[-1]:+.4e}, "
          f"#neg = {int((eigs<-1e-6).sum())}/256")
    print(f"  ‖Δρ‖_F = {np.linalg.norm(rho_det_n - rho_ED_n):.5f}")

    print(f"\n  Deterministic vs ED:")
    print(f"  {'t':>5} | {'C_det':>12} | {'C_ED':>12} | {'gap':>12}")
    for t in TIMES:
        C_det = np.real(np.trace(rho_det_n @ UOU[t] @ n0_op))
        print(f"  {t:>5.2f} | {C_det:>+12.6f} | {C_ED[t]:>+12.6f} | "
              f"{C_det - C_ED[t]:>+12.6f}")

    # Also: what does the prior MC at β=2 m=0.5 a_τ=0.5 give?  (for comparison)
    # MC was at a_τ=0.5 N_E=5 (so 36 link dof — not enumerable).  Compare to MC results.
    print(f"\n  For reference: MC at β=2 m=0.5 a_τ=0.5 (N_E=5) gave C(0)≈+0.27333")
    print(f"  This DETERMINISTIC at a_τ=1 (N_E=3) result vs ED tests construction at N_E=3.")
    print(f"  If C_det matches ED to ~1e-3 → construction OK at N_E=3, structural bug is at higher N_E.")
    print(f"  If C_det MISSES ED → structural bug already at N_E=3.")


if __name__ == "__main__":
    main()
