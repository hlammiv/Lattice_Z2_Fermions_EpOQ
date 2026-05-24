"""Phase 38 production verification: β=4, m=2 MC with ORDER=1 W
(time-ordered Lie, NOT palindrome).

If the order=1 fix carries to β=4 production, we expect C(t) closer to
ED than the prior W_ORDER=2 results.

Two a_τ values to check Trotter direction:
  a_τ=0.5, N_E=9
  a_τ=0.25, N_E=17
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

import epoq_classical_sampler as cs
cs.M_MASS = 2.0
import epoq_quantum_simulator as qs
qs.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, G_HOP = 2.0, 1.0, 0.5, 0.5
BETA = 4.0
W_ORDER = 1   # THE FIX — was 2 in the failing production runs

A_TAUS = [0.5, 0.25]
N_CHAINS = 4
N_SWEEPS = 30_000
N_WARMUP = 3_000
TIMES = [0.0, 0.5, 1.0, 2.0]


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.6)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


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


def run_one(a_tau, H, rho_ED_n, UOU_cache, n0_op, C_ED):
    N_E = int(round(BETA / a_tau)) + 1
    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    m_action = a_tau * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)
    print(f"\n  a_τ={a_tau}, N_E={N_E}, K_E={K_E:.4f}, K_M={K_M:.4f}", flush=True)

    C_chains = {t: [] for t in TIMES}
    rho_chains = []
    t_total0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
        t0 = time.time()
        pt_results, _ = run_metropolis_pt(
            geom, K_E_ladder=K_E_ladder, K_M=K_M,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6)
        top = pt_results[-1]
        configs = top.configs

        V3 = geom.V_3
        idx_to_psi = _fock_index_to_psi_map(V3)
        rho = np.zeros((256, 256), dtype=complex)
        for U in configs:
            W_full, _ = compute_combined_weight_trotter(
                geom, U, a_tau=a_tau, K_E=0.0, K_M=0.0,
                m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER)
            W_psi = np.zeros((16, 16), dtype=complex)
            for li in range(16):
                for lj in range(16):
                    W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
            g_top = gauge_qc_bits_from_slice(U, geom.N_E - 1)
            g_bot = gauge_qc_bits_from_slice(U, 0)
            for pt_idx in range(16):
                for pb_idx in range(16):
                    bit_a = (pt_idx << 4) | g_top
                    bit_b = (pb_idx << 4) | g_bot
                    rho[bit_a, bit_b] += W_psi[pt_idx, pb_idx]
        rho = (rho + rho.conj().T) / 2   # post-hoc Hermitization
        rho_chains.append(rho)
        rn = rho / np.trace(rho).real
        for t in TIMES:
            C_chains[t].append(np.real(np.trace(rn @ UOU_cache[t] @ n0_op)))
        twall = time.time() - t0
        print(f"    chain {ci+1}/{N_CHAINS}, walltime={twall:.0f}s, "
              + "  ".join(f"C({t})={C_chains[t][-1]:+.5f}" for t in TIMES),
              flush=True)

    total_wall = time.time() - t_total0
    rho_combined = np.mean(rho_chains, axis=0)
    rho_combined_n = rho_combined / np.trace(rho_combined).real
    eigs = np.real(np.linalg.eigvalsh(rho_combined_n)); eigs.sort()
    d_ED = np.linalg.norm(rho_combined_n - rho_ED_n)
    print(f"\n  Cross-chain at a_τ={a_tau} (walltime {total_wall:.0f}s):")
    print(f"  ‖Δρ‖_F = {d_ED:.4f}, min_eig={eigs[0]:+.4e}, "
          f"#neg = {int((eigs<-1e-6).sum())}/256")
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(N_CHAINS)
        gap = mean - C_ED[t]
        print(f"  t={t}: C_lat={mean:+.5f}±{sem:.5f}, C_ED={C_ED[t]:+.5f}, "
              f"gap={gap:+.5f}")
    return {t: (np.mean(C_chains[t]), np.std(C_chains[t], ddof=1)/np.sqrt(N_CHAINS),
                np.mean(C_chains[t]) - C_ED[t]) for t in TIMES}, d_ED


def main():
    print(f"β=4, m=2, ORDER=1 W (Lie time-ordered) verification")
    print(f"  Was previously failing with W_ORDER=2 (palindrome)")
    print(f"  Free U_t MC, gauge_only, {N_CHAINS} chains × {N_SWEEPS} sweeps")

    H = build_H_QC()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    UOU = {t: expm(-1j * H * t).conj().T @ n0_op @ expm(-1j * H * t) for t in TIMES}
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op)) for t in TIMES}
    print(f"\nED reference: " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES))

    results = {}
    for a_tau in A_TAUS:
        results[a_tau], _ = run_one(a_tau, H, rho_ED_n, UOU, n0_op, C_ED)

    # Compare to prior palindrome data from check_beta4_atau_scan.py
    PALINDROME_PRIOR = {
        0.5:  {0.0: +0.00065, 0.5: +0.00481, 1.0: -0.02573, 2.0: -0.01479},
        0.25: {0.0: -0.00573, 0.5: +0.00668, 1.0: -0.02220, 2.0: -0.01541},
    }
    print("\n" + "=" * 80)
    print("β=4 m=2 — Order=1 vs palindrome comparison (gap = C_lat − C_ED)")
    print("=" * 80)
    for a_tau in A_TAUS:
        print(f"\na_τ={a_tau}")
        print(f"  {'t':>5} | {'order=1 gap':>14} | {'palindrome gap':>16} | {'improvement':>12}")
        for t in TIMES:
            gap_new = results[a_tau][t][2]
            gap_old = PALINDROME_PRIOR[a_tau][t]
            improvement = abs(gap_old) - abs(gap_new)
            print(f"  {t:>5.2f} | {gap_new:>+14.5f} | {gap_old:>+16.5f} | "
                  f"{improvement:>+12.5f}")


if __name__ == "__main__":
    main()
