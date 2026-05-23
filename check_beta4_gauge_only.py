"""β=4 test of the gauge_only MC.  Key question: does removing det M let
C(t) match ED at the methodology-paper temperature?

Setup matches the big_run.py we killed, but with action_type='gauge_only'.
Single a_τ=0.5 (N_E=9) point first; if clean, expand to an a_τ scan.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

# IMPORTANT: methodology paper at m=2.  Set this BEFORE other imports that
# might cache build_pauli_terms() on the old default.
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
A_TAU = 0.5
N_E = int(round(BETA / A_TAU)) + 1  # 9
W_ORDER = 2

N_CHAINS = 4
N_SWEEPS = 50_000
N_WARMUP = 5_000


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


def main():
    print("=" * 80, flush=True)
    print(f"β=4 test: gauge_only MC at a_τ={A_TAU}, N_E={N_E}, m={M_HAM}", flush=True)
    print(f"  {N_CHAINS} chains × {N_SWEEPS} sweeps each, 6-PT-replicas inside each", flush=True)
    print("=" * 80, flush=True)

    H = build_H_QC()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    # n_0 = (I - Z_q4)/2
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus

    TIMES = [0.0, 0.5, 1.0, 2.0]
    UOU_cache = {}
    for t in TIMES:
        Ut = expm(-1j * H * t)
        UOU_cache[t] = Ut.conj().T @ n0_op @ Ut
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU_cache[t] @ n0_op)) for t in TIMES}
    print(f"\nED reference: " + "  ".join(f"C({t})={C_ED[t]:+.5f}" for t in TIMES), flush=True)

    K_E = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    print(f"\nMC: K_E_target={K_E:.4f}, K_M={K_M:.4f}, m_action={m_action}")
    print(f"    action_type='gauge_only'\n", flush=True)

    K_E_ladder = build_K_E_ladder(K_E)

    C_chains = {t: [] for t in TIMES}
    rho_chains = []
    t0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
        print(f"  chain {ci+1}/{N_CHAINS} (seed={seed})...", flush=True)
        t_chain = time.time()
        pt_results, _ = run_metropolis_pt(
            geom, K_E_ladder=K_E_ladder, K_M=K_M,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6)
        top = pt_results[-1]
        configs = top.configs
        signs = np.array(top.sign_history, dtype=int)

        V3 = geom.V_3
        idx_to_psi = _fock_index_to_psi_map(V3)
        rho = np.zeros((256, 256), dtype=complex)
        for U in configs:
            W_full, _ = compute_combined_weight_trotter(
                geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
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
        rho = (rho + rho.conj().T) / 2
        rho_chains.append(rho)
        rn = rho / np.trace(rho).real
        for t in TIMES:
            C_chains[t].append(np.real(np.trace(rn @ UOU_cache[t] @ n0_op)))
        twall_chain = time.time() - t_chain
        print(f"    ⟨sgn⟩={float(signs.mean()):+.4f}, walltime={twall_chain:.0f}s",
              flush=True)
        print(f"    " + "  ".join(f"C({t})={C_chains[t][-1]:+.5f}" for t in TIMES),
              flush=True)

    walltime = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Cross-chain summary ({N_CHAINS} × {N_SWEEPS} sweeps, walltime {walltime:.0f}s)")
    print(f"{'='*80}")
    print(f"  {'t':>5} | {'⟨C⟩':>12} | {'σ_chains':>10} | {'C_ED':>10} | "
          f"{'gap':>10} | {'σ-units':>8}")
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(N_CHAINS)
        gap = mean - C_ED[t]
        sigma_units = gap / sem if sem > 0 else float('inf')
        print(f"  {t:>5.2f} | {mean:>+12.5f} | {sem:>10.5f} | {C_ED[t]:>+10.5f} | "
              f"{gap:>+10.5f} | {sigma_units:>+8.2f}")

    # Combined ρ̃
    rho_combined = np.mean(rho_chains, axis=0)
    rho_combined_n = rho_combined / np.trace(rho_combined).real
    eigs = np.real(np.linalg.eigvalsh(rho_combined_n))
    eigs.sort()
    print(f"\nCombined ρ̃: min_eig={eigs[0]:+.4e}, max_eig={eigs[-1]:+.4e}, "
          f"#neg={int((eigs<-1e-6).sum())}/256")
    print(f"  ‖ρ̃_lat - ρ̃_ED‖_F = {np.linalg.norm(rho_combined_n - rho_ED_n):.4f}")
    print(f"  C(t) from combined ρ̃:")
    for t in TIMES:
        C_lat = np.real(np.trace(rho_combined_n @ UOU_cache[t] @ n0_op))
        print(f"    t={t}: lat={C_lat:+.5f}, ED={C_ED[t]:+.5f}, gap={C_lat-C_ED[t]:+.5f}")


if __name__ == "__main__":
    main()
