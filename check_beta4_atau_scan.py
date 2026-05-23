"""β=4 a_τ scan: {0.333, 0.25} with gauge_only MC.

Combined with the existing a_τ=0.5 run (in /tmp/beta4_gauge_only.log) we get
3 points for Trotter-trend analysis: a_τ ∈ {0.5, 0.333, 0.25}.

Same setup as check_beta4_gauge_only.py: m=2, β=4, 4 chains × 50k sweeps each,
6 PT replicas inside each chain.  Stores ρ̃ pickles for later analysis.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, time, pickle
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
W_ORDER = 2

A_TAUS = [0.333333333, 0.25]
N_CHAINS = 4
N_SWEEPS = 50_000
N_WARMUP = 5_000
TIMES = [0.0, 0.5, 1.0, 2.0]

OUT_DIR = "/home/hlamm/Desktop/QC/logdet/m1_toy/_atau_scan_pkl"
os.makedirs(OUT_DIR, exist_ok=True)


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


def build_n0_op():
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    return 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus


def run_one_atau(a_tau, H, rho_ED_n, UOU_cache, n0_op, C_ED):
    N_E = int(round(BETA / a_tau)) + 1
    print("=" * 80, flush=True)
    print(f"a_τ = {a_tau:.5f}, N_E = {N_E}, β = {(N_E-1)*a_tau:.4f}", flush=True)
    print("=" * 80, flush=True)

    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    m_action = a_tau * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)

    C_chains = {t: [] for t in TIMES}
    rho_chains = []
    t_total0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
        t0 = time.time()
        print(f"  chain {ci+1}/{N_CHAINS} (seed={seed})...", flush=True)
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
        rho = (rho + rho.conj().T) / 2
        rho_chains.append(rho)
        rn = rho / np.trace(rho).real
        for t in TIMES:
            C_chains[t].append(np.real(np.trace(rn @ UOU_cache[t] @ n0_op)))
        twall = time.time() - t0
        print(f"    ⟨sgn⟩={float(signs.mean()):+.4f}, walltime={twall:.0f}s",
              flush=True)
        print(f"    " + "  ".join(f"C({t})={C_chains[t][-1]:+.5f}" for t in TIMES),
              flush=True)

    total_wall = time.time() - t_total0
    rho_combined = np.mean(rho_chains, axis=0)
    rho_combined_n = rho_combined / np.trace(rho_combined).real
    eigs = np.real(np.linalg.eigvalsh(rho_combined_n))
    eigs.sort()

    print(f"\n  Cross-chain at a_τ={a_tau:.4f} (walltime {total_wall:.0f}s):")
    print(f"  {'t':>5} | {'⟨C⟩':>12} | {'σ':>10} | {'C_ED':>10} | "
          f"{'gap':>10}")
    summary = {}
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(N_CHAINS)
        gap = mean - C_ED[t]
        summary[t] = (mean, sem, gap)
        print(f"  {t:>5.2f} | {mean:>+12.5f} | {sem:>10.5f} | {C_ED[t]:>+10.5f} | "
              f"{gap:>+10.5f}")
    print(f"  ρ̃ min_eig={eigs[0]:+.4e}, max_eig={eigs[-1]:+.4e}, "
          f"#neg={int((eigs<-1e-6).sum())}/256")
    print(f"  ‖ρ̃_lat - ρ̃_ED‖_F = {np.linalg.norm(rho_combined_n - rho_ED_n):.4f}")

    # Save for later analysis
    pkl_path = os.path.join(OUT_DIR, f"atau_{a_tau:.4f}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'a_tau': a_tau, 'N_E': N_E, 'BETA': BETA, 'M_HAM': M_HAM,
            'rho_combined': rho_combined,
            'eigs': eigs,
            'C_chains': C_chains,
            'walltime_total': total_wall,
        }, f)
    print(f"  saved to {pkl_path}", flush=True)
    return summary


def main():
    print(f"β=4 a_τ scan, gauge_only MC")
    print(f"  Params: m={M_HAM}, g_E={G_E_HAM}, g_M={G_M_HAM}, g_hop={G_HOP}")
    print(f"  Scan a_τ values: {A_TAUS}")
    print(f"  {N_CHAINS} chains × {N_SWEEPS} sweeps, 6 PT replicas inside each")
    print(f"  Output dir: {OUT_DIR}\n", flush=True)

    H = build_H_QC()
    rho_ED = expm(-BETA * H)
    rho_ED_n = rho_ED / np.trace(rho_ED).real

    n0_op = build_n0_op()
    UOU_cache = {}
    for t in TIMES:
        Ut = expm(-1j * H * t)
        UOU_cache[t] = Ut.conj().T @ n0_op @ Ut
    C_ED = {t: np.real(np.trace(rho_ED_n @ UOU_cache[t] @ n0_op)) for t in TIMES}
    print(f"ED reference (β={BETA}, m={M_HAM}):")
    for t in TIMES:
        print(f"  C({t}) = {C_ED[t]:+.5f}")

    summaries = {}
    for a_tau in A_TAUS:
        s = run_one_atau(a_tau, H, rho_ED_n, UOU_cache, n0_op, C_ED)
        summaries[a_tau] = s
        print(flush=True)

    # Final scan table including a_τ=0.5 from prior run
    print("=" * 80)
    print("Trotter scan summary (incl. a_τ=0.5 from prior gauge_only run):")
    print("=" * 80)
    prior_atau05 = {0.0: 0.01417, 0.5: -0.00774, 1.0: -0.01246, 2.0: -0.00184}
    prior_taus = {**{0.5: prior_atau05}, **summaries}
    print(f"  {'t':>5} | {'C_ED':>10} | "
          f"{'a_τ=0.5':>10} | {'a_τ=0.333':>10} | {'a_τ=0.25':>10}")
    for t in TIMES:
        row = f"  {t:>5.2f} | {C_ED[t]:>+10.5f} | "
        for atau in [0.5, A_TAUS[0], A_TAUS[1]]:
            if atau == 0.5:
                row += f"{prior_atau05[t]:>+10.5f} | "
            else:
                row += f"{summaries[atau][t][0]:>+10.5f} | "
        print(row)

    print(f"\n  gaps (C_lat − C_ED):")
    print(f"  {'t':>5} | "
          f"{'a_τ=0.5':>10} | {'a_τ=0.333':>10} | {'a_τ=0.25':>10}")
    for t in TIMES:
        row = f"  {t:>5.2f} | "
        for atau in [0.5, A_TAUS[0], A_TAUS[1]]:
            if atau == 0.5:
                gap = prior_atau05[t] - C_ED[t]
            else:
                gap = summaries[atau][t][2]
            row += f"{gap:>+10.5f} | "
        print(row)


if __name__ == "__main__":
    main()
