"""β=4 Strang gauge-fermion splitting test (Phase 35).

Doubled-lattice convention: N_E_strang = 2*N_E_orig − 1 slices, with
  - Half-step gauge transitions everywhere (K_E_half coupling)
  - K_M applied only at ODD slices (intermediate gauge states)
  - T_F at ODD slices with FULL a_τ_orig
  - Boundary states at slices 0 and N_E_strang − 1 (both even)

If Strang is O(a_τ²) overall (vs Lie's O(a_τ)), we expect the C(t) gap at
β=4 to be significantly smaller than the Lie scan from Phase 29 at the
same nominal a_τ.
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

A_TAUS_ORIG = [0.5, 0.333333333, 0.25]
N_CHAINS = 4
N_SWEEPS = 50_000
N_WARMUP = 5_000
TIMES = [0.0, 0.5, 1.0, 2.0]

OUT_DIR = "/home/hlamm/Desktop/QC/logdet/m1_toy/_strang_pkl"
os.makedirs(OUT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.7)
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


def run_one_atau(a_tau_orig, H, rho_ED_n, UOU_cache, n0_op, C_ED):
    """Strang at given a_τ_orig: doubled lattice with half-step gauge,
    T_F at odd slices with full a_τ_orig."""
    N_E_orig = int(round(BETA / a_tau_orig)) + 1
    N_E_strang = 2 * N_E_orig - 1  # ODD count
    a_half = a_tau_orig / 2.0
    print("=" * 80, flush=True)
    print(f"Strang: a_τ_orig={a_tau_orig:.5f}, N_E_strang={N_E_strang}, "
          f"a_half={a_half:.5f}", flush=True)
    print(f"  half-step β check: (N_E_strang−1)·a_half = "
          f"{(N_E_strang-1)*a_half:.4f} (target {BETA})", flush=True)
    print("=" * 80, flush=True)

    # Suzuki-mapped couplings at HALF-step
    K_E_half = -0.5 * np.log(np.tanh(a_half * G_E_HAM))
    # K_M at ODD slices with FULL a_τ weight (one H_M per Strang step)
    K_M_full = a_tau_orig * G_M_HAM
    m_action = a_half * M_HAM   # action mass scales with the MC time step (a_half)
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E_strang, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_half)

    print(f"  K_E_half={K_E_half:.4f}, K_M_full={K_M_full:.4f}\n", flush=True)

    C_chains = {t: [] for t in TIMES}
    rho_chains = []
    t_total0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
        t0 = time.time()
        print(f"  chain {ci+1}/{N_CHAINS} (seed={seed})...", flush=True)
        pt_results, _ = run_metropolis_pt(
            geom, K_E_ladder=K_E_ladder, K_M=K_M_full,
            n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=5,
            seed=seed, action_type='gauge_only', n_workers=6,
            strang_M=True,
        )
        top = pt_results[-1]
        configs = top.configs
        signs = np.array(top.sign_history, dtype=int)

        V3 = geom.V_3
        idx_to_psi = _fock_index_to_psi_map(V3)
        rho = np.zeros((256, 256), dtype=complex)
        for U in configs:
            W_full, _ = compute_combined_weight_trotter(
                geom, U, a_tau=a_half,           # for the unused gauge K params inside W
                K_E=0.0, K_M=0.0,
                m_obs=M_HAM, g_hop=G_HOP, order=W_ORDER,
                strang=True, a_tau_F=a_tau_orig,
            )
            W_psi = np.zeros((16, 16), dtype=complex)
            for li in range(16):
                for lj in range(16):
                    W_psi[idx_to_psi[li], idx_to_psi[lj]] = W_full[li, lj]
            # Boundary gauge states: slices 0 and N_E_strang-1 (both even)
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

    print(f"\n  Cross-chain at a_τ_orig={a_tau_orig:.4f} (walltime {total_wall:.0f}s):")
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

    pkl_path = os.path.join(OUT_DIR, f"strang_atau_{a_tau_orig:.4f}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'a_tau_orig': a_tau_orig, 'N_E_strang': N_E_strang,
            'BETA': BETA, 'M_HAM': M_HAM,
            'rho_combined': rho_combined,
            'eigs': eigs,
            'C_chains': C_chains,
            'walltime_total': total_wall,
        }, f)
    print(f"  saved to {pkl_path}", flush=True)
    return summary


def main():
    print(f"β=4 Strang gauge-fermion test")
    print(f"  Params: m={M_HAM}, g_E={G_E_HAM}, g_M={G_M_HAM}, g_hop={G_HOP}")
    print(f"  a_τ_orig values: {A_TAUS_ORIG}")
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
    for a_tau in A_TAUS_ORIG:
        s = run_one_atau(a_tau, H, rho_ED_n, UOU_cache, n0_op, C_ED)
        summaries[a_tau] = s
        print(flush=True)

    # Comparison table: Strang vs Lie (Phase 29 numbers)
    print("=" * 80)
    print("Strang vs Lie comparison (C(t) gap to ED, β=4)")
    print("=" * 80)
    # Lie data from check_beta4_atau_scan.py / check_beta4_gauge_only.py:
    LIE_GAPS = {
        0.5:   {0.0: +0.00065, 0.5: +0.00481, 1.0: -0.02573, 2.0: -0.01479},
        0.333: {0.0: -0.00331, 0.5: +0.00570, 1.0: -0.02388, 2.0: -0.01538},
        0.25:  {0.0: -0.00573, 0.5: +0.00668, 1.0: -0.02220, 2.0: -0.01541},
    }
    for a_tau in A_TAUS_ORIG:
        a_lie_key = 0.333 if abs(a_tau - 0.333333333) < 1e-4 else a_tau
        print(f"\n  a_τ_orig = {a_tau:.4f}")
        print(f"  {'t':>5} | {'gap_Strang':>12} | {'gap_Lie':>12} | "
              f"{'ratio':>10}")
        for t in TIMES:
            gap_s = summaries[a_tau][t][2]
            gap_l = LIE_GAPS[a_lie_key].get(t)
            ratio = gap_s / gap_l if gap_l != 0 else float('inf')
            print(f"  {t:>5.2f} | {gap_s:>+12.5f} | {gap_l:>+12.5f} | "
                  f"{ratio:>+10.3f}")


if __name__ == "__main__":
    main()
