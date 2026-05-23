"""Intermediate-β diagnostic.

The β=4 m=2 scans (Lie + Strang) both converge to a value DIFFERENT from
ED as a_τ → 0.  But the β=0.5 m=0.5 deterministic test matched ED to 10⁻⁴.

Disentangle (β, m) effects with a quick scan:
  Block A:  β=2, m=0.5   (low-β/low-m baseline)
  Block B:  β=2, m=2     (low-β + high-m)
  Block C:  β=4, m=0.5   (high-β + low-m)

If A converges to ED but B doesn't → mass is the problem
If A and B converge but C doesn't → β is the problem
If none converge → deeper structural issue

3 a_τ values per block: 0.5, 0.25, 0.125 (N_E up to 17).
4 chains × 20k sweeps each (lighter than before).
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
import epoq_quantum_simulator as qs

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

G_E_HAM, G_M_HAM, G_HOP = 1.0, 0.5, 0.5
W_ORDER = 2

BLOCKS = [
    ("A",  2.0, 0.5),
    ("B",  2.0, 2.0),
    ("C",  4.0, 0.5),
]
A_TAUS = [0.5, 0.25, 0.125]
N_CHAINS = 4
N_SWEEPS = 20_000
N_WARMUP = 3_000
TIMES = [0.0, 0.5, 1.0]

OUT_DIR = "/home/hlamm/Desktop/QC/logdet/m1_toy/_intermediate_beta_pkl"
os.makedirs(OUT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=6):
    K_E_low = max(0.05, K_E_target - 0.7)
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def build_H_QC(m_hamilt):
    """Rebuild H_QC with the desired mass.  We override cs.M_MASS for the
    Pauli-term build, then restore so other modules don't get confused."""
    saved = cs.M_MASS
    cs.M_MASS = m_hamilt
    qs.M_MASS = m_hamilt
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
    cs.M_MASS = saved  # restore (but we'll set it again per block)
    return (H + H.conj().T) / 2


def build_n0_op():
    Z = np.diag([1.0, -1.0]).astype(complex)
    n0_minus = np.array([[1.0]], dtype=complex)
    for k in reversed(range(8)):
        m = Z if k == 4 else np.eye(2, dtype=complex)
        n0_minus = np.kron(n0_minus, m)
    return 0.5 * np.eye(256, dtype=complex) - 0.5 * n0_minus


def run_one(block_label, beta, m_hamilt, a_tau):
    """One Lie MC scan point at given (β, m, a_τ)."""
    N_E = int(round(beta / a_tau)) + 1
    cs.M_MASS = m_hamilt   # set for build_pauli_terms even if rebuild happens later
    qs.M_MASS = m_hamilt
    print(f"  [{block_label}] β={beta}, m={m_hamilt}, a_τ={a_tau} → N_E={N_E}",
          flush=True)

    K_E = -0.5 * np.log(np.tanh(a_tau * G_E_HAM))
    K_M = a_tau * G_M_HAM
    m_action = a_tau * m_hamilt
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E)

    C_chains = {t: [] for t in TIMES}
    t0 = time.time()
    for ci in range(N_CHAINS):
        seed = 2026 + ci * 11111
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
                m_obs=m_hamilt, g_hop=G_HOP, order=W_ORDER)
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
        rn = rho / np.trace(rho).real
        UOU = _MAYBE_CACHE[(beta, m_hamilt)]['UOU']
        n0op = _MAYBE_CACHE[(beta, m_hamilt)]['n0']
        for t in TIMES:
            C_chains[t].append(np.real(np.trace(rn @ UOU[t] @ n0op)))
    twall = time.time() - t0
    summary = {}
    for t in TIMES:
        vals = np.array(C_chains[t])
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(N_CHAINS) if N_CHAINS > 1 else 0.0
        summary[t] = (mean, sem)
    return summary, twall


_MAYBE_CACHE = {}


def main():
    print(f"Intermediate-β diagnostic")
    print(f"  Blocks: " + ", ".join(f"{lab}(β={b},m={m})" for lab,b,m in BLOCKS))
    print(f"  a_τ scan: {A_TAUS}")
    print(f"  {N_CHAINS} chains × {N_SWEEPS} sweeps", flush=True)
    print(f"  Output: {OUT_DIR}\n")

    n0_op = build_n0_op()
    # Precompute ED references for all (β, m) blocks
    ed_refs = {}
    for label, beta, m_hamilt in BLOCKS:
        H = build_H_QC(m_hamilt)
        rho_ED = expm(-beta * H)
        rho_ED_n = rho_ED / np.trace(rho_ED).real
        UOU = {}
        for t in TIMES:
            Ut = expm(-1j * H * t)
            UOU[t] = Ut.conj().T @ n0_op @ Ut
        C_ED = {t: np.real(np.trace(rho_ED_n @ UOU[t] @ n0_op)) for t in TIMES}
        ed_refs[(beta, m_hamilt)] = C_ED
        _MAYBE_CACHE[(beta, m_hamilt)] = {'UOU': UOU, 'n0': n0_op}
        print(f"  [{label}] β={beta}, m={m_hamilt}: " +
              "  ".join(f"C({t})_ED={C_ED[t]:+.5f}" for t in TIMES))

    print("\n")
    # Per-block scans
    all_results = {}
    for label, beta, m_hamilt in BLOCKS:
        print(f"{'='*80}")
        print(f"BLOCK {label}: β={beta}, m={m_hamilt}")
        print(f"{'='*80}", flush=True)
        block_results = {}
        for a_tau in A_TAUS:
            summary, twall = run_one(label, beta, m_hamilt, a_tau)
            block_results[a_tau] = summary
            C_ED = ed_refs[(beta, m_hamilt)]
            gap_strs = "  ".join(
                f"C({t})_lat={summary[t][0]:+.5f} gap={summary[t][0]-C_ED[t]:+.5f}"
                for t in TIMES)
            print(f"    walltime={twall:.0f}s | {gap_strs}", flush=True)
        all_results[label] = block_results
        # Save pickle
        with open(os.path.join(OUT_DIR, f"block_{label}.pkl"), 'wb') as f:
            pickle.dump({'block_results': block_results, 'C_ED': ed_refs[(beta, m_hamilt)],
                         'beta': beta, 'm': m_hamilt}, f)
        print()

    # Final comparison table
    print("=" * 80)
    print("FINAL CONVERGENCE TABLE — gap (C_lat − C_ED) by (a_τ, t)")
    print("=" * 80)
    for label, beta, m_hamilt in BLOCKS:
        C_ED = ed_refs[(beta, m_hamilt)]
        print(f"\n[{label}] β={beta}, m={m_hamilt}")
        header = "  a_τ       | " + " | ".join(f"C({t})_ED={C_ED[t]:+.4f}".rjust(20) for t in TIMES)
        print(header)
        for a_tau in A_TAUS:
            row = f"  {a_tau:>7.4f}  | "
            row += " | ".join(
                f"gap={all_results[label][a_tau][t][0]-C_ED[t]:+.5f}".rjust(20)
                for t in TIMES)
            print(row)
        # Trend: is C(0) approaching ED?
        c0_traj = [all_results[label][a_tau][0.0][0] for a_tau in A_TAUS]
        c0_target = C_ED[0.0]
        print(f"  C(0) trajectory: {c0_traj} → ED {c0_target:+.5f}")


if __name__ == "__main__":
    main()
