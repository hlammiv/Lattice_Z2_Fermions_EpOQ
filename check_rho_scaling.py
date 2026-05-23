"""Scaling test: build ρ̃_lat from subsets of MC samples (N_use = 100, 250, 500,
1000, 2000, 4000, 8000, 16000) and check whether the min eigenvalue and
||Δρ||_F vs ED shrink with N. If so, ρ̃ → Hermitian PD asymptotically and the
indefiniteness is a finite-sample artifact. If not, it's structural."""
from __future__ import annotations
import sys, time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_quantum_simulator; epoq_quantum_simulator.M_MASS = 2.0
import epoq_classical_sampler; epoq_classical_sampler.M_MASS = 2.0

from action_z2_staggered import LatticeGeometry
from action_z2_metropolis_pt import run_metropolis_pt
from transfer_matrix_kbc_trotterized import compute_combined_weight_trotter
from action_minkowski_stitch import gauge_qc_bits_from_slice
from action_corner_direct_v3 import _fock_index_to_psi_map

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
A_TAU = 0.25
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 64000, 5
N_INSTANCES = 4
W_ORDER = 2


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.10, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.05:
        K_E_low = K_E_target - 0.05
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def collect_sample_contributions(seed):
    """Run a PT instance; return list of per-config (16x16 W_psi, g_top, g_bot, sign)."""
    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)

    pt_results, _ = run_metropolis_pt(
        geom, K_E_ladder=K_E_ladder, K_M=K_M,
        n_sweeps=N_SWEEPS, n_warmup=N_WARMUP, swap_every=SWAP_EVERY,
        seed=seed, action_type='forward')
    top = pt_results[-1]
    configs = top.configs
    signs = np.array(top.sign_history, dtype=int)

    samples = []
    for U_idx, U in enumerate(configs):
        w = float(signs[U_idx])
        W_full, _ = compute_combined_weight_trotter(
            geom, U, a_tau=A_TAU, K_E=0.0, K_M=0.0,
            m_obs=M_HAM, g_hop=0.5, order=W_ORDER)
        W_psi = np.zeros((16, 16), dtype=complex)
        for lex_i in range(16):
            for lex_j in range(16):
                W_psi[idx_to_psi[lex_i], idx_to_psi[lex_j]] = W_full[lex_i, lex_j]
        g_top = gauge_qc_bits_from_slice(U, geom.N_E - 1)
        g_bot = gauge_qc_bits_from_slice(U, 0)
        samples.append((w, W_psi, g_top, g_bot))
    return samples


def build_rho_from_samples(samples):
    rho = np.zeros((256, 256), dtype=complex)
    for w, W_psi, g_top, g_bot in samples:
        for psi_top in range(16):
            for psi_bot in range(16):
                bit_a = (psi_top << 4) | g_top
                bit_b = (psi_bot << 4) | g_bot
                rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    rho = (rho + rho.conj().T) / 2   # Hermitize per EρOQ
    return rho


def main():
    print("=" * 80, flush=True)
    print(f"ρ̃ scaling test @ a_τ={A_TAU}, m={M_HAM}, β_bare={BETA}, palindrome W")
    print("=" * 80, flush=True)

    # Build H_QC
    P2 = {'I': np.eye(2, dtype=complex), 'X':np.array([[0,1],[1,0]], dtype=complex),
          'Y':np.array([[0,-1j],[1j,0]], dtype=complex),
          'Z':np.diag([1,-1]).astype(complex)}
    def pauli(factors, nq=8):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors: by_q[q] = P2[ax]
        r = by_q[nq-1]
        for q in range(nq-2, -1, -1): r = np.kron(r, by_q[q])
        return r
    H = sum(c * pauli(f) for c, f in epoq_classical_sampler.build_pauli_terms())
    H = (H + H.conj().T) / 2
    rho_ED_4 = expm(-4.0 * H); rho_ED_4 /= np.trace(rho_ED_4).real

    # Collect ALL samples from all instances first
    print("Collecting MC samples...", flush=True)
    all_samples = []
    for i in range(N_INSTANCES):
        seed = 2026 + i * 11111
        t0 = time.time()
        sams = collect_sample_contributions(seed)
        all_samples.extend(sams)
        print(f"  inst {i+1}: {len(sams)} samples, {time.time()-t0:.0f}s",
              flush=True)
    print(f"Total: {len(all_samples)} samples", flush=True)

    # Build ρ̃ at increasing subsets
    N_uses = [1000, 4000, 16000, 64000, len(all_samples)]
    print(f"\n{'N_use':>7} | {'min_eig':>10} | {'max_eig':>10} | "
          f"{'# neg':>6} | {'||Δρ(β=4)||':>12} | {'best β_eff':>10} | {'best ||Δρ||':>12}")
    print("-" * 90)
    results = []
    betas = np.linspace(0.5, 8.0, 31)
    for N_use in N_uses:
        sub = all_samples[:N_use]
        rho = build_rho_from_samples(sub)
        if abs(np.trace(rho).real) < 1e-12:
            print(f"{N_use:>7d} | empty/degenerate")
            continue
        rho_normed = rho / np.trace(rho).real
        eigs = np.real(np.linalg.eigvalsh(rho_normed))
        eigs.sort()
        min_eig, max_eig = eigs[0], eigs[-1]
        n_neg = int((eigs < -1e-6).sum())
        d_at4 = np.linalg.norm(rho_normed - rho_ED_4)
        best_beta = None; best_d = float('inf')
        for be in betas:
            re = expm(-be * H); re /= np.trace(re).real
            d = np.linalg.norm(rho_normed - re)
            if d < best_d: best_d = d; best_beta = be
        results.append((N_use, min_eig, max_eig, n_neg, d_at4, best_beta, best_d))
        print(f"{N_use:>7d} | {min_eig:>+10.4e} | {max_eig:>+10.4e} | "
              f"{n_neg:>6d} | {d_at4:>12.4f} | {best_beta:>10.3f} | {best_d:>12.4f}")

    # Save results so we can fit scaling
    print(f"\nScaling check (min eigenvalue vs N):")
    print(f"  If finite-sample: |min_eig| ~ 1/sqrt(N)")
    print(f"  If structural:    |min_eig| ~ const")
    print(f"\n  log10(N) -> log10(|min_eig|):")
    for N, m, *_ in results:
        print(f"    {np.log10(N):.2f} -> {np.log10(abs(m)):+.2f}  (|min|={abs(m):.4f})")


if __name__ == "__main__":
    main()
