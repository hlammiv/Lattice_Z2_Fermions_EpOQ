"""Cleaner scaling test: N_INSTANCES independent PT chains, each warmed and
thermalized separately. Build ρ̃_i per instance; report inter-instance SEM of
|min_eig| and ||Δρ(β=4)||. This tells us whether the eigenvalue indefiniteness
is a structural property of the lattice ρ̃ at this a_τ or just sampling noise.

16 instances × 16000 sweeps each, sequential (each uses Pool(8) internally,
no oversubscription on 20-CPU box)."""
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
from epoq_quantum_simulator import build_trottered_UOU_cache

M_HAM, G_E_HAM, G_M_HAM, BETA = 2.0, 1.0, 0.5, 4.0
A_TAU = 0.25
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 16000, 5
N_INSTANCES = 16
W_ORDER = 2
TIMES = [0.0, 0.5, 1.0, 2.0]


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.10, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.05:
        K_E_low = K_E_target - 0.05
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def build_rho_for_instance(seed):
    """Run one PT instance, build 256×256 ρ̃ from its samples, Hermitize."""
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

    rho = np.zeros((256, 256), dtype=complex)
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
        for psi_top in range(16):
            for psi_bot in range(16):
                bit_a = (psi_top << 4) | g_top
                bit_b = (psi_bot << 4) | g_bot
                rho[bit_a, bit_b] += w * W_psi[psi_top, psi_bot]
    rho = (rho + rho.conj().T) / 2
    return rho, float(signs.mean())


def main():
    print("=" * 80, flush=True)
    print(f"ρ̃ inter-instance test @ a_τ={A_TAU}, m={M_HAM}, β={BETA}, palindrome W")
    print(f"  {N_INSTANCES} INDEPENDENT instances, warm={N_WARMUP} swp={N_SWEEPS}")
    print("=" * 80, flush=True)

    # H_QC
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
    rho_ED = expm(-BETA * H); rho_ED_normed = rho_ED / np.trace(rho_ED).real
    betas = np.linspace(0.5, 8.0, 31)

    # Observable side: cached UOU and n_0 for C(t) computation
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    print(f"Building cached_UOU for times {TIMES}...", flush=True)
    cached_UOU = build_trottered_UOU_cache(TIMES, n_trotter=200, order=2)

    # ED reference C(t)
    refs = {}
    for t in TIMES:
        Ut = expm(-1j * H * t)
        refs[t] = (np.trace(rho_ED @ (Ut.conj().T @ n0_op @ Ut) @ n0_op)
                    / np.trace(rho_ED)).real
    print(f"ED C(t) ref: " + "  ".join(f"C({t})={refs[t]:+.5f}" for t in TIMES),
          flush=True)

    def compute_Cts_from_rho(rho_norm):
        """C(t) = Tr(ρ_norm · n_0(t) · n_0(0)) = Tr(ρ · UOU[t] · n_0)."""
        out = {}
        for t in TIMES:
            UOU = cached_UOU[t]
            out[t] = np.real(np.trace(rho_norm @ UOU @ n0_op))
        return out

    rhos = []
    per_inst_min_eig = []
    per_inst_sgn = []
    per_inst_dist_at4 = []
    per_inst_best_beta = []
    per_inst_best_dist = []
    per_inst_Cts = {t: [] for t in TIMES}
    print(f"\n{'inst':>4} | {'⟨sgn⟩':>7} | {'min_eig':>10} | "
          f"{'||Δρ||':>10} | " +
          "".join(f"{'C('+f'{t:.1f}'+')':>11} " for t in TIMES))
    print("-" * (40 + 12 * len(TIMES)))
    for i in range(N_INSTANCES):
        seed = 2026 + i * 11111
        t0 = time.time()
        rho_i, sgn_i = build_rho_for_instance(seed)
        rhos.append(rho_i)
        per_inst_sgn.append(sgn_i)
        rho_norm = rho_i / np.trace(rho_i).real
        eigs = np.real(np.linalg.eigvalsh(rho_norm))
        eigs.sort()
        min_e = eigs[0]
        d4 = np.linalg.norm(rho_norm - rho_ED_normed)
        best_b, best_d = None, float('inf')
        for be in betas:
            re = expm(-be * H); re /= np.trace(re).real
            d = np.linalg.norm(rho_norm - re)
            if d < best_d:
                best_d = d; best_b = be
        Cs_i = compute_Cts_from_rho(rho_norm)
        per_inst_min_eig.append(min_e)
        per_inst_dist_at4.append(d4)
        per_inst_best_beta.append(best_b)
        per_inst_best_dist.append(best_d)
        for t in TIMES:
            per_inst_Cts[t].append(Cs_i[t])
        twall = time.time() - t0
        print(f"{i+1:>4d} | {sgn_i:>+7.3f} | {min_e:>+10.4e} | "
              f"{d4:>10.4f} | " +
              "".join(f"{Cs_i[t]:>+10.5f} " for t in TIMES) +
              f" ({twall:.0f}s)", flush=True)

    # Combined ρ̃
    print(f"\nCombined ρ̃ (average over {N_INSTANCES} instances):")
    rho_combined = np.mean(rhos, axis=0)
    rho_combined_norm = rho_combined / np.trace(rho_combined).real
    eigs_combined = np.real(np.linalg.eigvalsh(rho_combined_norm))
    eigs_combined.sort()
    n_neg = int((eigs_combined < -1e-6).sum())
    d_combined = np.linalg.norm(rho_combined_norm - rho_ED_normed)
    best_b_c, best_d_c = None, float('inf')
    for be in betas:
        re = expm(-be * H); re /= np.trace(re).real
        d = np.linalg.norm(rho_combined_norm - re)
        if d < best_d_c:
            best_d_c = d; best_b_c = be
    print(f"  min_eig = {eigs_combined[0]:+.4e},  max_eig = {eigs_combined[-1]:+.4e},  "
          f"# neg = {n_neg}/256")
    print(f"  ||Δρ(β=4)|| = {d_combined:.4f},  "
          f"best β_eff = {best_b_c:.3f}, best ||Δρ|| = {best_d_c:.4f}")

    # C(t) from combined ρ̃ and inter-instance SEM
    Cs_combined = compute_Cts_from_rho(rho_combined_norm)
    print(f"\nC(t) results:")
    print(f"  {'t':>5} | {'C(t) combined':>14} | {'inter-inst mean ± SEM':>26} | "
          f"{'ED ref':>10} | {'σ from ED':>10}")
    for t in TIMES:
        vals = np.array(per_inst_Cts[t])
        m = vals.mean(); s = vals.std(ddof=1) / np.sqrt(len(vals))
        gap = m - refs[t]
        sigmas = abs(gap)/s if s > 0 else float('nan')
        print(f"  {t:>5.1f} | {Cs_combined[t]:>+14.5f} | "
              f"{m:>+12.5f} ± {s:>.5f} | "
              f"{refs[t]:>+10.5f} | {sigmas:>9.1f}σ")

    # --- Cumulative-instance scaling analysis ---
    # Build cumulative averages over k=1, 2, 4, 8, 16 instances and look at
    # |min_eig| as a function of k. If it scales as 1/sqrt(k), we're noise-
    # dominated. If it plateaus, we've reached the structural floor.
    print(f"\nCumulative-instance scaling of |min_eig| of the average ρ̃:")
    print(f"  {'k':>3} | {'|min_eig|':>10} | {'||Δρ(β=4)||':>11} | "
          f"{'1/sqrt(k) ratio':>16} | " +
          "".join(f"{'C('+f'{t:.1f}'+')':>11} " for t in TIMES))
    k_values = [k for k in (1, 2, 4, 8, 16) if k <= N_INSTANCES]
    baseline_min = None
    for k in k_values:
        rho_k = np.mean(rhos[:k], axis=0)
        rho_k_norm = rho_k / np.trace(rho_k).real
        eigs_k = np.real(np.linalg.eigvalsh(rho_k_norm))
        eigs_k.sort()
        mn = eigs_k[0]
        dk = np.linalg.norm(rho_k_norm - rho_ED_normed)
        Cs_k = compute_Cts_from_rho(rho_k_norm)
        if baseline_min is None:
            baseline_min = abs(mn)
            ratio_str = "(baseline)"
        else:
            expected = baseline_min / np.sqrt(k)
            ratio_str = f"{abs(mn)/expected:>5.2f}x exp"
        print(f"  {k:>3d} | {mn:>+10.4e} | {dk:>11.4f} | "
              f"{ratio_str:>16} | " +
              "".join(f"{Cs_k[t]:>+10.5f} " for t in TIMES))

    print(f"\nInter-instance stats:")
    me = np.array(per_inst_min_eig)
    print(f"  ⟨sgn⟩ per inst: mean={np.mean(per_inst_sgn):+.3f}, "
          f"std={np.std(per_inst_sgn, ddof=1):.3f}")
    print(f"  min_eig per inst: mean={me.mean():+.4f}, "
          f"std={me.std(ddof=1):.4f}, SEM={me.std(ddof=1)/np.sqrt(len(me)):.4f}")
    print(f"  Combined |min_eig| = {abs(eigs_combined[0]):.4e}  "
          f"vs single-inst |min_eig| mean = {np.abs(me).mean():.4e}")
    if abs(eigs_combined[0]) < 0.5 * np.abs(me).mean():
        print("  → Inter-instance averaging IMPROVED the indefiniteness "
              "(consistent with sampling noise).")
    else:
        print("  → Inter-instance averaging did NOT improve significantly "
              "(structural indefiniteness).")


if __name__ == "__main__":
    main()
