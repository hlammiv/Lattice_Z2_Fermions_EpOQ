"""6-hour, 16-core (2-parallel × 8 replicas) run at a_τ=0.25 to gain statistics.

Architecture:
  - 8 PT instances, each 500k sweeps after 5k warmup → 4M total samples
  - ProcessPoolExecutor(2): runs 2 PT instances concurrently
  - Each PT internally uses Pool(8) for replicas
  - Total active workers: 2 × 8 = 16 cores, leaves 4 free
  - Per-instance walltime ~92 min, 4 batches of 2-parallel ≈ 6 hours wall

Saves per-instance ρ̃ pickle so we can compute cumulative-k stats later
without rerunning."""
from __future__ import annotations
import sys, time, os, pickle
import numpy as np
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
N_WARMUP, N_SWEEPS, SWAP_EVERY = 5000, 500000, 5
N_INSTANCES = 8
W_ORDER = 2
TIMES = [0.0, 0.5, 1.0, 2.0]
N_PARALLEL = 2   # 2 PT instances at once × 8 cores each = 16 cores
CKPT_DIR = "/home/hlamm/Desktop/QC/logdet/m1_toy/_ckpt_big_run"  # persistent across /tmp reboots
os.makedirs(CKPT_DIR, exist_ok=True)


def build_K_E_ladder(K_E_target, n_replicas=8):
    K_E_low = max(0.10, K_E_target - 0.7)
    if K_E_low >= K_E_target - 0.05:
        K_E_low = K_E_target - 0.05
    return list(np.linspace(K_E_low, K_E_target, n_replicas))


def worker_instance(args):
    """Worker: run one PT instance, build Hermitized ρ̃, return + pickle to disk."""
    instance_idx, seed = args
    ckpt_path = os.path.join(CKPT_DIR, f"inst_{instance_idx}_seed_{seed}.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        return instance_idx, data

    K_E_target = -0.5 * np.log(np.tanh(A_TAU * G_E_HAM))
    K_M = A_TAU * G_M_HAM
    N_E = int(round(BETA / A_TAU)) + 1
    m_action = A_TAU * M_HAM
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=N_E, m=m_action)
    K_E_ladder = build_K_E_ladder(K_E_target)
    V3 = geom.V_3
    idx_to_psi = _fock_index_to_psi_map(V3)

    t0 = time.time()
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
    twall = time.time() - t0
    data = {'rho': rho, 'sgn': float(signs.mean()), 'walltime': twall, 'seed': seed,
            'n_samples': len(configs)}
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    return instance_idx, data


def main():
    print("=" * 80, flush=True)
    print(f"BIG-STATS run @ a_τ={A_TAU}, m={M_HAM}, β={BETA}, palindrome W")
    print(f"  {N_INSTANCES} INDEPENDENT instances × {N_SWEEPS} sweeps each, "
          f"{N_PARALLEL}-parallel")
    print(f"  Total {N_INSTANCES * N_SWEEPS} samples; ckpt dir {CKPT_DIR}")
    print("=" * 80, flush=True)

    # H_QC, ED ref
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
    rho_ED = expm(-BETA * H); rho_ED_n = rho_ED / np.trace(rho_ED).real
    n0_op = 0.5 * np.eye(256, dtype=complex) - 0.5 * pauli([(4, 'Z')])
    cached_UOU = build_trottered_UOU_cache(TIMES, n_trotter=200, order=2)
    refs = {}
    for t in TIMES:
        Ut = expm(-1j * H * t)
        refs[t] = (np.trace(rho_ED @ (Ut.conj().T @ n0_op @ Ut) @ n0_op) / np.trace(rho_ED)).real
    print(f"ED ref: " + "  ".join(f"C({t})={refs[t]:+.5f}" for t in TIMES), flush=True)

    # Dispatch jobs in parallel
    jobs = [(i, 2026 + i * 11111) for i in range(N_INSTANCES)]
    results = [None] * N_INSTANCES
    t_start = time.time()
    print(f"\nLaunching {N_INSTANCES} instances with {N_PARALLEL}-parallel...",
          flush=True)
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
        future_to_idx = {ex.submit(worker_instance, job): job[0] for job in jobs}
        for fut in as_completed(future_to_idx):
            inst_idx, data = fut.result()
            results[inst_idx] = data
            elapsed = (time.time() - t_start) / 60
            print(f"  [{elapsed:5.1f} min] inst {inst_idx+1} done: "
                  f"⟨sgn⟩={data['sgn']:+.3f}, "
                  f"PT+ρ̃ walltime={data['walltime']:.0f}s, "
                  f"samples={data['n_samples']}", flush=True)

    # Compile combined ρ̃
    rhos = [r['rho'] for r in results]
    rho_combined = np.mean(rhos, axis=0)
    rho_combined_n = rho_combined / np.trace(rho_combined).real

    print(f"\n{'='*80}")
    print(f"Combined ρ̃ stats ({N_INSTANCES} instances × {N_SWEEPS} sweeps):")
    eigs = np.real(np.linalg.eigvalsh(rho_combined_n))
    eigs.sort()
    n_neg = int((eigs < -1e-6).sum())
    print(f"  min_eig = {eigs[0]:+.4e}, max_eig = {eigs[-1]:+.4e}, # neg = {n_neg}/256")
    print(f"  ||Δρ(β=4)||_F = {np.linalg.norm(rho_combined_n - rho_ED_n):.4f}")

    # C(t) from combined ρ̃ AND per-instance
    print(f"\nC(t) at combined ρ̃ vs ED:")
    print(f"  {'t':>5} | {'C(t) combined':>14} | {'ED ref':>10} | gap")
    Ct_per_inst = {t: [] for t in TIMES}
    for r in results:
        rn = r['rho'] / np.trace(r['rho']).real
        for t in TIMES:
            Ct_per_inst[t].append(np.real(np.trace(rn @ cached_UOU[t] @ n0_op)))
    for t in TIMES:
        Ct_comb = np.real(np.trace(rho_combined_n @ cached_UOU[t] @ n0_op))
        print(f"  {t:>3.1f} | {Ct_comb:>+14.5f} | {refs[t]:>+10.5f} | "
              f"{Ct_comb - refs[t]:+.5f}")

    # Cumulative scaling
    print(f"\nCumulative |min_eig| over first k instances:")
    print(f"  {'k':>3} | {'|min_eig|':>10} | {'||Δρ||':>10} | "
          + "  ".join(f"{'C('+f'{t:.1f}'+')':>9}" for t in TIMES))
    for k in (1, 2, 4, 8):
        if k > N_INSTANCES: continue
        rk = np.mean(rhos[:k], axis=0)
        rk_n = rk / np.trace(rk).real
        ek = np.real(np.linalg.eigvalsh(rk_n))
        ek.sort()
        d = np.linalg.norm(rk_n - rho_ED_n)
        Cts = [np.real(np.trace(rk_n @ cached_UOU[t] @ n0_op)) for t in TIMES]
        print(f"  {k:>3d} | {ek[0]:>+10.4e} | {d:>10.4f} | "
              + "  ".join(f"{c:>+9.5f}" for c in Cts))

    print(f"\nTotal walltime: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
