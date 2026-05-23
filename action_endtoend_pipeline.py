"""
M_act_6: end-to-end EρOQ-faithful pipeline.

1. Classical action MC: sample gauge configs U via Metropolis on Z₂ links
   with weight e^(−S_g) · det M_OBC[U].
2. For each U: compute G^bdy[U] (direct OR pseudofermion variant), sample
   Fock-state corner pairs (Ψ_i, Ψ_j) from Slater-determinant weights.
3. For each (U, Ψ_i, Ψ_j): compute Minkowski-leg matrix element
   ⟨Ψ_j|U_M†(t) n_0 U_M(t)|Ψ_i⟩ via Trotter on H.
4. Aggregate via ratio estimator and bootstrap for error bars.

Compares against physical-sector ED reference (z2 setup) and against
Method 3 (Z-C) results from earlier sandbox.
"""

from __future__ import annotations
import sys
import time
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')

from action_z2_staggered import (
    LatticeGeometry, Z2GaugeConfig, build_dirac_matrix, compute_G_bdy,
    gauge_action,
)
from action_z2_metropolis import run_metropolis
from action_corner_direct import (
    fock_pair_weights_array, sample_corner_pairs_categorical,
)
from action_corner_pseudofermion import pseudofermion_G_bdy_estimate
from action_minkowski_stitch import (
    matrix_element_minkowski, qc_basis_index, static_observable_diagonal,
)


# ----------------------------------------------------------------------------
# Reference: physical-sector ED on the 2x2 Hilbert space at finite β = N_E·a
# (8 qubits, same as z2_setup.py). Provides ⟨n_0(t) n_0(0)⟩_β.
# ----------------------------------------------------------------------------
def physical_sector_reference(beta: float, times: list[float]) -> dict:
    """Compute C(t) = ⟨n_0(t) n_0⟩_β in the physical sector via ED.

    Uses the 8-qubit z2_setup convention. Beta should equal N_E·a in
    natural units; for our toy a = 1, so β = N_E.
    """
    NQ, DIM = 8, 256
    P2 = {
        'I': np.eye(2, dtype=complex), 'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex), 'Z': np.diag([1,-1]).astype(complex),
    }
    def pauli_term_dense(factors, nq=NQ):
        by_q = {q: P2['I'] for q in range(nq)}
        for q, ax in factors: by_q[q] = P2[ax]
        result = by_q[nq-1]
        for q in range(nq-2, -1, -1): result = np.kron(result, by_q[q])
        return result

    import epoq_classical_sampler as cs
    H_terms = cs.build_pauli_terms()
    H_dense = sum(c * pauli_term_dense(fac) for c, fac in H_terms)
    H_dense = (H_dense + H_dense.conj().T) / 2
    n0_dense = 0.5 * np.eye(DIM, dtype=complex) - 0.5 * pauli_term_dense([(4, 'Z')])

    # Physical-sector projector via z2_setup Gauss operators
    links_at = {(0,0):[0,2],(0,1):[0,3],(1,0):[1,2],(1,1):[1,3]}
    parity = {(0,0):+1,(0,1):-1,(1,0):-1,(1,1):+1}
    matter_q = {(0,0):4,(0,1):5,(1,0):6,(1,1):7}

    def G_dense(site):
        G = np.eye(DIM, dtype=complex)
        for ql in links_at[site]:
            G = G @ pauli_term_dense([(ql, 'X')])
        sign = pauli_term_dense([(matter_q[site], 'Z')])
        if parity[site] == -1: sign = -sign
        return G @ sign

    P_phys = np.eye(DIM, dtype=complex)
    for s in [(0,0),(0,1),(1,0),(1,1)]:
        P_phys = P_phys @ (np.eye(DIM, dtype=complex) + G_dense(s)) / 2
    P_phys = (P_phys + P_phys.conj().T) / 2

    evals_P, evecs_P = np.linalg.eigh(P_phys)
    phys_basis = evecs_P[:, evals_P > 0.5]
    H_phys = phys_basis.conj().T @ H_dense @ phys_basis
    H_phys = (H_phys + H_phys.conj().T) / 2
    n0_phys = phys_basis.conj().T @ n0_dense @ phys_basis
    n0_phys = (n0_phys + n0_phys.conj().T) / 2
    rho_phys = expm(-beta * H_phys); Z_phys = np.trace(rho_phys).real

    out = {}
    for t in times:
        Ut = expm(-1j * H_phys * t)
        n0_t = Ut.conj().T @ n0_phys @ Ut
        out[t] = np.trace(rho_phys @ n0_t @ n0_phys).real / Z_phys
    return out


# ----------------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------------
def run_action_epoq_pipeline(
    geom: LatticeGeometry, K: float, times: list[float],
    n_gauge: int = 100, n_corner_per_gauge: int = 20,
    n_warmup_gauge: int = 50, seed: int = 2026,
    use_pseudofermion: bool = False, n_pf: int = 1000,
    n_trotter_real: int = 200,
):
    """Run the full action-EρOQ pipeline. Returns observable estimates."""
    rng = np.random.default_rng(seed)

    # Phase 1: classical action MC for gauge configs
    print(f"Phase 1: classical action MC (n_gauge={n_gauge}, n_warmup={n_warmup_gauge}, K={K})...")
    t0 = time.time()
    mc_result = run_metropolis(
        geom, K=K, n_sweeps=n_gauge, n_warmup=n_warmup_gauge, seed=seed,
    )
    t_mc = time.time() - t0
    print(f"  wall-clock: {t_mc:.1f}s, accept rate: {mc_result.accept_rate:.3f}")

    # Phase 2: corner-state extraction per gauge config
    print(f"Phase 2: corner-state extraction "
          f"({'pseudofermion' if use_pseudofermion else 'direct'}, "
          f"n_corner={n_corner_per_gauge} per config)...")
    t0 = time.time()
    all_samples = []   # list of (U, psi_i, psi_j, sign_slater)
    for U in mc_result.configs:
        if use_pseudofermion:
            G_bdy = pseudofermion_G_bdy_estimate(geom, U, n_pf, rng)
            G_bdy = G_bdy.astype(complex)
        else:
            G_bdy = compute_G_bdy(geom, U)
        pairs, signs, _ = sample_corner_pairs_categorical(
            G_bdy, N_samples=n_corner_per_gauge, rng=rng)
        for k in range(n_corner_per_gauge):
            all_samples.append((U, int(pairs[k, 0]), int(pairs[k, 1]), int(signs[k])))
    t_corner = time.time() - t0
    print(f"  wall-clock: {t_corner:.1f}s, n_samples_total: {len(all_samples)}")

    # Phase 3: Minkowski matrix elements
    print(f"Phase 3: QC matrix elements (n_trotter={n_trotter_real})...")
    t0 = time.time()
    contribs = {t: [] for t in times}
    is_diag = []
    for k, (U, psi_i, psi_j, sign_slater) in enumerate(all_samples):
        if k % 100 == 0:
            print(f"  sample {k}/{len(all_samples)}...", flush=True)
        # diagonal in 8-qubit basis: same gauge config at both temporal slices
        # AND same fermion content
        bits_top = qc_basis_index(geom, U, 0, psi_i)
        bits_bot = qc_basis_index(geom, U, geom.N_E - 1, psi_j)
        is_diag.append(bits_top == bits_bot)
        n0_j = static_observable_diagonal(geom, U, psi_j)
        for t in times:
            mat_el = matrix_element_minkowski(
                geom, U, psi_i, psi_j, t=t, n_trotter=n_trotter_real)
            contribs[t].append(sign_slater * n0_j * mat_el.real)
    t_qc = time.time() - t0
    is_diag = np.array(is_diag, dtype=bool)
    print(f"  wall-clock: {t_qc:.1f}s")
    print(f"  diagonal sample fraction: {is_diag.mean():.3f}")
    print(f"  negative-sign fraction: "
          f"{np.mean([s for _,_,_,s in all_samples] == -1):.3f}", end=" ")
    sign_vals = np.array([s for _,_,_,s in all_samples])
    print(f"(actually: {np.mean(sign_vals < 0):.3f})")

    # Phase 4: ratio estimator with bootstrap
    print("Phase 4: ratio estimator + bootstrap...")
    out_C = {}
    out_err = {}
    for t in times:
        contribs_arr = np.array(contribs[t])
        # Ratio estimator: numerator / denominator
        # Denom: Σ_k sign_slater_k × δ(i, j) × n_0_i = #{diagonal samples with n_0_i = 1}
        # For each diagonal sample (bits_top == bits_bot), the "observable" reduces to n_0_i
        # since ⟨i|U_M†(0) n_0 U_M(0)|i⟩ = n_0_i.
        # Actually for general t, the contribution is sign × n_0_j × mat_el. At t=0,
        # mat_el is δ_ij × n_0_i, so contribution becomes sign × n_0_j × n_0_i × δ_ij.
        # The denominator we want is what corresponds to Z: sum of diagonal density-matrix.
        # For our sampling weighted by |ρ_ji|: Z_estimator = Σ samples (sign × δ_diag).
        denom = np.sum([sign_vals[k] if is_diag[k] else 0 for k in range(len(all_samples))])
        if abs(denom) < 1e-12:
            print(f"  WARNING: zero denominator at t={t}, skipping bootstrap")
            out_C[t], out_err[t] = float('nan'), float('nan')
            continue
        # Bootstrap
        N = len(contribs_arr)
        rng_b = np.random.default_rng(31337)
        means = []
        for _ in range(100):
            idx = rng_b.choice(N, N, replace=True)
            num = contribs_arr[idx].sum()
            den = np.sum([sign_vals[i] if is_diag[i] else 0 for i in idx])
            if abs(den) < 1e-12:
                continue
            means.append(num / den)
        out_C[t] = np.mean(means) if means else float('nan')
        out_err[t] = np.std(means) if means else float('nan')
    return {
        'C': out_C, 'err': out_err,
        'n_samples': len(all_samples), 'is_diag_fraction': is_diag.mean(),
        'walltime': {'mc': t_mc, 'corner': t_corner, 'qc': t_qc},
    }


def _self_test():
    geom = LatticeGeometry(Lx=2, Ly=2, N_E=4, m=0.5)
    K = 1.0
    times = [0.0, 0.5, 1.0, 2.0]

    # Reference from physical-sector ED
    # NOTE: β in the Euclidean lattice is N_E · a (here a=1, so β=4); whereas the
    # z2_setup reference used β=0.5. Match the comparison correctly: the
    # action-MC's "β_eff" is determined by the lattice extent N_E and a_τ.
    # For now, use the physical β = N_E to match the lattice setup; print both
    # references to cross-check.
    ref_at_Nt = physical_sector_reference(beta=geom.N_E, times=times)
    print(f"Reference C(t) at β = N_E = {geom.N_E}:")
    for t in times:
        print(f"  t={t}: {ref_at_Nt[t]:.6f}")
    print()

    # Run pipeline with DIRECT corner extraction
    print("=" * 60)
    print("Pipeline: DIRECT corner extraction")
    print("=" * 60)
    result_direct = run_action_epoq_pipeline(
        geom, K=K, times=times,
        n_gauge=100, n_corner_per_gauge=20,
        n_warmup_gauge=50, seed=2026,
        use_pseudofermion=False,
        n_trotter_real=200,
    )
    print(f"\nResults:")
    print(f"  {'t':>5} | {'C (action-MC direct)':>22} | {'reference':>12}")
    for t in times:
        rf = ref_at_Nt[t]
        m, e = result_direct['C'][t], result_direct['err'][t]
        z = (m - rf) / e if e > 0 else 0
        print(f"  {t:>5.2f} | {m:>9.6f} ± {e:.4f}    | {rf:>12.6f}  (z={z:+.2f})")


if __name__ == '__main__':
    _self_test()
