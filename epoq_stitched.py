"""
EρOQ-faithful stitched pipeline: classical Euclidean sampler +
Trotter-circuit Minkowski simulator → thermal correlator C(t).

Combines:
  - epoq_classical_sampler.py  (samples (i,j) corner pairs from |⟨j|e^{-βH}|i⟩|
                                via imaginary-time Trotter on the Pauli-term H)
  - epoq_quantum_simulator.py  (computes ⟨j|U†(t) O U(t)|i⟩ via real-time Trotter
                                on the same Pauli-term H)

Neither piece uses dense matrix exponentiation, eigh, or any np.kron-expanded
dense Hamiltonian — both work entirely with Pauli-term lists and state vectors.

Output:
  C(t) estimates with bootstrap error bars
  Compared against an ED reference (which is allowed to use kron+expm —
  for validation only, not part of the production pipeline).
"""

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, '/home/hlamm/Desktop/QC/logdet/m1_toy')
import epoq_classical_sampler as cs
import epoq_quantum_simulator as qs

# Parameters
beta = 0.5
n_trotter_imag = 50
n_trotter_real = 200    # ~5% Trotter error at t=1 per agent 2 timing study
N_samples = 200
N_bootstrap = 50
times = [0.0, 0.5, 1.0, 2.0]

# Observable: n_0 = (I - Z_4) / 2 = projection onto matter-site (0,0) occupied
n0_obs = [(0.5 + 0j, []), (-0.5 + 0j, [(4, 'Z')])]

# --- Get Pauli term list from sampler (both modules build the same H) -----
H_terms_cs = cs.build_pauli_terms()
H_terms_qs = qs.build_pauli_terms()

# Brief consistency check on the two H's
print(f"Sampler H: {len(H_terms_cs)} Pauli terms")
print(f"Simulator H: {len(H_terms_qs)} Pauli terms")
print("(They may differ by identity-shift terms that cancel in U(t) and ρ.)")

# --- (1) Classical-side sampling ------------------------------------------
print(f"\nSampling {N_samples} corner pairs (β={beta}, n_trotter_imag={n_trotter_imag})...")
rng = np.random.default_rng(2026)
out = cs.sample_corner_pairs(beta, N_samples, rng, n_trotter=n_trotter_imag)
pairs = out['pairs']
signs = out['signs']
total_weight = out['total_weight']
G = out['G']
Z_beta = G.diagonal().real.sum()
print(f"  Z(β) (from sampler propagator diagonal) = {Z_beta:.6f}")
print(f"  Total |G| weight = {total_weight:.6f}")
print(f"  Negative-sign fraction in samples: {np.mean(signs < 0):.4f}")
print(f"  Unique i values in samples: {len(set(pairs[:, 0]))}/{N_samples}")

# --- (2) Quantum-side matrix element per sample --------------------------
print(f"\nEvaluating Trotter-circuit matrix elements (n_trotter_real={n_trotter_real})...")
# Per-sample contributions to C(t) at each time
# Estimator: C(t) ≈ (total_weight / Z_beta / N) × Σ_k sign_k × O_j_k × ⟨i_k|U†OU|j_k⟩
# (See derivation in module docstring.)

# For O = n_0 diagonal, O|j⟩ = O_j |j⟩ where O_j = (j>>4)&1
# So we only need to evaluate ⟨i|U†OU|j⟩ for samples with O_j != 0.

contributions = {t: np.zeros(N_samples) for t in times}

for k in range(N_samples):
    if k % 50 == 0:
        print(f"  sample {k}/{N_samples}...")
    i = int(pairs[k, 0])
    j = int(pairs[k, 1])
    Oj = (j >> 4) & 1
    if Oj == 0:
        continue   # contribution is 0 if O annihilates |j⟩
    for t in times:
        # qs.observable_matrix_element(i_bit, j_bit, ...) returns ⟨j|U†OU|i⟩
        # We want ⟨i|U†OU|j⟩; by symmetry of Hermitian O and unitary U, this is conjugate:
        # ⟨i|U†OU|j⟩ = ⟨j|U†OU|i⟩*  ← only true if O is Hermitian, which n_0 is
        # We just take the real part anyway for C(t), so we can call either order.
        mat_el = qs.observable_matrix_element(j, i, n0_obs, t, n_trotter=n_trotter_real)
        contributions[t][k] = signs[k] * Oj * mat_el.real

# --- (3) Aggregate with bootstrap error bars ------------------------------
print(f"\nAggregating with N_bootstrap={N_bootstrap} replicates...")

results = {}
for t in times:
    arr = contributions[t]
    boot_means = []
    rng_b = np.random.default_rng(31337)
    for b in range(N_bootstrap):
        idx = rng_b.choice(N_samples, N_samples, replace=True)
        m = arr[idx].mean() * total_weight / Z_beta
        boot_means.append(m)
    results[t] = (np.mean(boot_means), np.std(boot_means))

# --- (4) ED reference (validation only) -----------------------------------
# Build dense H from Pauli term list — for validation reference only.
print("\nBuilding ED reference (validation only, allowed to use kron+expm)...")
P_dense = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}
NQ = 8

def pauli_term_dense(factors):
    """Build dense 256×256 matrix for a Pauli term given as [(qubit, axis), ...]."""
    pauli_at = {q: P_dense['I'] for q in range(NQ)}
    for q, ax in factors:
        pauli_at[q] = P_dense[ax]
    # kron in decreasing qubit order so qubit q corresponds to bit (i >> q) & 1
    result = pauli_at[NQ - 1]
    for q in range(NQ - 2, -1, -1):
        result = np.kron(result, pauli_at[q])
    return result

H_dense = np.zeros((256, 256), dtype=complex)
for coef, factors in H_terms_cs:
    H_dense += coef * pauli_term_dense(factors)

# Sanity: H_dense should be Hermitian
assert np.allclose(H_dense, H_dense.conj().T), "Built H is not Hermitian!"

rho_ED = expm(-beta * H_dense)
Z_ED = np.trace(rho_ED).real

# n_0 as dense operator
n0_dense = pauli_term_dense([]) * 0.5 + pauli_term_dense([(4, 'Z')]) * (-0.5)

def C_ED(t):
    Ut = expm(-1j * H_dense * t)
    n0_t = Ut.conj().T @ n0_dense @ Ut
    return np.trace(rho_ED @ n0_t @ n0_dense).real / Z_ED

# --- (5) Report -----------------------------------------------------------
print("\n" + "=" * 78)
print("RESULTS: EρOQ-faithful C(t) vs ED reference")
print("=" * 78)
print(f"  {'t':>5} | {'EρOQ mean':>12} | {'EρOQ std':>10} | {'ED reference':>14} | {'(EρOQ - ED) / std':>20}")
print("  ------+--------------+------------+----------------+---------------------")
for t in times:
    m, s = results[t]
    ed = C_ED(t)
    z = (m - ed) / s if s > 0 else 0
    print(f"  {t:>5.2f} | {m:>12.6f} | {s:>10.4f} | {ed:>14.6f} | {z:>+20.3f}")

print()
print("Pipeline architecture:")
print("  - Classical Euclidean: imaginary-time Trotter on Pauli H, n_trotter_imag={}"
      .format(n_trotter_imag))
print("  - Quantum Minkowski: real-time Trotter on Pauli H, n_trotter_real={}"
      .format(n_trotter_real))
print("  - Neither uses expm or eigh; H never materialized as a dense matrix")
print("    during the production EρOQ pipeline (only in the ED reference for")
print("    validation).")
print()
print("If |EρOQ - ED| / std is within ~±2, the pipeline is consistent within")
print("the combined Trotter + stochastic uncertainty.")
