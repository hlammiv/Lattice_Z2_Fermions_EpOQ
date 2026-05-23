# Lattice Z₂ Fermions EρOQ

A 2+1d Z₂ gauge + staggered Kogut–Susskind fermion toy implementation of the
EρOQ (Euclidean ρ Observable on Quantum) pipeline. Built to demonstrate the
Lagrangian path-integral classical leg → Minkowski-evolution QC leg pipeline
matches direct ED of the lattice Hamiltonian for a 2×2 OBC spatial lattice.

## What it computes

For the 2+1d Hamiltonian
```
H_QC = −g_E Σ_l X_l  −  g_M Π_{plaq} Z  +  m Σ_site (−1)^(x+y) n_site
       +  g_hop Σ_l η_l (ψ̄_a Z_l ψ_b + h.c.)
```
on a 2×2 OBC spatial lattice (4 link qubits + 4 matter qubits, 8 qubits total),
the pipeline evaluates the thermal Minkowski 2-point correlator
```
C(t) = ⟨n_0(t)·n_0(0)⟩_β  =  Tr[ρ̃·U(t)†·n_0·U(t)·n_0] / Tr[ρ̃]
```
where ρ̃ is built by Monte Carlo sampling of Z₂ gauge configurations and
applying per-config fermion transfer matrices.

## How the construction works

The path-integral identity (see `memory/project_epoq_observable_correctness_derivation.md`):
```
⟨g_a, ψ_a | e^{−βH_QC} | g_b, ψ_b⟩  =  ∫ DU(τ) e^{−S_g[U]} · W_U[ψ_a, ψ_b]
```
with
- `S_g[U]`: lattice Z₂ Wilson action with K_E (temporal plaquettes) and K_M (spatial plaquettes), Suzuki-mapped from g_E and g_M.
- `W_U[ψ_a, ψ_b] = ⟨ψ_a| T_F[g(N−1)]·…·T_F[g(0)] |ψ_b⟩`: fermion transfer matrix product, with `T_F[g(t)] = expm(−a_τ · H_F[g(t)])` evaluated EXACTLY (no internal Trotter splitting) at the gauge eigenstate of each slice.

**Important**: no `det M` in the gauge measure. Including it double-counts the fermion sector — see `memory/feedback_no_det_M_with_corner_states.md`.

## Status

Working (verified at β=0.5 with deterministic enumeration to 10⁻⁴ on C(0)):
- gauge Monte Carlo sampling (parallel-tempering, sign-problem-free with `action_type='gauge_only'`)
- per-config fermion W via exact `expm(−a_τ H_F)`
- palindrome ordering across time slices (Hermitian-PD per-config W)
- staggered KS phases throughout (action, T_F, H_QC)
- 8-qubit basis assembly consistent with `epoq_classical_sampler` H_QC

Known finite-a_τ systematic at β=4 (Phase 31): the Trotter splitting
`e^{−a_τH} ≈ e^{−a_τH_g}·e^{−a_τH_F[U_op]}` introduces an `O(β·a_τ)` error
from `[H_E, H_F]` (gauge X_l flips don't commute with the gauge-dependent
fermion hopping). Confirmed by pure-gauge cross-check (`check_pure_gauge.py`,
Phase 34): the gauge sector alone shows `O(a_τ²)` Trotter error from
`[H_E, H_M]`. Strang splitting (Phase 35) is the planned fix.

## Key entry points

| File | Role |
|---|---|
| `epoq_classical_sampler.py` | H_QC Pauli-string builder and Trotter routines |
| `epoq_quantum_simulator.py` | Minkowski U(t) and observable matrix elements |
| `action_z2_staggered.py` | Lattice geometry, gauge config dataclass, S_g action |
| `action_z2_kernels.py` | Numba-jitted hot loops (det M, S_g) |
| `action_z2_metropolis.py` | Single-link Metropolis MC; `action_type='gauge_only'` is the correct setting |
| `action_z2_metropolis_pt.py` | Parallel-tempering wrapper |
| `transfer_matrix_kbc_trotterized.py` | T_F = expm(−a_τ H_KS) per slice; palindrome W |
| `action_corner_direct_v3.py` | Corner-state (ψ_a, ψ_b) Fock-basis utilities |
| `action_minkowski_stitch.py` | (gauge, ψ) → 8-qubit basis assembly |

## Reproducing key results

```bash
# β=0.5, a_τ=0.5 deterministic test (exact path-integral enumeration, no MC)
python check_rho_deterministic.py

# β=0.5 MC validation against deterministic, gauge_only MC
python validate_gauge_only_mc.py

# β=4 a_τ scan with gauge_only MC
python check_beta4_atau_scan.py

# Pure-gauge cross-check (g_hop=m=0) — isolates gauge-sector Trotter error
python check_pure_gauge.py
```

## Open issues

- **Phase 31**: gauge-fermion `[H_E, H_F]` Trotter error at β=4 (this is the residual systematic)
- **Phase 35**: implement Strang splitting between H_g and H_F (the fix)

## References

- arXiv:2011.11677 (Gustafson & Lamm) — original 2+1d Z₂ EρOQ paper
- arXiv:2307.14508 — EρOQ with DMQMC fermion treatment
- arXiv:2407.13080 (Fleming, Shyamsundar, Unmuth-Yockey) — logdet anchor reference

## License & authorship

Internal research code for the EρOQ methodology paper; not yet finalized.
