# Full residual theory audit

The retained mathematical reference is the complete B.10 identity and its source-dependent mild representation. No general Hamiltonian is asserted to reduce to a homogeneous heat equation.

Removed the frozen coefficient solvers, their JAX/PyTorch/NumPy wrappers, associated examples, statistical scripts, safety tests, plotting scripts, cached example results, and local configuration. Those files did not integrate the complete residual. Their path inventory is recorded separately; scientific source-omission negative controls in the full residual tests remain.

Removed unsupported README claims of population certification, dimension-independent performance, and a numerical viscosity bound with unit constant. A relative iterate change is not an equation defect. A finite-query sampling error does not imply uniform sign certification.

The multidimensional diagnostic residuals remain too large for validated full residual claims. Re-rendering cached fields does not recompute the solution. Full nonlinear stochastic source reconstruction, multidimensional refinement, obstacle constraints, and uniform certification remain unimplemented.

MAPF's sampler-dependent precomputation is disabled explicitly because its solver has been removed. The independent MAPF grid precomputation and existing cache consumers remain separate from the full residual implementation.
