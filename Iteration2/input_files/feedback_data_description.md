The dataset description explains the observed limitations in the analysis, specifically regarding the N=5 and N=10 configurations.

1. **Constraint**: Limited sample size and low vortex density.
   - **Limitation**: The dataset contains only 5 tracers per configuration. For N=5 and N=10, the analysis results (e.g., Hill estimator instability, poorly constrained residence time exponents, and non-monotonic MSD scaling) are directly attributable to the small number of tracers and the near-integrable nature of these sparse vortex fields.
   - **Effect**: These constraints prevent the reliable estimation of heavy-tail statistics and anomalous exponents for low-vorticity regimes, as the tracer trajectories are dominated by individual vortex interactions rather than ensemble-level Lévy-stable statistics.

2. **Constraint**: Measurement noise floor (σ=0.02 m).
   - **Limitation**: The noise floor is significant relative to the small physical displacements in the N=5 and N=10 configurations.
   - **Effect**: This explains the "pathological" behavior of the Hill estimator and the inflated μ values reported in Section 3.1, confirming that the noise floor masks the physical signal in sparse configurations.

3. **Constraint**: Finite simulation time (24.75 s).
   - **Limitation**: The observation window is limited.
   - **Effect**: This explains the observed exponential cutoffs in residence time distributions for N=5 and N=10, as the finite-time horizon of the simulation truncates the potential power-law regime, limiting the validity of long-term superdiffusive conclusions for these configurations.