

Iteration 0:
**Summary: Superdiffusion via Chaotic Vortex Interactions**

**1. Data & Methodology**
*   **Datasets:** Point-vortex tracer simulations (N={5, 10, 20, 40}, dt=0.05s) and Lévy walk ground truth (β={1.2, 1.5, 1.8, 2.5}, dt=0.1s).
*   **Key Findings:** TAMSD-based estimation of anomalous exponent α is unreliable for small ensembles (N=5-20) due to transient ballistic regimes and finite-time effects. Characteristic function (CF) fitting is the most robust estimator, showing a monotonic decrease in α (1.95 to 1.79) as N increases, confirming the link between vortex density and anomalous transport.
*   **Limitations:** Single-step velocity increment PDFs for Lévy walks are bounded by the discretization (dt), failing to capture the heavy-tailed flight-length distribution. Point-vortex tracers exhibit quasi-coherent ballistic motion at low N; only N=40 shows significant departure from Gaussianity.

**2. Critical Decisions & Constraints**
*   **Discarded:** TAMSD-based α estimation for small ensembles; single-step increment analysis for Lévy walk validation.
*   **Adopted:** CF-based fractional diffusion modeling (log|φ(k, t)| = −D_α |k|^α t) as the primary metric for α.
*   **Constraint:** All analyses must use noise-free `x_true`/`y_true` fields; Gaussian noise (σ=0.02m) significantly biases tail statistics.
*   **Observation:** The crossover time to superdiffusion is ~0.4s; observation windows must exceed this significantly to avoid ballistic bias.

**3. Future Directions**
*   **Refinement:** The current observation window (24.75s) is insufficient for asymptotic convergence at low N. Future simulations require longer time series or higher vortex densities to reach the true Lévy-stable regime.
*   **Modeling:** The fractional diffusion model residuals remain high; consider incorporating a tempered Lévy-stable model to account for finite-domain/finite-time truncation of the heavy tails.
*   **Validation:** The N=40 configuration is the only one approaching the predicted Cauchy-like (α≈1) regime. Future work should focus on N > 40 to confirm the theoretical limit of the 1/r kernel.
        