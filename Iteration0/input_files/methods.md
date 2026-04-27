1. **Data Preprocessing and Sensitivity Analysis**
   - Load the `point_vortex_tracers.npy` and `levy_walk_trajectories.npy` datasets.
   - Apply a Savitzky-Golay filter to the noisy coordinates. Perform a sensitivity analysis on the filter window size by comparing the resulting velocity increment PDFs against those derived from the `x_true`/`y_true` fields to ensure heavy-tail statistics are preserved and not artificially truncated.
   - Verify the integrity of the `msd_true` field and ensure tracer trajectories remain within the effective vortex cluster radius to avoid boundary-induced bias.

2. **TAMSD and Local Scaling Analysis**
   - Calculate the Time-Averaged Mean Squared Displacement (TAMSD) for each tracer and walker.
   - Compute the logarithmic derivative of the TAMSD, $d(\log \text{MSD})/d(\log t)$, to identify the time-dependent evolution of the anomalous exponent $\alpha(t)$.
   - Define the "crossover time" as the transition point from short-time ballistic/trapping regimes to the asymptotic superdiffusive regime, using a sliding-window log-log regression to extract the stable $\alpha_{emp}$ for each configuration.

3. **Velocity Increment PDF and Tail Estimation**
   - Compute velocity increments $\Delta x$ and $\Delta y$ for both datasets.
   - Implement a Hill Plot or similar tail-index estimation technique to determine the optimal threshold for the power-law tail, ensuring the MLE fit for the stability index $\mu$ is not biased by the Gaussian core.
   - Compare empirical $\mu$ values from point-vortex data against the theoretical $\mu = \beta + 1$ from the Lévy walk ground truth.

4. **Fractional Diffusion Modeling via Characteristic Functions**
   - Instead of solving the fractional PDE in real space, compute the empirical characteristic function $\phi(k, t) = \langle \exp(ik \cdot \Delta x) \rangle$ for the tracer distributions.
   - Fit the model $\phi(k, t) = \exp(-D_\alpha |k|^\alpha t)$ to the data to extract the fractional order $\alpha$ and generalized diffusion coefficient $D_\alpha$.
   - Use the Kolmogorov-Smirnov test to evaluate the goodness-of-fit.

5. **Ergodicity and Statistical Significance**
   - Compute the ergodicity-breaking parameter $EB = \text{Var}(\text{TAMSD}) / \text{Mean}(\text{TAMSD})^2$ as a function of lag time.
   - Use bootstrapping to calculate confidence intervals for $EB$, $\alpha_{emp}$, and $\mu$ to account for the limited sample size (20 tracers per configuration).
   - Compare $EB$ profiles at long lag times to distinguish between transient non-ergodicity and true Lévy-like behavior.

6. **Effective Theory Mapping**
   - Construct a mapping function between vortex density $N$ and Lévy walk parameters $(\alpha, \beta)$ using a distance metric (e.g., Kullback-Leibler divergence) between the empirical velocity increment PDFs of the point-vortex tracers and the Lévy walk ground truth.
   - Quantify the mapping error using a Bayesian approach to establish confidence intervals for the predicted vortex density $N$ given an observed tracer trajectory.

7. **Validation of Chaoticity and Finite-N Corrections**
   - Incorporate finite $N$ and circulation variance $\gamma_{std}$ into the analytical $1/r$ kernel model to derive corrected theoretical predictions for $\alpha$.
   - Calculate the divergence rate of nearby tracers as a proxy for the Lyapunov exponent to quantify the physical "chaoticity" of each vortex configuration.
   - Correlate the empirical $\alpha(N)$ with the proxy Lyapunov exponent to validate the link between Hamiltonian chaoticity and the observed transport regime.