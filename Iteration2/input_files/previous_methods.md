1. **Data Preparation and Noise Mitigation**
   - Load the `point_vortex_tracers.npy` and `levy_walk_trajectories.npy` datasets.
   - Apply a Savitzky-Golay filter to the `x_noisy` and `y_noisy` fields to smooth high-frequency Gaussian noise.
   - Validate the filtering by comparing results against `x_true`/`y_true` to ensure the noise floor is suppressed without introducing bias.

2. **Displacement Distribution Analysis**
   - Compute displacement distributions $P(\Delta x, \Delta t)$ and $P(\Delta y, \Delta t)$ for a range of lag times $\Delta t$.
   - Construct the empirical characteristic function $\phi(k, \Delta t) = \langle \exp(ik \cdot \Delta x) \rangle$ by averaging over all tracers/trajectories for each configuration ($N \in \{5, 10, 20, 40\}$ and $\beta \in \{1.2, 1.5, 1.8, 2.5\}$).
   - Visualize the transition from Gaussian-like behavior at small $\Delta t$ to heavy-tailed Lévy-stable behavior at large $\Delta t$.

3. **Fractional Diffusion Modeling**
   - Fit the fractional model $\phi(k, \Delta t) = \exp(-D_\alpha |k|^\alpha \Delta t)$ to the empirical characteristic functions.
   - Extract the anomalous exponent $\alpha(\Delta t)$ and the generalized diffusion coefficient $D_\alpha(\Delta t)$ across the entire time range.

4. **Crossover Time Identification**
   - Define the physical crossover time $\tau_c$ as the time at which the empirical $\alpha(\Delta t)$ enters the 95% confidence interval of the theoretical $\alpha$ for the corresponding Lévy walk class, or where the derivative $|d\alpha/d(\Delta t)|$ falls below 5% of its initial value.
   - Use bootstrapping across tracer ensembles to quantify the uncertainty in $\tau_c$.

5. **Ergodicity and Sensitivity Analysis**
   - Calculate the ergodicity-breaking parameter $EB(\Delta t) = \text{Var}(\text{TAMSD}) / \text{Mean}(\text{TAMSD})^2$.
   - Perform a sensitivity check on the $N=20$ configuration using sub-sampling to determine if observed anomalies are robust or driven by small ensemble size.
   - Use $EB$ profiles to define the limits of the ensemble-averaged fractional diffusion model.

6. **Effective Theory Mapping**
   - Establish a mapping between vortex density $N$ and transport parameters $(\alpha, D_\alpha)$.
   - Compare the empirical $D_\alpha(N)$ and $\alpha(N)$ against the Lévy walk ground truth $D_\alpha(\beta)$ and $\alpha(\beta)$.
   - Identify the "Lévy-equivalence" by matching point-vortex asymptotic values to the corresponding Lévy walk class.

7. **Statistical Validation**
   - Perform a Kolmogorov-Smirnov test on the displacement distributions $P(\Delta x, \Delta t)$ at the identified crossover time $\tau_c$ to evaluate the goodness-of-fit of the fractional diffusion model.
   - Contrast the point-vortex results against the Lévy walk null hypothesis to confirm the transition from coherent trapping to chaotic mixing.