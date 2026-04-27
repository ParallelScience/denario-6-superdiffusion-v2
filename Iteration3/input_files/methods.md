1. **Spatial Decomposition and Regime Classification**
   - Calculate the instantaneous distance of each tracer to the nearest point vortex for all $N \in \{5, 10, 20, 40\}$.
   - Implement a time-dependent sliding-window Okubo-Weiss criterion to classify tracers into "trapped" (stable orbits) and "chaotic" (diffusive) populations, accounting for tracers that transition between regimes over time.
   - Compute the fraction of trapped tracers as a function of $N$ to quantify the reduction of stable regions as the vortex field becomes more chaotic.

2. **Generalized Langevin Equation (GLE) Framework**
   - Implement a GLE model: $m \frac{dv}{dt} = -\int_0^t \gamma(t-s) v(s) ds + \eta(t)$, where the memory kernel $\gamma(t)$ is related to the random force autocorrelation via the Fluctuation-Dissipation Theorem.
   - Estimate the memory kernel decay exponent $\beta$ from the Velocity Autocorrelation Function (VACF) decay $C_v(t) \sim t^{-\beta}$.
   - Validate the model by checking the consistency between the VACF decay exponent and the MSD scaling exponent $\alpha$, specifically testing the relation $\alpha = 2 - \beta$ for the superdiffusive regime.

3. **Ergodicity-Breaking (EB) Analysis**
   - Compute the EB parameter $EB(\Delta t) = \text{Var}(\text{TAMSD}) / \langle \text{TAMSD} \rangle^2$ separately for the "trapped" and "chaotic" populations.
   - Compare these conditioned EB profiles against the global EB profile of the Lévy walk dataset to isolate whether non-ergodicity arises from spatial heterogeneity (trapping) or intrinsic Hamiltonian stochasticity.

4. **Lyapunov-Based Interaction Time Scaling**
   - Calculate the global Lyapunov exponent $\lambda$ of the vortex-vortex system (not the tracers) to define the characteristic interaction time $\tau_{int} \approx 1/\lambda$.
   - For the "chaotic" population, compute local Lyapunov exponents or local interaction times based on the distance to the nearest vortex to provide a granular explanation for variations in tracer transport.
   - Normalize the time axis of the MSD and VACF plots by $\tau_{int}$ to test for universal scaling laws in the chaotic regime, excluding the "trapped" population to prevent skewing.

5. **Velocity Increment PDF and Tail Analysis**
   - Compute velocity increment PDFs at varying lag times $\Delta t$ for both datasets.
   - Fit the tails of the point-vortex velocity increments to a Lévy-stable distribution to extract the stability index $\mu$.
   - Compare the evolution of $\mu(\Delta t)$ against the Lévy walk ground truth to identify time scales where Hamiltonian memory effects cause deviations from memoryless Lévy-stable behavior.

6. **Quantifying Hamiltonian Memory vs. Lévy-Stable Statistics**
   - Calculate the Mutual Information $I(v(t), v(t+\Delta t))$ to quantify velocity correlation persistence.
   - Use KL divergence to measure the distance between the point-vortex velocity PDF and the best-fit Lévy-stable distribution.
   - Map the "Memory-Lévy" plane (KL divergence vs. Mutual Information) to visualize the transition from memory-dominated (trapped) to Lévy-dominated (superdiffusive) regimes, interpreting the divergence as a measure of Hamiltonian "stickiness."

7. **Validation of Anomalous Exponents**
   - Perform a Bayesian estimation of the anomalous exponent $\alpha$ for point-vortex tracers.
   - Use the Lévy walk dataset as a control for the $0.02$ m noise floor to calibrate the estimator's sensitivity.
   - Report $\alpha(N)$ with credible intervals, ensuring the distinction between ballistic and superdiffusive regimes is statistically robust and physically grounded in the Hamiltonian dynamics.