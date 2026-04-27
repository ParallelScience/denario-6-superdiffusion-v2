1. **Noise Sensitivity and Filtering Validation**
   - Quantify the impact of the $0.02$ m Gaussian noise on velocity increment tails by comparing the tail index $\mu$ of `x_true` vs `x_noisy`.
   - Perform a sensitivity analysis on the Savitzky-Golay filter window size; plot $\mu$ as a function of window size to identify the "plateau" region where noise is suppressed while physical heavy tails are preserved.
   - Verify that all subsequent mechanistic metrics (residence times, VACF) are computed for both `x_true` and `x_noisy` to ensure physical conclusions are not noise-driven artifacts.

2. **Mechanistic Residence Time Analysis**
   - Define a dynamic "trapped" state for tracers using a velocity threshold ($|v| < v_{threshold}$) or the Okubo-Weiss criterion to identify regions of closed streamlines, ensuring the definition accounts for the dynamic nature of the vortex field.
   - Calculate the distribution of residence times $P(\tau_{res})$ for each configuration ($N \in \{5, 10, 20, 40\}$).
   - Test for power-law scaling $P(\tau_{res}) \sim \tau_{res}^{-\gamma}$ and link any observed cutoffs to the finite-time horizon of vortex interactions.

3. **Velocity Autocorrelation and Memory Effects**
   - Compute the Velocity Autocorrelation Function (VACF) $C_v(\Delta t)$ for all tracers.
   - Compare the point-vortex VACF decay against the Lévy walk ground truth; identify non-zero, long-lived correlation structures in the point-vortex system as evidence of Hamiltonian memory.
   - Quantify memory using Mutual Information $I(v(t), v(t+\Delta t))$ to demonstrate deviations from the memoryless (Markovian) nature of the Lévy walk.

4. **Sliding-Window Non-Stationarity Analysis**
   - Implement a sliding-window approach to calculate the local anomalous exponent $\alpha(t)$ and the ergodicity-breaking parameter $EB(t)$.
   - Normalize the window size by the characteristic vortex-vortex interaction time to distinguish between system-wide stationarity and local vortex-tracer interaction effects.
   - Compare $EB(t)$ profiles between datasets to visualize the transition from ergodic to non-ergodic regimes.

5. **Lévy Walk as a Null Model**
   - Use the `levy_walk_trajectories.npy` as a null model for "pure" Lévy-stable transport.
   - Quantify the divergence between the point-vortex system and the Lévy walk by comparing their respective VACF decay rates and $EB(t)$ profiles.
   - Characterize the specific time scales and vortex densities where point-vortex dynamics deviate from the Lévy walk, attributing these to Hamiltonian persistence.

6. **Phase-Space Mapping and Breakdown Analysis**
   - Construct a state-space map of $(\alpha, \mu)$ for both datasets.
   - Quantify the "Lévy-equivalence" using the Kullback-Leibler (KL) divergence between the velocity increment PDFs of the point-vortex tracers and the best-fit Lévy-stable distributions.
   - Map the breakdown of this equivalence as a function of $N$ and $t$, documenting the physical conditions where Hamiltonian "stickiness" dominates over Lévy-stable flight statistics.