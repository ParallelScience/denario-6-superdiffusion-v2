1. **Data Preprocessing and Noise Assessment**
   - Utilize the `x_true` and `y_true` fields for primary physical calculations to avoid smoothing artifacts.
   - Apply a Savitzky-Golay filter to the `x_noisy` and `y_noisy` fields to create a denoised proxy for validation and sensitivity analysis.
   - Quantify the impact of the 0.02 m noise floor by comparing the excess kurtosis of velocity increments between `true` and `noisy` datasets to ensure heavy-tailed observations are physical.

2. **Spatial Heterogeneity and Environment Classification**
   - Compute the Okubo-Weiss criterion $Q = s^2 - \omega^2$ using the true vortex positions and circulations. Use a local interpolation method or kernel-density estimation to compute velocity gradients, avoiding singularities at vortex centers.
   - Classify tracer environments as "trapped" ($Q < 0$) or "chaotic" ($Q > 0$).
   - Calculate the fraction of time each tracer spends in trapped regions as a function of $N$ to quantify spatial heterogeneity.

3. **Crossover Time ($t_c$) and MSD Scaling**
   - Compute the Time-Averaged Mean Squared Displacement (TAMSD) for each tracer.
   - Calculate the local exponent $\alpha(t) = d(\log \text{TAMSD}) / d(\log t)$ using a rolling window wide enough to suppress noise.
   - Define $t_c$ as the time where the second derivative of the log-log MSD reaches a local maximum, marking the onset of the transition from ballistic ($\alpha \approx 2$) to diffusive ($\alpha \approx 1$) regimes.
   - Report $t_c$ with bootstrap confidence intervals to account for tracer variance.

4. **Velocity Increment PDF and Memory Decay**
   - Compute velocity increments $\Delta v$ for lag times $\Delta t$ spanning from the sub-ballistic interaction time to the diffusive regime.
   - Analyze the excess kurtosis of these increments to identify the "memory loss" timescale.
   - Compare the kurtosis decay rate of point-vortex tracers against the Lévy walk ground truth to determine if the vortex-driven dynamics converge toward the Lévy-stable limit.

5. **Ergodicity-Breaking (EB) Quantification**
   - Calculate the EB parameter $EB = \text{Var}(\text{TAMSD}) / \langle \text{TAMSD} \rangle^2$ for each $N$ configuration.
   - Correlate the EB scalar with the fraction of trapped tracers to determine if non-ergodicity is driven by quenched spatial heterogeneity.
   - Provide error bars for EB values using the 20 tracers available per configuration.

6. **Regime Mapping and Theoretical Synthesis**
   - Construct a "Regime Map" plotting $t_c$ versus the EB scalar, using $N$ as a color-coded variable to visualize the transition trajectory.
   - Overlay the Lévy walk ground truth as a reference set of points on the same plane to identify where point-vortex dynamics mimic or deviate from Lévy-stable behavior.
   - Synthesize the results to define the predictive relationship between vortex density $N$ and the observed transport regime, characterizing the limits of the effective theory.