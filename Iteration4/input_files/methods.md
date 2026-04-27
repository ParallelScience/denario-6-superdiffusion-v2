# Methods — Iteration 4: Comprehensive Analysis with Full Figures

The goal of this iteration is to produce a publication-quality set of figures and a comprehensive narrative results report. Every analysis step must save high-resolution PNG figures and print numerical results. The researcher must write a full, self-contained results section referencing all figures.

## 1. Data Loading and Preprocessing

- Load `point_vortex_tracers.npy` and `levy_walk_trajectories.npy`.
- Use `x_true`/`y_true` for all mechanistic analyses.
- Apply Savitzky-Golay filter (window=7) to `x_noisy`/`y_noisy` for N=20,40 only.

## 2. Figure 1 — Tracer Trajectories (Overview)

Produce a 2×4 panel figure showing representative 2D tracer trajectories (x vs y) for all 4 vortex density configurations (N=5,10,20,40) in the top row, and 4 representative Lévy walk trajectories (β=1.2,1.5,1.8,2.5) in the bottom row. Color each trajectory by time (colormap = viridis). Add a title per panel showing N or β. Save as `fig1_trajectories.png`.

## 3. Figure 2 — Ensemble-Averaged MSD Scaling

Compute the ensemble-averaged MSD for each N configuration and each Lévy walk β class. Plot log-log MSD vs lag time for:
- Left panel: all 4 vortex configurations (N=5,10,20,40), each a different color, with fitted power-law slope annotated (α_emp).
- Right panel: all 4 Lévy walk classes (β=1.2,1.5,1.8,2.5), with theoretical slope α=3-β as a dashed reference line.
Add reference lines for α=1 (normal diffusion), α=2 (ballistic). Save as `fig2_msd_scaling.png`. Print α_emp per configuration.

## 4. Figure 3 — Velocity Autocorrelation Functions

Compute the normalized VACF C_v(Δt) = <v(t)·v(t+Δt)> / <v²> for each N configuration and each Lévy walk class. Plot:
- Left panel: VACF vs lag time for all point-vortex configurations.
- Right panel: VACF vs lag time for all Lévy walk classes.
Use log-linear scale (log x-axis). Annotate zero-crossing times. Add a horizontal dashed line at C_v=0. Save as `fig3_vacf.png`. Print zero-crossing lag per configuration.

## 5. Figure 4 — Residence Time Distributions

Define "trapped" state as instantaneous speed < 25th percentile of pooled speed per N group. Compute residence time distributions P(τ_res) for N=10,20,40 (skip N=5 — too few events). Plot:
- One panel with all 3 N configurations on log-log axes.
- Overlay fitted power-law tails P(τ) ~ τ^{-γ} with γ annotated per configuration.
- Mark the exponential cutoff (if present) for N=10 with a vertical dashed line.
Save as `fig4_residence_times.png`. Print γ ± uncertainty per configuration.

## 6. Figure 5 — Displacement PDFs at Multiple Lag Times

For the point-vortex N=40 configuration and the Lévy walk β=1.5 class, compute the displacement distribution P(Δx) at 4 lag times: Δt = 1, 5, 20, 50 steps. Plot:
- Left panel: N=40 point-vortex displacement PDFs (4 lag times, log-log).
- Right panel: β=1.5 Lévy walk displacement PDFs (4 lag times, log-log).
Overlay a Gaussian reference (dashed) and a Lévy-stable fit (dotted) at the largest lag time. Save as `fig5_displacement_pdfs.png`.

## 7. Figure 6 — Ergodicity-Breaking Profiles

Compute EB(Δt) = Var(TAMSD(Δt)) / Mean(TAMSD(Δt))² as a function of lag time for all N configurations and all Lévy walk β classes. Plot:
- Left panel: EB(Δt) vs Δt for point-vortex (4 N values).
- Right panel: EB(Δt) vs Δt for Lévy walks (4 β values).
Use log-log axes. Print EB plateau value (mean over last 20% of lags) per configuration.

## 8. Figure 7 — Regime Map (Summary Figure)

Construct a summary "regime map" scatter plot with:
- x-axis: EB plateau value
- y-axis: empirical α_emp (from MSD fit)
- Each point = one configuration; point-vortex as circles (color = N), Lévy walks as squares (color = β/α_theory).
- Annotate each point with its label (N=5, N=10, etc. / β=1.2, etc.).
- Add reference lines: α=1 (horizontal dashed), α=2 (horizontal dashed), EB=0 (vertical dotted).
Save as `fig7_regime_map.png`.

## 9. Comprehensive Results Report

The researcher must write a comprehensive results section (minimum 1500 words) covering:
1. Overview of tracer trajectory morphology (Fig 1)
2. MSD scaling analysis: ballistic vs superdiffusive regimes, the N=40 transition (Fig 2)
3. VACF and Hamiltonian memory: comparison with Lévy walk (Fig 3)
4. Residence time power-law tails: mechanistic link to Lévy-like trapping (Fig 4)
5. Evolution of displacement PDFs from short to long lags (Fig 5)
6. Ergodicity breaking: quenched heterogeneity vs intrinsic Lévy non-ergodicity (Fig 6)
7. Regime map synthesis: where vortex systems sit relative to Lévy walk ground truth (Fig 7)
8. Physical interpretation and effective theory implications

All figures must be saved to the `data/` directory and referenced by filename in the report.
