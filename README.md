# denario-6-superdiffusion-v2

**Scientist:** denario-6
**Date:** 2026-04-27

## Superdiffusion via Chaotic Vortex Interactions — Dataset

This project investigates the effective theory of superdiffusion arising from chaotic interactions between point vortices in 2D. The goal is to derive Lévy flight statistics analytically from the Kirchhoff-Onsager Hamiltonian and validate against two complementary datasets: (1) direct point-vortex simulations with passive tracer trajectories, and (2) synthetic Lévy walk trajectories with known anomalous exponents.

---

### File 1: Point-Vortex Tracer Simulations

**Path:** `/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy`

NumPy structured array, 10,000 rows (20 tracers × 500 time steps).

| Field          | dtype    | Description                                                              |
|----------------|----------|--------------------------------------------------------------------------|
| trajectory_id  | int32    | Tracer ID 1–20                                                           |
| n_vortices     | int32    | Number of point vortices (5, 10, 20, or 40) — 5 tracers per config      |
| gamma_std      | float64  | Standard deviation of vortex circulations Γ ~ N(0, γ_std²), γ_std=1.0  |
| time           | float64  | Observation time t = step × dt, dt=0.05 s, range [0, 24.75] s           |
| x_noisy        | float64  | Tracer x-position with Gaussian noise σ=0.02 m                          |
| y_noisy        | float64  | Tracer y-position with Gaussian noise σ=0.02 m                          |
| x_true         | float64  | Noise-free tracer x-position                                             |
| y_true         | float64  | Noise-free tracer y-position                                             |
| msd_true       | float64  | True squared displacement from origin (x_true²+y_true²)                 |

**Physics model:** N point vortices with positions (x_i, y_i) and circulations Γ_i evolve under the Kirchhoff-Onsager Hamiltonian:
  H = -1/(4π) Σ_{i≠j} Γ_i Γ_j ln(r_ij)
Equations of motion (RK4 integration):
  dx_i/dt =  (1/2π) Σ_{j≠i} Γ_j (y_i - y_j) / r_ij²
  dy_i/dt = -(1/2π) Σ_{j≠i} Γ_j (x_i - x_j) / r_ij²
A passive tracer is advected by the vortex velocity field (also RK4). Vortex positions are evolved with forward Euler. Initial vortex positions are uniform random in [-10, 10]². Tracer starts near origin.

**Vortex configurations (5 tracers each):**
| n_vortices | Expected regime               |
|------------|-------------------------------|
| 5          | Weakly chaotic                |
| 10         | Moderately chaotic            |
| 20         | Chaotic, onset of Lévy-like   |
| 40         | Strongly chaotic              |

**Key question:** Does the tracer MSD scale anomalously (α > 1) and does α increase with n_vortices? Do the velocity increment PDFs develop heavy tails consistent with a Lévy-stable distribution as predicted by the 1/r kernel?

---

### File 2: Lévy Walk Trajectories (Ground Truth)

**Path:** `/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy`

NumPy structured array, 24,000 rows (40 trajectories × 600 time steps).

| Field          | dtype    | Description                                                              |
|----------------|----------|--------------------------------------------------------------------------|
| trajectory_id  | int32    | Trajectory ID 1–40                                                       |
| beta           | float64  | Lévy walk tail index β (flight-time distribution P(τ) ~ τ^{-(1+β)})     |
| alpha_theory   | float64  | Theoretical anomalous exponent α = 3 − β (valid for 1 < β < 2)          |
| time           | float64  | Observation time (dt=0.1 s, range [0, 59.9] s)                          |
| x_noisy        | float64  | x-position with Gaussian noise σ=0.05 m                                 |
| y_noisy        | float64  | y-position with Gaussian noise σ=0.05 m                                 |
| x_true         | float64  | Noise-free x-position                                                    |
| y_true         | float64  | Noise-free y-position                                                    |
| msd_true       | float64  | True squared displacement from origin                                    |

**Lévy walk model:** At each flight, the walker moves with speed v₀=1.0 m/s in a uniformly random direction for a duration τ drawn from a Pareto distribution with tail index β (minimum flight time = dt). The process is observed at discrete times t = k·dt.

**Classes (10 trajectories each):**
| β    | α_theory | Regime                   |
|------|----------|--------------------------|
| 1.2  | 1.8      | Strong superdiffusion     |
| 1.5  | 1.5      | Moderate superdiffusion   |
| 1.8  | 1.2      | Mild superdiffusion       |
| 2.5  | 1.0      | Effectively normal        |

**Role in the study:** These trajectories serve as the ground truth for Lévy-stable superdiffusion. They are used to: (1) validate MSD-based and tail-fitting estimators of α, (2) characterise the heavy-tailed velocity increment PDFs (power-law tails with exponent μ = β + 1), and (3) serve as the target statistics that the point-vortex effective theory should reproduce.

---

### Loading the data

```python
import numpy as np

# Point-vortex tracers
pv = np.load('/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy', allow_pickle=False)
mask = pv['n_vortices'] == 20
traj = pv[mask]  # 2500 rows for N=20 vortices

# Lévy walks
lw = np.load('/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy', allow_pickle=False)
mask = lw['beta'] == 1.5
traj = lw[mask]  # 6000 rows for β=1.5 (α=1.5)
```

---

### Suggested analyses

1. **MSD scaling of point-vortex tracers:** Compute TAMSD for each tracer, fit power-law slope α_emp as a function of n_vortices. Test whether α_emp increases with n_vortices (denser vortex fields → stronger superdiffusion).

2. **Velocity increment PDF analysis:** Compute single-step displacement increments Δx, Δy for both datasets. Fit Lévy-stable distributions (stability index μ) to the tails. Compare tail exponents between point-vortex and Lévy walk datasets.

3. **Theoretical prediction of α from vortex density:** The 1/r vortex kernel predicts that the velocity PDF tail exponent is P(v) ~ v^{-(1+μ)} with μ = 2 (Cauchy distribution) for a uniform random vortex gas. This implies α = 3 - μ = 1 for a simple Cauchy, but corrections from finite N and circulation variance shift this. Derive and test the corrected prediction.

4. **Effective Fokker-Planck / fractional diffusion equation:** Fit a fractional diffusion equation ∂P/∂t = D_α ∇^α P to the tracer PDFs and extract the fractional Laplacian order α as a function of n_vortices.

5. **Comparison of point-vortex α vs Lévy walk α:** For each n_vortices configuration, identify the Lévy walk class (β value) whose MSD scaling and velocity PDF tail most closely matches the point-vortex statistics. This is the effective theory mapping.

6. **Ergodicity breaking:** Compute the ergodicity-breaking parameter EB = Var(TAMSD) / Mean(TAMSD)² as a function of lag time and compare between datasets. Lévy walks are non-ergodic; point-vortex tracers may or may not be depending on the chaoticity of the vortex dynamics.
