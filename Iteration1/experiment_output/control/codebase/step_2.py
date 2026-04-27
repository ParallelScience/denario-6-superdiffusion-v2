# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

def compute_displacements_for_config(positions_x, positions_y, lag):
    dx_list = []
    dy_list = []
    for x, y in zip(positions_x, positions_y):
        T = len(x)
        if T > lag:
            dx_list.append(x[lag:] - x[:T - lag])
            dy_list.append(y[lag:] - y[:T - lag])
    if len(dx_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(dx_list), np.concatenate(dy_list)

def empirical_cf_abs(displacements, k_grid):
    if len(displacements) == 0:
        return np.zeros(len(k_grid))
    phase = np.outer(k_grid, displacements)
    phi_real = np.mean(np.cos(phase), axis=1)
    phi_imag = np.mean(np.sin(phase), axis=1)
    return np.sqrt(phi_real**2 + phi_imag**2)

def fit_fractional_diffusion(k_grid, phi_abs, dt_val, noise_floor=0.1):
    mask = (phi_abs > noise_floor) & (k_grid > 0) & (phi_abs < 0.9999)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    log_k = np.log(k_grid[mask])
    neg_log_phi = -np.log(phi_abs[mask])
    valid = np.isfinite(log_k) & np.isfinite(neg_log_phi) & (neg_log_phi > 0)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan
    log_neg_log_phi = np.log(neg_log_phi[valid])
    log_k_v = log_k[valid]
    slope, intercept, r_val, _, _ = linregress(log_k_v, log_neg_log_phi)
    alpha = slope
    log_D_alpha = intercept - np.log(dt_val)
    D_alpha = np.exp(log_D_alpha)
    r_squared = r_val**2
    if not (0.3 < alpha < 3.5) or not np.isfinite(D_alpha) or D_alpha <= 0:
        return np.nan, np.nan, np.nan
    return alpha, D_alpha, r_squared

def compute_alpha_profile(positions_x, positions_y, lag_indices, dt, k_grid, noise_floor=0.1):
    n = len(lag_indices)
    alpha_arr = np.full(n, np.nan)
    D_alpha_arr = np.full(n, np.nan)
    r2_arr = np.full(n, np.nan)
    for i, lag in enumerate(lag_indices):
        dx, dy = compute_displacements_for_config(positions_x, positions_y, lag)
        if len(dx) < 20:
            continue
        phi_x = empirical_cf_abs(dx, k_grid)
        phi_y = empirical_cf_abs(dy, k_grid)
        phi_combined = 0.5 * (phi_x + phi_y)
        dt_val = lag * dt
        a, D, r2 = fit_fractional_diffusion(k_grid, phi_combined, dt_val, noise_floor)
        alpha_arr[i] = a
        D_alpha_arr[i] = D
        r2_arr[i] = r2
    return alpha_arr, D_alpha_arr, r2_arr

def find_crossover_time(alpha_arr, lag_times, threshold_frac=0.05):
    valid = np.isfinite(alpha_arr)
    if valid.sum() < 4:
        return np.nan
    idx_valid = np.where(valid)[0]
    a_valid = alpha_arr[idx_valid]
    t_valid = lag_times[idx_valid]
    if len(a_valid) < 4:
        return np.nan
    deriv = np.abs(np.gradient(a_valid, t_valid))
    max_deriv = np.max(deriv)
    if max_deriv == 0:
        return float(t_valid[0])
    stable = deriv < threshold_frac * max_deriv
    stable_idx = np.where(stable)[0]
    if len(stable_idx) == 0:
        return np.nan
    return float(t_valid[stable_idx[0]])

def bootstrap_alpha_profile(positions_x, positions_y, lag_indices, dt, k_grid, n_bootstrap=200, noise_floor=0.1, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n_tracers = len(positions_x)
    n_lags = len(lag_indices)
    alpha_boot = np.full((n_bootstrap, n_lags), np.nan)
    lag_times = lag_indices * dt
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_tracers, size=n_tracers)
        bx = [positions_x[i] for i in idx]
        by = [positions_y[i] for i in idx]
        a_arr, _, _ = compute_alpha_profile(bx, by, lag_indices, dt, k_grid, noise_floor)
        alpha_boot[b] = a_arr
    alpha_ci_low = np.nanpercentile(alpha_boot, 2.5, axis=0)
    alpha_ci_high = np.nanpercentile(alpha_boot, 97.5, axis=0)
    tau_c_bootstrap = np.array([find_crossover_time(alpha_boot[b], lag_times) for b in range(n_bootstrap)])
    return alpha_ci_low, alpha_ci_high, tau_c_bootstrap, alpha_boot

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    pv = np.load(pv_path, allow_pickle=False)
    pv_dt = 0.05
    pv_n_configs = np.array([5, 10, 20, 40])
    n_k = 60
    k_grid = np.logspace(-2, 1.5, n_k)
    pv_n_lags_total = 500
    pv_lag_step = max(1, pv_n_lags_total // 80)
    pv_lag_indices = np.arange(1, pv_n_lags_total // 2, pv_lag_step)
    pv_lag_times = pv_lag_indices * pv_dt
    for n_v in pv_n_configs:
        mask = pv['n_vortices'] == n_v
        sub = pv[mask]
        traj_ids = np.unique(sub['trajectory_id'])
        positions_x = []
        positions_y = []
        for tid in traj_ids:
            tmask = sub['trajectory_id'] == tid
            sort_idx = np.argsort(sub['time'][tmask])
            positions_x.append(sub['x_true'][tmask][sort_idx])
            positions_y.append(sub['y_true'][tmask][sort_idx])
        alpha_arr, D_alpha_arr, r2_arr = compute_alpha_profile(positions_x, positions_y, pv_lag_indices, pv_dt, k_grid)
        ci_low, ci_high, tau_c_boot, alpha_boot_all = bootstrap_alpha_profile(positions_x, positions_y, pv_lag_indices, pv_dt, k_grid, n_bootstrap=200, noise_floor=0.1, rng_seed=42 + int(n_v))
        np.savez(os.path.join(data_dir, "pv_results_n" + str(n_v) + ".npz"), alpha=alpha_arr, D_alpha=D_alpha_arr, ci_low=ci_low, ci_high=ci_high, tau_c_boot=tau_c_boot)