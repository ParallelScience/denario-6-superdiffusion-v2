# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy import stats
from step_2 import reshape_trajectories

def reshape_trajectories_noisy(data, id_field, group_field, n_steps, sort_field='time'):
    unique_pairs = []
    seen = set()
    for row in data:
        key = (float(row[id_field]), float(row[group_field]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)
    n_tracers = len(unique_pairs)
    x2d_noisy = np.zeros((n_tracers, n_steps), dtype=np.float64)
    y2d_noisy = np.zeros((n_tracers, n_steps), dtype=np.float64)
    for i, (tid, gv) in enumerate(unique_pairs):
        mask = (data[id_field] == tid) & (data[group_field] == gv)
        rows = data[mask]
        sort_idx = np.argsort(rows[sort_field])
        rows = rows[sort_idx]
        x2d_noisy[i] = rows['x_noisy']
        y2d_noisy[i] = rows['y_noisy']
    return np.array([p[0] for p in unique_pairs]), np.array([p[1] for p in unique_pairs]), x2d_noisy, y2d_noisy

def compute_per_tracer_tamsd(x2d, y2d, max_lag_frac=0.4):
    n_tracers, n_steps = x2d.shape
    max_lag = int(n_steps * max_lag_frac)
    tamsd = np.zeros((n_tracers, max_lag), dtype=np.float64)
    for delta in range(1, max_lag + 1):
        dx = x2d[:, delta:] - x2d[:, :n_steps - delta]
        dy = y2d[:, delta:] - y2d[:, :n_steps - delta]
        sd = dx ** 2 + dy ** 2
        tamsd[:, delta - 1] = sd.mean(axis=1)
    return tamsd, np.arange(1, max_lag + 1)

def bayesian_alpha_grid(tamsd_matrix, lag_times, alpha_grid, fit_start_frac=0.05, fit_end_frac=0.85):
    n_lags = tamsd_matrix.shape[1]
    n_alpha = len(alpha_grid)
    n = len(lag_times)
    i_start = max(1, int(n * fit_start_frac))
    i_end = max(i_start + 2, int(n * fit_end_frac))
    lt = lag_times[i_start:i_end]
    tm = tamsd_matrix[:, i_start:i_end]
    log_lt = np.log(lt)
    log_tm = np.log(np.clip(tm, 1e-30, None))
    log_posterior = np.zeros(n_alpha, dtype=np.float64)
    n_pts = log_tm.shape[0] * log_tm.shape[1]
    for k, alpha in enumerate(alpha_grid):
        predicted = log_lt[np.newaxis, :] * alpha
        residuals = log_tm - predicted
        log_D_hat = residuals.mean()
        sigma2_hat = ((residuals - log_D_hat) ** 2).mean()
        log_posterior[k] = -0.5 * n_pts * np.log(max(sigma2_hat, 1e-10))
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    map_alpha = alpha_grid[np.argmax(posterior)]
    cdf = np.cumsum(posterior)
    return log_posterior, posterior, map_alpha, alpha_grid[np.searchsorted(cdf, 0.025)], alpha_grid[np.searchsorted(cdf, 0.975)]

def run_bayesian_group(x2d, y2d, dt, alpha_grid, max_lag_frac=0.4):
    tamsd, lag_idx = compute_per_tracer_tamsd(x2d, y2d, max_lag_frac=max_lag_frac)
    log_post, post, map_a, ci_lo, ci_hi = bayesian_alpha_grid(tamsd, lag_idx * dt, alpha_grid)
    return {'map_alpha': map_a, 'ci_low': ci_lo, 'ci_high': ci_hi}

if __name__ == '__main__':
    pv = np.load('/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy', allow_pickle=False)
    lw = np.load('/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy', allow_pickle=False)
    alpha_grid = np.linspace(0.5, 2.5, 200)
    _, lw_gv_true, lw_x_true, lw_y_true = reshape_trajectories(lw, 'trajectory_id', 'beta', 600)
    _, lw_gv_noisy, lw_x_noisy, lw_y_noisy = reshape_trajectories_noisy(lw, 'trajectory_id', 'beta', 600)
    for beta in sorted(np.unique(lw_gv_true)):
        mask = lw_gv_true == beta
        res_true = run_bayesian_group(lw_x_true[mask], lw_y_true[mask], 0.1, alpha_grid)
        res_noisy = run_bayesian_group(lw_x_noisy[mask], lw_y_noisy[mask], 0.1, alpha_grid)
        print("beta=" + str(beta) + " | alpha_true=" + str(round(res_true['map_alpha'], 3)) + " | alpha_noisy=" + str(round(res_noisy['map_alpha'], 3)))
    _, pv_gv_true, pv_x_true, pv_y_true = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', 500)
    _, pv_gv_noisy, pv_x_noisy, pv_y_noisy = reshape_trajectories_noisy(pv, 'trajectory_id', 'n_vortices', 500)
    for nv in sorted(np.unique(pv_gv_true)):
        mask = pv_gv_true == nv
        res = run_bayesian_group(pv_x_noisy[mask], pv_y_noisy[mask], 0.05, alpha_grid)
        print("n_vortices=" + str(nv) + " | alpha_noisy=" + str(round(res['map_alpha'], 3)) + " CI=[" + str(round(res['ci_low'], 3)) + ", " + str(round(res['ci_high'], 3)) + "]")