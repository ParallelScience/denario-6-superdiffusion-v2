# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
import os

def compute_tamsd_single(x, y, max_lag=None):
    T = len(x)
    if max_lag is None:
        max_lag = T // 2
    tamsd = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:T - lag]
        dy = y[lag:] - y[:T - lag]
        tamsd[lag - 1] = np.mean(dx**2 + dy**2)
    return tamsd

def fit_powerlaw_tamsd(tamsd, lag_times, frac_min=0.10, frac_max=0.60):
    T_total = lag_times[-1]
    mask = (lag_times >= frac_min * T_total) & (lag_times <= frac_max * T_total)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    log_t = np.log(lag_times[mask])
    log_msd = np.log(tamsd[mask])
    valid = np.isfinite(log_t) & np.isfinite(log_msd)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    slope, intercept, r_value, _, _ = linregress(log_t[valid], log_msd[valid])
    return slope, intercept, r_value**2

def hill_estimator(data, frac=0.125):
    sorted_data = np.sort(data)[::-1]
    k = max(int(len(sorted_data) * frac), 10)
    tail = sorted_data[:k]
    threshold = tail[-1]
    if threshold <= 0:
        return np.nan, np.nan, k
    mu_hill = 1.0 / np.mean(np.log(tail / threshold))
    log_x = np.log(np.sort(tail))
    n = len(tail)
    log_rank = np.log(np.arange(n, 0, -1) / (n + 1))
    valid = np.isfinite(log_x) & np.isfinite(log_rank)
    if valid.sum() < 3:
        return mu_hill, np.nan, k
    slope_fit, intercept_fit, _, _, _ = linregress(log_x[valid], log_rank[valid])
    log_rank_pred = slope_fit * log_x[valid] + intercept_fit
    mae = np.mean(np.abs(log_rank[valid] - log_rank_pred))
    return mu_hill, mae, k

def compute_eb(tamsd_matrix):
    mean_tamsd = np.mean(tamsd_matrix, axis=0)
    var_tamsd = np.var(tamsd_matrix, axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        eb = np.where(mean_tamsd**2 > 0, var_tamsd / mean_tamsd**2, np.nan)
    return eb

def rmse_func(a, b):
    return np.sqrt(np.mean((a - b)**2))

def apply_savgol(arr, window, polyorder):
    n = len(arr)
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    min_w = polyorder + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    w = min(w, n if n % 2 != 0 else n - 1)
    return savgol_filter(arr, w, polyorder)

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    pv_n_configs = np.array([5, 10, 20, 40])
    lw_beta_configs = np.array([1.2, 1.5, 1.8, 2.5])
    sg_window = 11
    sg_polyorder = 3
    pv_traj_ids = np.unique(pv['trajectory_id'])
    lw_traj_ids = np.unique(lw['trajectory_id'])
    pv_filtered_x = np.zeros(len(pv))
    pv_filtered_y = np.zeros(len(pv))
    pv_rmse_before = {int(n): {'x': [], 'y': []} for n in pv_n_configs}
    pv_rmse_after = {int(n): {'x': [], 'y': []} for n in pv_n_configs}
    for tid in pv_traj_ids:
        mask = pv['trajectory_id'] == tid
        xn = pv['x_noisy'][mask]
        yn = pv['y_noisy'][mask]
        xt = pv['x_true'][mask]
        yt = pv['y_true'][mask]
        xf = apply_savgol(xn, sg_window, sg_polyorder)
        yf = apply_savgol(yn, sg_window, sg_polyorder)
        pv_filtered_x[mask] = xf
        pv_filtered_y[mask] = yf
        n_key = int(pv['n_vortices'][mask][0])
        pv_rmse_before[n_key]['x'].append(rmse_func(xn, xt))
        pv_rmse_before[n_key]['y'].append(rmse_func(yn, yt))
        pv_rmse_after[n_key]['x'].append(rmse_func(xf, xt))
        pv_rmse_after[n_key]['y'].append(rmse_func(yf, yt))
    lw_filtered_x = np.zeros(len(lw))
    lw_filtered_y = np.zeros(len(lw))
    lw_rmse_before = {float(b): {'x': [], 'y': []} for b in lw_beta_configs}
    lw_rmse_after = {float(b): {'x': [], 'y': []} for b in lw_beta_configs}
    for tid in lw_traj_ids:
        mask = lw['trajectory_id'] == tid
        xn = lw['x_noisy'][mask]
        yn = lw['y_noisy'][mask]
        xt = lw['x_true'][mask]
        yt = lw['y_true'][mask]
        xf = apply_savgol(xn, sg_window, sg_polyorder)
        yf = apply_savgol(yn, sg_window, sg_polyorder)
        lw_filtered_x[mask] = xf
        lw_filtered_y[mask] = yf
        b_key = float(lw['beta'][mask][0])
        lw_rmse_before[b_key]['x'].append(rmse_func(xn, xt))
        lw_rmse_before[b_key]['y'].append(rmse_func(yn, yt))
        lw_rmse_after[b_key]['x'].append(rmse_func(xf, xt))
        lw_rmse_after[b_key]['y'].append(rmse_func(yf, yt))
    print("Analysis complete.")