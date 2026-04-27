# filename: step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import warnings
import datetime
from step_1 import compute_tamsd_single, compute_eb
from step_2 import find_crossover_time
from step_3 import (get_positions_for_config_pv, get_positions_for_config_lw, compute_tamsd_ensemble, compute_hill_for_config)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
def get_ensemble_tamsd(positions_x, positions_y, dt):
    n_tracers = len(positions_x)
    T = len(positions_x[0])
    max_lag = T // 2
    lag_times = np.arange(1, max_lag + 1) * dt
    tamsd_matrix = np.zeros((n_tracers, max_lag))
    for i, (x, y) in enumerate(zip(positions_x, positions_y)):
        tamsd_matrix[i] = compute_tamsd_single(x, y, max_lag=max_lag)
    mean_tamsd = np.mean(tamsd_matrix, axis=0)
    return lag_times, mean_tamsd, tamsd_matrix
def fit_powerlaw_range(lag_times, tamsd, frac_min=0.10, frac_max=0.60):
    T_total = lag_times[-1]
    mask = (lag_times >= frac_min * T_total) & (lag_times <= frac_max * T_total)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    log_t = np.log(lag_times[mask])
    log_msd = np.log(tamsd[mask])
    valid = np.isfinite(log_t) & np.isfinite(log_msd)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    slope, intercept, r_val, _, _ = linregress(log_t[valid], log_msd[valid])
    t_fit = lag_times[mask][valid]
    msd_fit = np.exp(intercept) * t_fit**slope
    return slope, intercept, r_val**2, t_fit, msd_fit
def compute_displacement_pdf_logbins(positions_x, lag=1, n_bins=60):
    dx_all = []
    for x in positions_x:
        T = len(x)
        if T > lag:
            dx_all.append(x[lag:] - x[:T - lag])
    dx_signed = np.concatenate(dx_all)
    abs_dx = np.abs(dx_signed[dx_signed != 0])
    if len(abs_dx) < 10:
        return np.array([]), np.array([]), dx_signed
    p1 = np.percentile(abs_dx, 0.5)
    p99 = np.percentile(abs_dx, 99.5)
    if p1 <= 0:
        p1 = abs_dx[abs_dx > 0].min()
    bins = np.logspace(np.log10(p1), np.log10(p99), n_bins)
    counts, edges = np.histogram(abs_dx, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    return centers[valid], counts[valid], dx_signed
def fit_tail_powerlaw(centers, pdf, frac_tail=0.20):
    n = len(centers)
    k = max(int(n * frac_tail), 5)
    tail_x = centers[-k:]
    tail_y = pdf[-k:]
    valid = (tail_x > 0) & (tail_y > 0)
    if valid.sum() < 4:
        return np.nan, np.nan, np.array([]), np.array([])
    log_x = np.log(tail_x[valid])
    log_y = np.log(tail_y[valid])
    slope, intercept, r_val, _, _ = linregress(log_x, log_y)
    mu = -slope
    x_fit = np.array([tail_x[valid][0], tail_x[valid][-1]])
    y_fit = np.exp(intercept) * x_fit**slope
    return mu, r_val**2, x_fit, y_fit
if __name__ == '__main__':
    data_dir = 'data/'
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    pv_dt = 0.05
    lw_dt = 0.1
    pv_n_configs = np.array([5, 10, 20, 40])
    lw_beta_configs = np.array([1.2, 1.5, 1.8, 2.5])
    pv_positions = {}
    pv_alpha_emp = {}
    pv_r2_emp = {}
    for n_v in pv_n_configs:
        px, py = get_positions_for_config_pv(pv, n_v)
        alpha_vals, r2_vals, _, _ = compute_tamsd_ensemble(px, py, pv_dt)
        pv_positions[n_v] = (px, py)
        pv_alpha_emp[n_v] = float(np.nanmean(alpha_vals))
        pv_r2_emp[n_v] = float(np.nanmean(r2_vals))
    lw_positions = {}
    lw_alpha_emp = {}
    lw_r2_emp = {}
    for beta_val in lw_beta_configs:
        lx, ly = get_positions_for_config_lw(lw, beta_val)
        alpha_vals, r2_vals, _, _ = compute_tamsd_ensemble(lx, ly, lw_dt)
        lw_positions[beta_val] = (lx, ly)
        lw_alpha_emp[beta_val] = float(np.nanmean(alpha_vals))
        lw_r2_emp[beta_val] = float(np.nanmean(r2_vals))
    print('Analysis complete. Figures would be generated here.')