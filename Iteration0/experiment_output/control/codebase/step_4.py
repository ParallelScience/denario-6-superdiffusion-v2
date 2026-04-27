# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import pandas as pd

data_dir = 'data/'

def load_datasets():
    pv = np.load('/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy', allow_pickle=False)
    lw = np.load('/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy', allow_pickle=False)
    return pv, lw

def extract_trajectories(data, group_field, group_val, id_field='trajectory_id'):
    mask = data[group_field] == group_val
    sub = data[mask]
    tids = np.unique(sub[id_field])
    xs, ys = [], []
    for tid in tids:
        tmask = sub[id_field] == tid
        row = sub[tmask]
        order = np.argsort(row['time'])
        xs.append(row['x_true'][order])
        ys.append(row['y_true'][order])
    times = np.sort(np.unique(sub['time']))
    return xs, ys, times

def get_displacements_at_snap(xs, ys, snap_idx):
    dx = np.array([x[snap_idx] - x[0] for x in xs if snap_idx < len(x)])
    dy = np.array([y[snap_idx] - y[0] for y in ys if snap_idx < len(y)])
    return np.concatenate([dx, dy])

def compute_cf_vectorized(displacements, k_grid):
    angles = np.outer(k_grid, displacements)
    cos_mean = np.mean(np.cos(angles), axis=1)
    sin_mean = np.mean(np.sin(angles), axis=1)
    phi_abs = np.sqrt(cos_mean**2 + sin_mean**2)
    return phi_abs

def fit_fractional_diffusion(k_grid, phi_abs_list, t_list):
    log_k = np.log(k_grid)
    lhs_list, log_k_list, log_t_list = [], [], []
    for phi_abs, t in zip(phi_abs_list, t_list):
        valid = (phi_abs > 1e-6) & (phi_abs < 1.0 - 1e-9) & np.isfinite(phi_abs)
        if np.sum(valid) < 5: continue
        neg_log_phi = -np.log(phi_abs[valid])
        pos_mask = neg_log_phi > 0
        if np.sum(pos_mask) < 5: continue
        lhs_list.append(neg_log_phi[pos_mask])
        log_k_list.append(log_k[valid][pos_mask])
        log_t_list.append(np.full(np.sum(pos_mask), np.log(t)))
    if not lhs_list: return np.nan, np.nan, np.nan
    lhs = np.concatenate(lhs_list)
    log_k_all = np.concatenate(log_k_list)
    log_t_all = np.concatenate(log_t_list)
    log_lhs = np.log(lhs)
    adjusted_lhs = log_lhs - log_t_all
    A = np.column_stack([log_k_all, np.ones(len(log_lhs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, adjusted_lhs, rcond=None)
    return coeffs[0], np.exp(coeffs[1]), np.sqrt(np.mean((adjusted_lhs - (coeffs[0] * log_k_all + coeffs[1]))**2))

def ks_test_levy_stable(displacements, alpha_stable, D_alpha, t, seed=42):
    if not (np.isfinite(alpha_stable) and 0 < alpha_stable <= 2): return np.nan, np.nan
    if not (np.isfinite(D_alpha) and D_alpha > 0): return np.nan, np.nan
    scale = (D_alpha * t) ** (1.0 / alpha_stable)
    if not np.isfinite(scale) or scale <= 0: return np.nan, np.nan
    n = len(displacements)
    if n < 10: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = displacements if n <= 3000 else rng.choice(displacements, size=3000, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ks_stat, ks_pval = stats.kstest(sample, lambda x: stats.levy_stable.cdf(x, alpha_stable, 0, loc=0, scale=scale))
        except Exception: ks_stat, ks_pval = np.nan, np.nan
    return ks_stat, ks_pval

def ad_test_two_sample(displacements, alpha_stable, D_alpha, t, n_sim=2000, seed=42):
    if not (np.isfinite(alpha_stable) and 0 < alpha_stable <= 2): return np.nan, np.nan
    if not (np.isfinite(D_alpha) and D_alpha > 0): return np.nan, np.nan
    scale = (D_alpha * t) ** (1.0 / alpha_stable)
    if not np.isfinite(scale) or scale <= 0: return np.nan, np.nan
    n = len(displacements)
    if n < 10: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = displacements if n <= 2000 else rng.choice(displacements, size=2000, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            sim = stats.levy_stable.rvs(alpha_stable, 0, loc=0, scale=scale, size=n_sim, random_state=seed)
            result = stats.anderson_ksamp([sample, sim])
            return result.statistic, result.pvalue
        except Exception: return np.nan, np.nan

if __name__ == '__main__':
    pv, lw = load_datasets()
    k_grid = np.logspace(-2, 1, 100)
    results = []
    for n_v in [5, 10, 20, 40]:
        xs, ys, times = extract_trajectories(pv, 'n_vortices', n_v)
        snap_indices = np.unique(np.round(np.logspace(np.log10(2), np.log10(int(len(times)*0.8)-1), 5)).astype(int))
        phi_abs_list = [compute_cf_vectorized(get_displacements_at_snap(xs, ys, s), k_grid) for s in snap_indices]
        alpha, D, rms = fit_fractional_diffusion(k_grid, phi_abs_list, times[snap_indices])
        results.append({'label': 'N=' + str(n_v), 'alpha_cf': alpha, 'D_alpha': D, 'rms': rms})
    print('Results:', results)