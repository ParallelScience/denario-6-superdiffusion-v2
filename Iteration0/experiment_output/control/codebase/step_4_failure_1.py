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
import os
import warnings

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
        valid = (phi_abs > 1e-6) & np.isfinite(phi_abs)
        if np.sum(valid) < 5: continue
        log_phi = np.log(phi_abs[valid])
        lhs_list.append(-log_phi)
        log_k_list.append(log_k[valid])
        log_t_list.append(np.full(np.sum(valid), np.log(t)))
    if not lhs_list: return np.nan, np.nan, np.nan
    lhs = np.concatenate(lhs_list)
    log_k_all = np.concatenate(log_k_list)
    log_t_all = np.concatenate(log_t_list)
    valid_lhs = lhs > 0
    if np.sum(valid_lhs) < 5: return np.nan, np.nan, np.nan
    lhs, log_k_all, log_t_all = lhs[valid_lhs], log_k_all[valid_lhs], log_t_all[valid_lhs]
    log_lhs = np.log(lhs)
    A = np.column_stack([log_k_all, log_t_all, np.ones(len(log_lhs))])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_lhs, rcond=None)
    return coeffs[0], np.exp(coeffs[2]), 0

def ks_ad_test(displacements, alpha_stable, D_alpha, t):
    if not np.isfinite(alpha_stable) or alpha_stable <= 0 or alpha_stable > 2: return np.nan, np.nan, np.nan, np.nan
    if not np.isfinite(D_alpha) or D_alpha <= 0: return np.nan, np.nan, np.nan, np.nan
    scale = (D_alpha * t) ** (1.0 / alpha_stable)
    n = len(displacements)
    if n < 10: return np.nan, np.nan, np.nan, np.nan
    sample = displacements if n <= 5000 else np.random.default_rng(42).choice(displacements, size=5000, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ks_stat, ks_pval = stats.kstest(sample, lambda x: stats.levy_stable.cdf(x, alpha_stable, 0, loc=0, scale=scale))
        except Exception: ks_stat, ks_pval = np.nan, np.nan
    return ks_stat, ks_pval, np.nan, np.nan

if __name__ == '__main__':
    pv, lw = load_datasets()
    k_grid = np.logspace(-2, 1, 100)
    results = []
    for n_v in [5, 10, 20, 40]:
        xs, ys, times = extract_trajectories(pv, 'n_vortices', n_v)
        res = {'label': 'N=' + str(n_v)}
        snap_indices = np.linspace(10, len(times)-1, 5, dtype=int)
        phi_abs_list = [compute_cf_vectorized(np.concatenate([x[s]-x[0] for x in xs]), k_grid) for s in snap_indices]
        alpha, D, _ = fit_fractional_diffusion(k_grid, phi_abs_list, times[snap_indices])
        res.update({'alpha': alpha, 'D': D})
        results.append(res)
    print('Results:', results)