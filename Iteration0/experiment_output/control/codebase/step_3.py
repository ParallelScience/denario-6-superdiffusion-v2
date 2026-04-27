# filename: codebase/step_3.py
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
import time
import warnings

data_dir = 'data/'

def load_datasets():
    pv = np.load('/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy', allow_pickle=False)
    lw = np.load('/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy', allow_pickle=False)
    return pv, lw

def extract_increments_group(data, group_field, group_val, id_field='trajectory_id'):
    mask = data[group_field] == group_val
    sub = data[mask]
    tids = np.unique(sub[id_field])
    dx_list = []
    dy_list = []
    for tid in tids:
        tmask = sub[id_field] == tid
        row = sub[tmask]
        order = np.argsort(row['time'])
        x = row['x_true'][order]
        y = row['y_true'][order]
        dx_list.append(np.diff(x))
        dy_list.append(np.diff(y))
    delta_x = np.concatenate(dx_list)
    delta_y = np.concatenate(dy_list)
    return delta_x, delta_y, len(tids)

def hill_estimator(data, k_min=10, k_max_frac=0.25):
    x = np.sort(data)
    n = len(x)
    k_max = max(k_min + 1, int(n * k_max_frac))
    k_vals = np.arange(k_min, k_max + 1)
    mu_hill = np.empty(len(k_vals))
    for i, k in enumerate(k_vals):
        threshold = x[n - k - 1]
        if threshold <= 0:
            mu_hill[i] = np.nan
            continue
        log_ratios = np.log(x[n - k:] / threshold)
        if np.any(log_ratios <= 0):
            mu_hill[i] = np.nan
            continue
        mu_hill[i] = k / np.sum(log_ratios)
    return k_vals, mu_hill

def select_optimal_k(k_vals, mu_hill, window=30):
    valid = np.isfinite(mu_hill)
    if np.sum(valid) < window + 1:
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return k_vals[len(k_vals) // 2], np.nanmedian(mu_hill)
        mid = valid_idx[len(valid_idx) // 2]
        return k_vals[mid], mu_hill[mid]
    n = len(mu_hill)
    min_var = np.inf
    best_center = window // 2
    for i in range(window // 2, n - window // 2):
        sl = slice(i - window // 2, i + window // 2 + 1)
        seg = mu_hill[sl]
        if np.sum(np.isfinite(seg)) < window // 2:
            continue
        v = np.nanvar(seg)
        if v < min_var:
            min_var = v
            best_center = i
    return k_vals[best_center], mu_hill[best_center]

def compute_hill_both_tails(delta_x, delta_y, k_min=10, k_max_frac=0.25, window=30):
    abs_inc = np.abs(np.concatenate([delta_x, delta_y]))
    abs_inc = abs_inc[abs_inc > 0]
    k_vals, mu_hill = hill_estimator(abs_inc, k_min=k_min, k_max_frac=k_max_frac)
    k_opt, mu_opt = select_optimal_k(k_vals, mu_hill, window=window)
    return k_vals, mu_hill, k_opt, mu_opt, abs_inc

def fit_levy_stable(increments, max_samples=50000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    data = increments.copy()
    if len(data) > max_samples:
        idx = rng.choice(len(data), size=max_samples, replace=False)
        data = data[idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        params = stats.levy_stable.fit(data, f0=None, floc=0)
    return params[0], params[1], params[3], params[2]

def compute_increment_pdf_loglog(increments, n_bins=80):
    abs_inc = np.abs(increments)
    abs_inc = abs_inc[abs_inc > 0]
    lo = np.percentile(abs_inc, 0.1)
    hi = np.percentile(abs_inc, 99.9)
    if lo <= 0:
        lo = abs_inc[abs_inc > 0].min()
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts, edges = np.histogram(abs_inc, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    return centers[mask], counts[mask]

if __name__ == '__main__':
    pv, lw = load_datasets()
    nv_configs = [5, 10, 20, 40]
    beta_configs = [1.2, 1.5, 1.8, 2.5]
    alpha_theory_map = {1.2: 1.8, 1.5: 1.5, 1.8: 1.2, 2.5: 1.0}
    print('STEP 3: VELOCITY INCREMENT PDF AND TAIL INDEX ESTIMATION')
    for nv in nv_configs:
        dx, dy, n_traj = extract_increments_group(pv, 'n_vortices', nv)
        all_inc = np.concatenate([dx, dy])
        k_vals, mu_hill, k_opt, mu_opt, abs_inc = compute_hill_both_tails(dx, dy)
        alpha_s, beta_s, scale_s, loc_s = fit_levy_stable(all_inc)
        print('N=' + str(nv) + ' vortices: mu_Hill=' + str(round(mu_opt, 4)) + ', alpha_stable=' + str(round(alpha_s, 4)))
    for b in beta_configs:
        dx, dy, n_traj = extract_increments_group(lw, 'beta', b)
        all_inc = np.concatenate([dx, dy])
        k_vals, mu_hill, k_opt, mu_opt, abs_inc = compute_hill_both_tails(dx, dy)
        alpha_s, beta_s, scale_s, loc_s = fit_levy_stable(all_inc)
        print('beta=' + str(b) + ': mu_Hill=' + str(round(mu_opt, 4)) + ', alpha_stable=' + str(round(alpha_s, 4)))