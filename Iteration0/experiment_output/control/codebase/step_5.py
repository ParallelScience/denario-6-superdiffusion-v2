# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import time

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
        xs.append(row['x_true'][order].copy())
        ys.append(row['y_true'][order].copy())
    times = np.sort(np.unique(sub['time']))
    return xs, ys, times

def compute_tamsd_matrix_vectorized(xs, ys, max_lag):
    n_traj = len(xs)
    T = len(xs[0])
    X = np.array(xs)
    Y = np.array(ys)
    tamsd_matrix = np.empty((n_traj, max_lag), dtype=np.float64)
    for delta in range(1, max_lag + 1):
        dx = X[:, delta:] - X[:, :T - delta]
        dy = Y[:, delta:] - Y[:, :T - delta]
        tamsd_matrix[:, delta - 1] = np.mean(dx * dx + dy * dy, axis=1)
    return tamsd_matrix

def compute_eb_from_matrix(tamsd_matrix):
    n_traj = tamsd_matrix.shape[0]
    mean_tamsd = np.mean(tamsd_matrix, axis=0)
    if n_traj < 2:
        return np.full(tamsd_matrix.shape[1], np.nan), mean_tamsd
    var_tamsd = np.var(tamsd_matrix, axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        eb = np.where(mean_tamsd > 0, var_tamsd / (mean_tamsd ** 2), np.nan)
    return eb, mean_tamsd

def bootstrap_eb(tamsd_matrix, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    n_traj, n_lags = tamsd_matrix.shape
    eb_boot = np.empty((n_boot, n_lags), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, size=n_traj)
        boot_mat = tamsd_matrix[idx]
        eb_b, _ = compute_eb_from_matrix(boot_mat)
        eb_boot[b] = eb_b
    eb_ci_low = np.nanpercentile(eb_boot, 2.5, axis=0)
    eb_ci_high = np.nanpercentile(eb_boot, 97.5, axis=0)
    return eb_ci_low, eb_ci_high

def long_lag_eb_stats(eb, frac=0.2):
    n = len(eb)
    start = int(n * (1.0 - frac))
    seg = eb[start:]
    seg = seg[np.isfinite(seg)]
    if len(seg) == 0:
        return np.nan, np.nan
    return float(np.mean(seg)), float(np.std(seg))

if __name__ == '__main__':
    pv, lw = load_datasets()
    pv_dt, lw_dt, max_lag_frac, n_boot = 0.05, 0.1, 0.8, 200
    nv_configs, beta_configs = [5, 10, 20, 40], [1.2, 1.5, 1.8, 2.5]
    pv_colors, lw_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    pv_results, lw_results = {}, {}
    for nv in nv_configs:
        xs, ys, times = extract_trajectories(pv, 'n_vortices', nv)
        max_lag = int(len(times) * max_lag_frac)
        tamsd_mat = compute_tamsd_matrix_vectorized(xs, ys, max_lag)
        eb, mean_tamsd = compute_eb_from_matrix(tamsd_mat)
        eb_ci_low, eb_ci_high = bootstrap_eb(tamsd_mat, n_boot=n_boot)
        pv_results[nv] = {'lag_times': np.arange(1, max_lag + 1) * pv_dt, 'eb': eb, 'eb_ci_low': eb_ci_low, 'eb_ci_high': eb_ci_high, 'eb_long_mean': long_lag_eb_stats(eb)[0]}
    for b in beta_configs:
        xs, ys, times = extract_trajectories(lw, 'beta', b)
        max_lag = int(len(times) * max_lag_frac)
        tamsd_mat = compute_tamsd_matrix_vectorized(xs, ys, max_lag)
        eb, mean_tamsd = compute_eb_from_matrix(tamsd_mat)
        eb_ci_low, eb_ci_high = bootstrap_eb(tamsd_mat, n_boot=n_boot)
        lw_results[b] = {'lag_times': np.arange(1, max_lag + 1) * lw_dt, 'eb': eb, 'eb_ci_low': eb_ci_low, 'eb_ci_high': eb_ci_high, 'eb_long_mean': long_lag_eb_stats(eb)[0]}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, nv in enumerate(nv_configs):
        r = pv_results[nv]
        axes[0, 0].loglog(r['lag_times'], r['eb'], color=pv_colors[i], label='N=' + str(nv))
        axes[0, 0].fill_between(r['lag_times'], r['eb_ci_low'], r['eb_ci_high'], color=pv_colors[i], alpha=0.2)
    axes[0, 0].set(xlabel='Lag Time (s)', ylabel='EB', title='(a) EB vs Lag Time: Point-Vortex')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    for i, b in enumerate(beta_configs):
        r = lw_results[b]
        axes[0, 1].loglog(r['lag_times'], r['eb'], color=lw_colors[i], label='beta=' + str(b))
        axes[0, 1].fill_between(r['lag_times'], r['eb_ci_low'], r['eb_ci_high'], color=lw_colors[i], alpha=0.2)
    axes[0, 1].set(xlabel='Lag Time (s)', ylabel='EB', title='(b) EB vs Lag Time: Levy Walk')
    axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(nv_configs, [pv_results[nv]['eb_long_mean'] for nv in nv_configs], 'o-')
    axes[1, 0].set(xlabel='N_vortices', ylabel='EB_long', title='(c) EB_long vs N_vortices')
    axes[1, 1].plot(beta_configs, [lw_results[b]['eb_long_mean'] for b in beta_configs], 'o-')
    axes[1, 1].set(xlabel='beta', ylabel='EB_long', title='(d) EB_long vs beta')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'step5_eb_plot.png'))
    print('Saved to ' + os.path.join(data_dir, 'step5_eb_plot.png'))