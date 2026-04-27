# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from scipy import stats
import time as time_module

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

def extract_increments(data, group_field, group_val, id_field='trajectory_id'):
    xs, ys, _ = extract_trajectories(data, group_field, group_val, id_field)
    dx_list, dy_list = [], []
    for x, y in zip(xs, ys):
        dx_list.append(np.diff(x))
        dy_list.append(np.diff(y))
    return np.concatenate(dx_list + dy_list)

def compute_pdf_on_grid(increments, bins):
    counts, _ = np.histogram(increments, bins=bins, density=True)
    return counts

def kl_divergence_symmetric(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return 0.5 * (kl_pq + kl_qp)

def compute_kl_matrix(pv, lw, nv_configs, beta_configs, n_bins=200):
    all_inc = []
    for nv in nv_configs:
        all_inc.append(extract_increments(pv, 'n_vortices', nv))
    for b in beta_configs:
        all_inc.append(extract_increments(lw, 'beta', b))
    all_combined = np.concatenate(all_inc)
    lo, hi = np.percentile(all_combined, 0.5), np.percentile(all_combined, 99.5)
    bins = np.linspace(lo, hi, n_bins + 1)
    pv_pdfs = {nv: compute_pdf_on_grid(extract_increments(pv, 'n_vortices', nv), bins) for nv in nv_configs}
    lw_pdfs = {b: compute_pdf_on_grid(extract_increments(lw, 'beta', b), bins) for b in beta_configs}
    kl_matrix = np.empty((len(nv_configs), len(beta_configs)))
    for i, nv in enumerate(nv_configs):
        for j, b in enumerate(beta_configs):
            kl_matrix[i, j] = kl_divergence_symmetric(pv_pdfs[nv], lw_pdfs[b])
    return kl_matrix, pv_pdfs, lw_pdfs, bins

def finite_n_corrected_alpha(N, A=400.0, L_box=20.0):
    N = np.asarray(N, dtype=float)
    r_min = np.sqrt(A / N)
    mu_eff = 2.0 / (1.0 + r_min / L_box)
    return np.minimum(3.0 - mu_eff, 2.0)

def estimate_proxy_lyapunov(xs, ys, times, frac=0.2):
    T_use = max(3, int(len(times) * frac))
    slopes = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dist = np.sqrt((xs[i][:T_use] - xs[j][:T_use])**2 + (ys[i][:T_use] - ys[j][:T_use])**2)
            dist = np.where(dist > 0, dist, np.nan)
            valid = np.isfinite(np.log(dist))
            if np.sum(valid) >= 3:
                slope, _, _, _, _ = stats.linregress(times[:T_use][valid], np.log(dist[valid]))
                slopes.append(slope)
    return np.array(slopes)

if __name__ == '__main__':
    pv, lw = load_datasets()
    nv_configs = [5, 10, 20, 40]
    beta_configs = [1.2, 1.5, 1.8, 2.5]
    kl_matrix, _, _, _ = compute_kl_matrix(pv, lw, nv_configs, beta_configs)
    results = []
    for i, nv in enumerate(nv_configs):
        best_idx = np.argmin(kl_matrix[i])
        xs, ys, times = extract_trajectories(pv, 'n_vortices', nv)
        lyap = estimate_proxy_lyapunov(xs, ys, times)
        results.append({
            'n_vortices': nv,
            'alpha_theory_corrected': finite_n_corrected_alpha(nv),
            'lyap_mean': np.nanmean(lyap),
            'lyap_std': np.nanstd(lyap),
            'best_beta': beta_configs[best_idx],
            'kl_div': kl_matrix[i, best_idx]
        })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(data_dir, 'mapping_results.csv'), index=False)
    print(df.to_string())