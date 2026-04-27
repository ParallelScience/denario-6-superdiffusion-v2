# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress, ks_2samp
import os
import warnings
from step_1 import compute_tamsd_single, fit_powerlaw_tamsd, hill_estimator, compute_eb

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

def get_positions_for_config_pv(pv, n_v):
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
    return positions_x, positions_y

def get_positions_for_config_lw(lw, beta_val):
    mask = np.abs(lw['beta'] - beta_val) < 1e-6
    sub = lw[mask]
    traj_ids = np.unique(sub['trajectory_id'])
    positions_x = []
    positions_y = []
    for tid in traj_ids:
        tmask = sub['trajectory_id'] == tid
        sort_idx = np.argsort(sub['time'][tmask])
        positions_x.append(sub['x_true'][tmask][sort_idx])
        positions_y.append(sub['y_true'][tmask][sort_idx])
    return positions_x, positions_y

def compute_tamsd_ensemble(positions_x, positions_y, dt, frac_min=0.10, frac_max=0.60):
    n_tracers = len(positions_x)
    T = len(positions_x[0])
    max_lag = T // 2
    lag_times = np.arange(1, max_lag + 1) * dt
    tamsd_matrix = np.zeros((n_tracers, max_lag))
    alpha_vals = np.zeros(n_tracers)
    r2_vals = np.zeros(n_tracers)
    for i, (x, y) in enumerate(zip(positions_x, positions_y)):
        tamsd = compute_tamsd_single(x, y, max_lag=max_lag)
        tamsd_matrix[i] = tamsd
        alpha, _, r2 = fit_powerlaw_tamsd(tamsd, lag_times, frac_min, frac_max)
        alpha_vals[i] = alpha if np.isfinite(alpha) else np.nan
        r2_vals[i] = r2 if np.isfinite(r2) else np.nan
    return alpha_vals, r2_vals, tamsd_matrix, lag_times

def compute_hill_for_config(positions_x, positions_y, frac=0.125):
    dx_all = []
    dy_all = []
    for x, y in zip(positions_x, positions_y):
        dx_all.append(np.diff(x))
        dy_all.append(np.diff(y))
    dx_all = np.concatenate(dx_all)
    dy_all = np.concatenate(dy_all)
    abs_disp = np.sqrt(dx_all**2 + dy_all**2)
    abs_disp = abs_disp[abs_disp > 0]
    mu_hill, mae, k = hill_estimator(abs_disp, frac=frac)
    return mu_hill, mae, k

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
    lw_alpha_theory = np.array([1.8, 1.5, 1.2, 1.0])
    pv_alpha_emp = {}
    pv_mu_hill = {}
    for n_v in pv_n_configs:
        px, py = get_positions_for_config_pv(pv, n_v)
        alpha_vals, _, _, _ = compute_tamsd_ensemble(px, py, pv_dt)
        mu_h, _, _ = compute_hill_for_config(px, py, frac=0.125)
        pv_alpha_emp[n_v] = float(np.nanmean(alpha_vals))
        pv_mu_hill[n_v] = float(mu_h)
    lw_alpha_emp = {}
    lw_mu_hill = {}
    for beta_val in lw_beta_configs:
        lx, ly = get_positions_for_config_lw(lw, beta_val)
        alpha_vals, _, _, _ = compute_tamsd_ensemble(lx, ly, lw_dt)
        mu_h, _, _ = compute_hill_for_config(lx, ly, frac=0.125)
        lw_alpha_emp[beta_val] = float(np.nanmean(alpha_vals))
        lw_mu_hill[beta_val] = float(mu_h)
    mapping = {}
    for n_v in pv_n_configs:
        best_beta = None
        min_dist = float('inf')
        for b in lw_beta_configs:
            dist = np.sqrt((pv_alpha_emp[n_v] - lw_alpha_emp[b])**2 + (pv_mu_hill[n_v] - lw_mu_hill[b])**2)
            if dist < min_dist:
                min_dist = dist
                best_beta = b
        mapping[n_v] = best_beta
    np.save(os.path.join(data_dir, 'effective_theory_mapping.npy'), mapping)
    print('Mapping saved to data/effective_theory_mapping.npy')