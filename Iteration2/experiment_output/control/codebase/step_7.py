# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
import warnings

def load_datasets(data_dir):
    pv = np.load(os.path.join(data_dir, 'point_vortex_tracers_validated.npz'))['data']
    lw = np.load(os.path.join(data_dir, 'levy_walk_trajectories_validated.npz'))['data']
    return pv, lw

def extract_increments(arr, group_field, group_val, id_field, time_field, x_field, y_field, segment=None):
    mask = arr[group_field] == group_val
    sub = arr[mask]
    traj_ids = sorted(np.unique(sub[id_field]).tolist())
    dx_list, dy_list = [], []
    for tid in traj_ids:
        tmask = sub[id_field] == tid
        traj = sub[tmask]
        sort_idx = np.argsort(traj[time_field])
        x = traj[x_field][sort_idx]
        y = traj[y_field][sort_idx]
        T = len(x)
        if segment == 'early':
            end = max(2, int(0.25 * T))
            x, y = x[:end], y[:end]
        elif segment == 'late':
            start = min(T - 2, int(0.75 * T))
            x, y = x[start:], y[start:]
        dx_list.append(np.diff(x))
        dy_list.append(np.diff(y))
    dx_all = np.concatenate(dx_list)
    dy_all = np.concatenate(dy_list)
    return dx_all, dy_all, np.concatenate([np.abs(dx_all), np.abs(dy_all)])

def hill_estimator(data, tail_fraction=0.10):
    n = len(data)
    k = max(3, int(tail_fraction * n))
    sorted_data = np.sort(data)
    tail = sorted_data[-k:]
    threshold = tail[0]
    if threshold <= 0: return np.nan
    log_ratios = np.log(tail[1:] / threshold)
    mean_log = np.mean(log_ratios)
    return 1.0 / mean_log if mean_log > 0 else np.nan

def fit_levy_stable_to_increments(dx, dy):
    combined = np.concatenate([dx, dy])
    combined = combined[np.isfinite(combined)]
    if len(combined) < 20: return np.nan, np.nan, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            params = levy_stable.fit(combined, floc=0)
            return params[0], params[1], params[3], params[2]
        except Exception: return np.nan, np.nan, np.nan, np.nan

def compute_eamsd_alpha(arr, group_field, group_val, id_field, time_field, msd_field, frac_start=0.10, frac_end=0.70):
    mask = arr[group_field] == group_val
    sub = arr[mask]
    times = np.sort(np.unique(sub[time_field]))
    traj_ids = np.unique(sub[id_field])
    T = len(times)
    msd_matrix = np.full((len(traj_ids), T), np.nan)
    for i, tid in enumerate(traj_ids):
        tmask = sub[id_field] == tid
        traj = sub[tmask]
        sort_idx = np.argsort(traj[time_field])
        msd_matrix[i] = traj[msd_field][sort_idx]
    eamsd = np.nanmean(msd_matrix, axis=0)
    i_start, i_end = max(1, int(frac_start * T)), max(2, int(frac_end * T))
    t_fit, msd_fit = times[i_start:i_end], eamsd[i_start:i_end]
    valid = (t_fit > 0) & (msd_fit > 0) & np.isfinite(msd_fit)
    if valid.sum() < 3: return np.nan, np.nan, times, eamsd
    coeffs, cov = np.polyfit(np.log(t_fit[valid]), np.log(msd_fit[valid]), 1, cov=True)
    return coeffs[0], np.sqrt(cov[0, 0]), times, eamsd

if __name__ == '__main__':
    data_dir = 'data/'
    pv, lw = load_datasets(data_dir)
    pv_n_values = [5, 10, 20, 40]
    lw_beta_values = [1.2, 1.5, 1.8, 2.5]
    print('=' * 80)
    print('STEP 7: PHASE-SPACE MAPPING AND LEVY-STABLE FITTING')
    print('=' * 80)
    for nv in pv_n_values:
        a, _, _, _ = compute_eamsd_alpha(pv, 'n_vortices', nv, 'trajectory_id', 'time', 'msd_true')
        dx, dy, _ = extract_increments(pv, 'n_vortices', nv, 'trajectory_id', 'time', 'x_true', 'y_true')
        mu = hill_estimator(np.abs(np.concatenate([dx, dy])))
        print('N=' + str(nv) + ' | alpha_emp=' + str(round(a, 4)) + ' | Hill_mu=' + str(round(mu, 4)))