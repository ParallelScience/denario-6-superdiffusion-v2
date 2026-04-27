# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

def load_datasets(data_dir):
    pv = np.load(os.path.join(data_dir, 'point_vortex_tracers_validated.npz'))['data']
    lw = np.load(os.path.join(data_dir, 'levy_walk_trajectories_validated.npz'))['data']
    return pv, lw

def extract_increments_per_config(arr, group_field, group_values, x_field, y_field, id_field, time_field):
    increments = {}
    for gv in group_values:
        mask = arr[group_field] == gv
        sub = arr[mask]
        traj_ids = np.unique(sub[id_field])
        all_abs_inc = []
        for tid in traj_ids:
            tmask = sub[id_field] == tid
            traj = sub[tmask]
            sort_idx = np.argsort(traj[time_field])
            x = traj[x_field][sort_idx]
            y = traj[y_field][sort_idx]
            dx = np.diff(x)
            dy = np.diff(y)
            abs_inc = np.concatenate([np.abs(dx), np.abs(dy)])
            all_abs_inc.append(abs_inc)
        increments[gv] = np.concatenate(all_abs_inc)
    return increments

def hill_estimator(data, tail_fraction):
    n = len(data)
    k = max(3, int(tail_fraction * n))
    sorted_data = np.sort(data)
    tail = sorted_data[-k:]
    threshold = tail[0]
    if threshold <= 0:
        return np.nan
    log_ratios = np.log(tail[1:] / threshold)
    if len(log_ratios) == 0 or np.mean(log_ratios) == 0:
        return np.nan
    mu = 1.0 / np.mean(log_ratios)
    return mu

def apply_savgol_and_compute_increments(arr, group_field, group_values, id_field, time_field, window_sizes, polyorder=2):
    filtered_increments = {}
    for gv in group_values:
        filtered_increments[gv] = {}
        mask = arr[group_field] == gv
        sub = arr[mask]
        traj_ids = np.unique(sub[id_field])
        for ws in window_sizes:
            all_abs_inc = []
            for tid in traj_ids:
                tmask = sub[id_field] == tid
                traj = sub[tmask]
                sort_idx = np.argsort(traj[time_field])
                xn = traj['x_noisy'][sort_idx]
                yn = traj['y_noisy'][sort_idx]
                if len(xn) < ws:
                    continue
                xf = savgol_filter(xn, window_length=ws, polyorder=polyorder)
                yf = savgol_filter(yn, window_length=ws, polyorder=polyorder)
                dx = np.diff(xf)
                dy = np.diff(yf)
                abs_inc = np.concatenate([np.abs(dx), np.abs(dy)])
                all_abs_inc.append(abs_inc)
            filtered_increments[gv][ws] = np.concatenate(all_abs_inc) if all_abs_inc else np.array([])
    return filtered_increments

if __name__ == '__main__':
    data_dir = 'data/'
    pv, lw = load_datasets(data_dir)
    group_values = [5, 10, 20, 40]
    inc_true = extract_increments_per_config(pv, 'n_vortices', group_values, 'x_true', 'y_true', 'trajectory_id', 'time')
    inc_noisy = extract_increments_per_config(pv, 'n_vortices', group_values, 'x_noisy', 'y_noisy', 'trajectory_id', 'time')
    tail_fractions = np.linspace(0.05, 0.30, 26)
    mu_true = {gv: [hill_estimator(inc_true[gv], tf) for tf in tail_fractions] for gv in group_values}
    mu_noisy = {gv: [hill_estimator(inc_noisy[gv], tf) for tf in tail_fractions] for gv in group_values}
    var_across = np.var([mu_true[gv] for gv in group_values], axis=0)
    stable_idx = np.argmin(var_across)
    stable_tf = tail_fractions[stable_idx]
    window_sizes = [3, 5, 7, 11, 15, 21]
    inc_filt = apply_savgol_and_compute_increments(pv, 'n_vortices', group_values, 'trajectory_id', 'time', window_sizes)
    mu_filt = {gv: [hill_estimator(inc_filt[gv][ws], stable_tf) for ws in window_sizes] for gv in group_values}
    np.savez_compressed(os.path.join(data_dir, 'analysis_results.npz'), mu_true=mu_true, mu_noisy=mu_noisy, mu_filt=mu_filt, stable_tf=stable_tf)
    print('Analysis complete. Results saved to data/analysis_results.npz')