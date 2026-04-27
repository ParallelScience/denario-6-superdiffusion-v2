# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

def load_datasets(data_dir):
    pv = np.load(os.path.join(data_dir, 'point_vortex_tracers_validated.npz'))['data']
    lw = np.load(os.path.join(data_dir, 'levy_walk_trajectories_validated.npz'))['data']
    return pv, lw

def extract_traj_matrix(arr, group_field, group_val, id_field, time_field, x_field, y_field):
    mask = arr[group_field] == group_val
    sub = arr[mask]
    traj_ids = sorted(np.unique(sub[id_field]).tolist())
    n = len(traj_ids)
    T = np.sum(sub[id_field] == traj_ids[0])
    X = np.empty((n, T))
    Y = np.empty((n, T))
    for i, tid in enumerate(traj_ids):
        tmask = sub[id_field] == tid
        traj = sub[tmask]
        sort_idx = np.argsort(traj[time_field])
        X[i] = traj[x_field][sort_idx]
        Y[i] = traj[y_field][sort_idx]
    times = np.sort(sub[sub[id_field] == traj_ids[0]][time_field])
    return X, Y, times, traj_ids

def compute_tamsd_matrix(X, Y, max_lag_frac=0.25):
    n_trajs, T = X.shape
    max_lag = max(2, int(T * max_lag_frac))
    lags = np.arange(1, max_lag + 1)
    tamsd_matrix = np.empty((n_trajs, len(lags)))
    for k, lag in enumerate(lags):
        dx = X[:, lag:] - X[:, :T - lag]
        dy = Y[:, lag:] - Y[:, :T - lag]
        tamsd_matrix[:, k] = np.mean(dx**2 + dy**2, axis=1)
    return lags, tamsd_matrix

def fit_powerlaw_loglog(lags, tamsd_vec, dt):
    t = lags * dt
    valid = (tamsd_vec > 0) & np.isfinite(tamsd_vec)
    if valid.sum() < 3:
        return np.nan, np.nan
    log_t = np.log(t[valid])
    log_msd = np.log(tamsd_vec[valid])
    coeffs = np.polyfit(log_t, log_msd, 1)
    return coeffs[0], coeffs[1]

def compute_vacf_vectorized(X, Y, dt, max_lag_frac=0.5):
    n_trajs, T = X.shape
    Vx = np.diff(X, axis=1) / dt
    Vy = np.diff(Y, axis=1) / dt
    n_vel = Vx.shape[1]
    max_lag = max(2, int(n_vel * max_lag_frac))
    v2 = np.mean(Vx**2 + Vy**2, axis=1)
    vacf_all = np.zeros((n_trajs, max_lag))
    vacf_all[:, 0] = 1.0
    for lag in range(1, max_lag):
        n_pairs = n_vel - lag
        if n_pairs <= 0:
            break
        corr = np.mean(Vx[:, :n_pairs] * Vx[:, lag:lag + n_pairs] + Vy[:, :n_pairs] * Vy[:, lag:lag + n_pairs], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vacf_all[:, lag] = np.where(v2 > 0, corr / v2, 0.0)
    mean_vacf = np.mean(vacf_all, axis=0)
    zero_crossing_lag = 10
    for i in range(1, len(mean_vacf)):
        if mean_vacf[i] <= 0:
            zero_crossing_lag = i
            break
    lag_times = np.arange(max_lag) * dt
    return lag_times, mean_vacf, zero_crossing_lag

if __name__ == '__main__':
    data_dir = 'data/'
    pv, lw = load_datasets(data_dir)
    pv_n_values = [5, 10, 20, 40]
    lw_beta_values = [1.2, 1.5, 1.8, 2.5]
    dt_pv = 0.05
    dt_lw = 0.1
    for nv in pv_n_values:
        X, Y, _, _ = extract_traj_matrix(pv, 'n_vortices', nv, 'trajectory_id', 'time', 'x_true', 'y_true')
        _, _, zc = compute_vacf_vectorized(X, Y, dt_pv)
        print('N=' + str(nv) + ' VACF zero-crossing lag=' + str(zc) + ' steps')
    print('Analysis complete.')