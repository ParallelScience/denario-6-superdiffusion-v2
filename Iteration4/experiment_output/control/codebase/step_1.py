# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
import os

PV_PATH = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
LW_PATH = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
DATA_DIR = 'data/'
N_CONFIGS = [5, 10, 20, 40]
BETA_CONFIGS = [1.2, 1.5, 1.8, 2.5]
PV_DT = 0.05
LW_DT = 0.1
PV_NSTEPS = 500
LW_NSTEPS = 600

def extract_pv_trajectories(pv, n_vortices):
    mask = pv['n_vortices'] == n_vortices
    sub = pv[mask]
    traj_ids = np.unique(sub['trajectory_id'])
    n_traj = len(traj_ids)
    x_mat = np.zeros((n_traj, PV_NSTEPS))
    y_mat = np.zeros((n_traj, PV_NSTEPS))
    for i, tid in enumerate(traj_ids):
        tmask = sub['trajectory_id'] == tid
        rows = sub[tmask]
        order = np.argsort(rows['time'])
        x_mat[i] = rows['x_true'][order]
        y_mat[i] = rows['y_true'][order]
    return traj_ids, x_mat, y_mat

def compute_msd(x_mat, y_mat, max_lag=None):
    n_traj, n_steps = x_mat.shape
    if max_lag is None:
        max_lag = n_steps - 1
    lags = np.arange(1, max_lag + 1)
    msd_per_traj = np.zeros((n_traj, max_lag))
    for k_idx, k in enumerate(lags):
        dx = x_mat[:, k:] - x_mat[:, :n_steps - k]
        dy = y_mat[:, k:] - y_mat[:, :n_steps - k]
        sd = dx ** 2 + dy ** 2
        msd_per_traj[:, k_idx] = sd.mean(axis=1)
    msd_mean = msd_per_traj.mean(axis=0)
    return lags, msd_mean, msd_per_traj

if __name__ == '__main__':
    pv = np.load(PV_PATH, allow_pickle=False)
    lw = np.load(LW_PATH, allow_pickle=False)
    for n in N_CONFIGS:
        ids, x, y = extract_pv_trajectories(pv, n)
        lags, msd_mean, msd_per_traj = compute_msd(x, y)
        np.save(os.path.join(DATA_DIR, 'msd_pv_n' + str(n) + '.npy'), msd_mean)
    for b in BETA_CONFIGS:
        mask = np.abs(lw['beta'] - b) < 1e-6
        sub = lw[mask]
        ids = np.unique(sub['trajectory_id'])
        x = np.array([sub[sub['trajectory_id'] == tid]['x_true'] for tid in ids])
        y = np.array([sub[sub['trajectory_id'] == tid]['y_true'] for tid in ids])
        lags, msd_mean, msd_per_traj = compute_msd(x, y)
        np.save(os.path.join(DATA_DIR, 'msd_lw_beta' + str(b) + '.npy'), msd_mean)