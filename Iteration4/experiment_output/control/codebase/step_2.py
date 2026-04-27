# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import time

PV_PATH = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
LW_PATH = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
DATA_DIR = 'data/'

def extract_pv_traj(pv, n_vortices):
    mask = pv['n_vortices'] == n_vortices
    sub = pv[mask]
    traj_ids = np.unique(sub['trajectory_id'])
    n_traj = len(traj_ids)
    x_mat = np.zeros((n_traj, 500))
    y_mat = np.zeros((n_traj, 500))
    for i, tid in enumerate(traj_ids):
        tmask = sub['trajectory_id'] == tid
        rows = sub[tmask]
        order = np.argsort(rows['time'])
        x_mat[i] = rows['x_true'][order]
        y_mat[i] = rows['y_true'][order]
    return traj_ids, x_mat, y_mat

def extract_lw_traj(lw, beta):
    mask = np.abs(lw['beta'] - beta) < 1e-6
    sub = lw[mask]
    traj_ids = np.unique(sub['trajectory_id'])
    n_traj = len(traj_ids)
    x_mat = np.zeros((n_traj, 600))
    y_mat = np.zeros((n_traj, 600))
    for i, tid in enumerate(traj_ids):
        tmask = sub['trajectory_id'] == tid
        rows = sub[tmask]
        order = np.argsort(rows['time'])
        x_mat[i] = rows['x_true'][order]
        y_mat[i] = rows['y_true'][order]
    return traj_ids, x_mat, y_mat

def compute_msd_full(x_mat, y_mat, max_lag=None):
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
    return lags, msd_per_traj.mean(axis=0), msd_per_traj.std(axis=0), msd_per_traj

if __name__ == '__main__':
    pv = np.load(PV_PATH, allow_pickle=False)
    lw = np.load(LW_PATH, allow_pickle=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n in [5, 10, 20, 40]:
        _, x, y = extract_pv_traj(pv, n)
        lags, msd, std, _ = compute_msd_full(x, y)
        ax1.loglog(lags * 0.05, msd, label='N='+str(n))
        ax1.fill_between(lags * 0.05, np.maximum(msd - std, 1e-6), msd + std, alpha=0.2)
    ax1.plot(lags * 0.05, (lags * 0.05)**1 * msd[0]/(0.05), 'k--', alpha=0.5, label='alpha=1')
    ax1.plot(lags * 0.05, (lags * 0.05)**2 * msd[0]/(0.05**2), 'k:', alpha=0.5, label='alpha=2')
    ax1.set_title('Point-Vortex Tracers')
    ax1.set_xlabel('Lag time (s)')
    ax1.set_ylabel('MSD (m^2)')
    ax1.legend()
    for b in [1.2, 1.5, 1.8, 2.5]:
        _, x, y = extract_lw_traj(lw, b)
        lags, msd, std, _ = compute_msd_full(x, y)
        ax2.loglog(lags * 0.1, msd, label='beta='+str(b))
        ax2.fill_between(lags * 0.1, np.maximum(msd - std, 1e-6), msd + std, alpha=0.2)
        ax2.plot(lags * 0.1, (lags * 0.1)**(3-b) * msd[0]/(0.1**(3-b)), '--', alpha=0.5)
    ax2.plot(lags * 0.1, (lags * 0.1)**1 * msd[0]/(0.1), 'k--', alpha=0.5, label='alpha=1')
    ax2.plot(lags * 0.1, (lags * 0.1)**2 * msd[0]/(0.1**2), 'k:', alpha=0.5, label='alpha=2')
    ax2.set_title('Levy Walks')
    ax2.set_xlabel('Lag time (s)')
    ax2.set_ylabel('MSD (m^2)')
    ax2.legend()
    fname = 'fig2_msd_scaling_' + str(int(time.time())) + '.png'
    plt.savefig(os.path.join(DATA_DIR, fname))
    print('Saved to ' + os.path.join(DATA_DIR, fname))