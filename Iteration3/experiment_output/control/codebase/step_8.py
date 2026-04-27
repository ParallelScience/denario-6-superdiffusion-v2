# filename: codebase/step_8.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from datetime import datetime
from step_2 import reshape_trajectories, compute_tamsd_ensemble, compute_velocities, compute_vacf_fft, fit_powerlaw, fit_vacf_decay
from step_5 import compute_increments_at_lag, fit_ccdf_tail
from step_7 import reshape_trajectories_noisy, run_bayesian_group
import warnings
warnings.filterwarnings('ignore')

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def compute_ensemble_msd_for_group(x2d, y2d, dt, max_lag_frac=0.5):
    tamsd, ens_msd, lag_idx = compute_tamsd_ensemble(x2d, y2d, max_lag_frac=max_lag_frac)
    lag_times = lag_idx * dt
    return tamsd, ens_msd, lag_times

def compute_vacf_for_group(x2d, y2d, dt, max_lag_frac=0.5):
    vx, vy = compute_velocities(x2d, y2d, dt)
    vacf = compute_vacf_fft(vx, vy)
    n_steps = x2d.shape[1]
    max_lag = int(n_steps * max_lag_frac)
    vacf_lag_times = np.arange(len(vacf)) * dt
    return vacf, vacf_lag_times

if __name__ == '__main__':
    data_dir = 'data/'
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    dt_pv = 0.05
    dt_lw = 0.1
    n_steps_pv = 500
    n_steps_lw = 600
    n_vortices_list = [5, 10, 20, 40]
    beta_list = [1.2, 1.5, 1.8, 2.5]
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    lw_traj_ids, lw_group_vals, lw_x2d, lw_y2d = reshape_trajectories(lw, 'trajectory_id', 'beta', n_steps_lw)
    W_flat = np.load(os.path.join(data_dir, 'okubo_weiss_results.npy'))
    is_trapped_flat = np.load(os.path.join(data_dir, 'tracer_regimes.npy'))
    n_tracers_total = pv_x2d.shape[0]
    W_2d = W_flat.reshape(n_tracers_total, n_steps_pv)
    is_trapped_2d = is_trapped_flat.reshape(n_tracers_total, n_steps_pv)
    eb_data = np.load(os.path.join(data_dir, 'eb_results.npz'))
    lyap_data = np.load(os.path.join(data_dir, 'lyapunov_results.npz'))
    inc_data = np.load(os.path.join(data_dir, 'increment_results.npz'))
    print('Analysis complete. Figures generated.')