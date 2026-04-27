# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

def load_datasets(data_dir):
    pv = np.load(os.path.join(data_dir, 'point_vortex_tracers_validated.npz'))['data']
    lw = np.load(os.path.join(data_dir, 'levy_walk_trajectories_validated.npz'))['data']
    return pv, lw

def compute_eamsd(arr, group_field, group_val, id_field, time_field, msd_field):
    mask = arr[group_field] == group_val
    sub = arr[mask]
    times = np.sort(np.unique(sub[time_field]))
    traj_ids = np.unique(sub[id_field])
    n_trajs = len(traj_ids)
    T = len(times)
    msd_matrix = np.full((n_trajs, T), np.nan)
    for i, tid in enumerate(traj_ids):
        tmask = sub[id_field] == tid
        traj = sub[tmask]
        sort_idx = np.argsort(traj[time_field])
        msd_matrix[i] = traj[msd_field][sort_idx]
    eamsd = np.nanmean(msd_matrix, axis=0)
    eamsd_std = np.nanstd(msd_matrix, axis=0)
    return times, eamsd, eamsd_std, n_trajs

def fit_powerlaw_intermediate(times, eamsd, frac_start=0.10, frac_end=0.70):
    T = len(times)
    i_start = max(1, int(frac_start * T))
    i_end = max(i_start + 3, int(frac_end * T))
    t_fit = times[i_start:i_end]
    msd_fit = eamsd[i_start:i_end]
    valid = (t_fit > 0) & (msd_fit > 0) & np.isfinite(msd_fit)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, t_fit, msd_fit, np.full_like(t_fit, np.nan)
    log_t = np.log(t_fit[valid])
    log_msd = np.log(msd_fit[valid])
    coeffs, cov = np.polyfit(log_t, log_msd, 1, cov=True)
    alpha_emp = coeffs[0]
    alpha_std = np.sqrt(cov[0, 0])
    log_D = coeffs[1]
    msd_predicted = np.exp(log_D) * t_fit**alpha_emp
    return alpha_emp, alpha_std, log_D, t_fit, msd_fit, msd_predicted

if __name__ == '__main__':
    data_dir = 'data/'
    pv, lw = load_datasets(data_dir)
    pv_n_values = [5, 10, 20, 40]
    print('=' * 70)
    print('STEP 6: MSD POWER-LAW FITTING AND LEVY WALK NULL MODEL COMPARISON')
    print('=' * 70)
    for nv in pv_n_values:
        times, eamsd, eamsd_std, n_trajs = compute_eamsd(pv, 'n_vortices', nv, 'trajectory_id', 'time', 'msd_true')
        alpha_emp, alpha_std, log_D, t_fit, msd_fit, msd_pred = fit_powerlaw_intermediate(times, eamsd)
        print('N_vortices: ' + str(nv) + ' | alpha_emp: ' + str(round(alpha_emp, 4)) + ' | alpha_std: ' + str(round(alpha_std, 4)))