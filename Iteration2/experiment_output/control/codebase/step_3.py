# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_pv_data(data_dir):
    pv = np.load(os.path.join(data_dir, 'point_vortex_tracers_validated.npz'))['data']
    return pv

def compute_speeds_per_tracer(pv, n_vortices_val, dt=0.05):
    mask = pv['n_vortices'] == n_vortices_val
    sub = pv[mask]
    traj_ids = sorted(np.unique(sub['trajectory_id']).tolist())
    speeds_dict = {}
    times_dict = {}
    for tid in traj_ids:
        tmask = sub['trajectory_id'] == tid
        traj = sub[tmask]
        sort_idx = np.argsort(traj['time'])
        x = traj['x_true'][sort_idx]
        y = traj['y_true'][sort_idx]
        t = traj['time'][sort_idx]
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2) / dt
        speed_full = np.concatenate([speed, [speed[-1]]])
        speeds_dict[tid] = speed_full
        times_dict[tid] = t
    return traj_ids, speeds_dict, times_dict

def compute_pooled_threshold(speeds_dict, percentile=25):
    all_speeds = np.concatenate(list(speeds_dict.values()))
    threshold = np.percentile(all_speeds, percentile)
    return threshold

def find_residence_times(speed_array, threshold, dt=0.05):
    trapped_mask = speed_array < threshold
    residence_times = []
    in_trap = False
    count = 0
    for val in trapped_mask:
        if val:
            in_trap = True
            count += 1
        else:
            if in_trap:
                residence_times.append(count * dt)
                count = 0
                in_trap = False
    if in_trap and count > 0:
        residence_times.append(count * dt)
    return np.array(residence_times), trapped_mask

def mle_powerlaw(data, tau_min=None):
    if tau_min is None:
        tau_min = np.min(data)
    tail = data[data >= tau_min]
    n = len(tail)
    if n < 3:
        return np.nan, np.nan, tau_min, n, -np.inf
    log_ratios = np.log(tail / tau_min)
    mean_log = np.mean(log_ratios)
    if mean_log <= 0:
        return np.nan, np.nan, tau_min, n, -np.inf
    gamma = 1.0 + 1.0 / mean_log
    gamma_std = (gamma - 1.0) / np.sqrt(n)
    log_likelihood = n * np.log(gamma - 1.0) - n * np.log(tau_min) - gamma * np.sum(np.log(tail / tau_min))
    return gamma, gamma_std, tau_min, n, log_likelihood

def mle_powerlaw_exponential_cutoff(data, tau_min=None):
    if tau_min is None:
        tau_min = np.min(data)
    tail = data[data >= tau_min]
    n = len(tail)
    if n < 5:
        return np.nan, np.nan, -np.inf
    def neg_log_likelihood(params):
        g, lam = params
        if g <= 1.0 or lam <= 0:
            return 1e12
        log_norm = np.log(lam**(g - 1.0)) - np.log(np.sum([k**(-g) * np.exp(-lam * k * tau_min) for k in range(1, 500)]) * tau_min**(1 - g))
        ll = n * log_norm - g * np.sum(np.log(tail)) - lam * np.sum(tail)
        return -ll
    gamma_init, _, _, _, _ = mle_powerlaw(tail, tau_min=tau_min)
    if np.isnan(gamma_init):
        gamma_init = 2.0
    lambda_init = 1.0 / np.mean(tail)
    try:
        result = minimize(neg_log_likelihood, x0=[gamma_init, lambda_init], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
        gamma_cut, lambda_cut = result.x
        log_likelihood_cut = -result.fun
    except Exception:
        gamma_cut, lambda_cut, log_likelihood_cut = np.nan, np.nan, -np.inf
    return gamma_cut, lambda_cut, log_likelihood_cut

if __name__ == '__main__':
    data_dir = 'data/'
    dt = 0.05
    pv = load_pv_data(data_dir)
    n_values = [5, 10, 20, 40]
    all_speeds = {}
    all_times = {}
    all_traj_ids = {}
    for nv in n_values:
        tids, spd, tms = compute_speeds_per_tracer(pv, nv, dt=dt)
        all_speeds[nv] = spd
        all_times[nv] = tms
        all_traj_ids[nv] = tids
    thresholds = {}
    for nv in n_values:
        thresholds[nv] = compute_pooled_threshold(all_speeds[nv], percentile=25)
    all_residence_times = {}
    for nv in n_values:
        pooled_rt = []
        for tid in all_traj_ids[nv]:
            rt, _ = find_residence_times(all_speeds[nv][tid], thresholds[nv], dt=dt)
            pooled_rt.append(rt)
        all_residence_times[nv] = np.concatenate(pooled_rt) if any(len(r) > 0 for r in pooled_rt) else np.array([dt])
    fit_results = {}
    for nv in n_values:
        rt = all_residence_times[nv]
        tau_min_fit = np.median(rt)
        gamma, gamma_std, tau_min_used, n_tail, ll_pl = mle_powerlaw(rt, tau_min=tau_min_fit)
        gamma_cut, lambda_cut, ll_cut = mle_powerlaw_exponential_cutoff(rt, tau_min=tau_min_fit)
        fit_results[nv] = {'gamma': gamma, 'gamma_std': gamma_std, 'tau_min': tau_min_used, 'n_tail': n_tail, 'll_powerlaw': ll_pl, 'gamma_cut': gamma_cut, 'lambda_cut': lambda_cut, 'll_cutoff': ll_cut, 'delta_ll': ll_cut - ll_pl}
    print('Residence time analysis complete.')