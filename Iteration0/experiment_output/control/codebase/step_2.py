# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import os

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
        xs.append(row['x_true'][order])
        ys.append(row['y_true'][order])
    times = np.sort(np.unique(sub['time']))
    return xs, ys, times

def compute_tamsd_single(x, y, max_lag):
    T = len(x)
    tamsd = np.empty(max_lag)
    for delta in range(1, max_lag + 1):
        dx = x[delta:] - x[:T - delta]
        dy = y[delta:] - y[:T - delta]
        tamsd[delta - 1] = np.mean(dx * dx + dy * dy)
    return tamsd

def compute_all_tamsds(xs, ys, max_lag):
    stack = np.array([compute_tamsd_single(x, y, max_lag) for x, y in zip(xs, ys)])
    ensemble_tamsd = np.mean(stack, axis=0)
    return stack, ensemble_tamsd

def compute_local_alpha(lag_times, ensemble_tamsd, window=15):
    log_t = np.log(lag_times)
    log_msd = np.log(ensemble_tamsd)
    n = len(log_t)
    alpha_t = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        sl = slice(i - half, i + half + 1)
        lt = log_t[sl]
        lm = log_msd[sl]
        if np.any(~np.isfinite(lm)):
            continue
        coeffs = np.polyfit(lt, lm, 1)
        alpha_t[i] = coeffs[0]
    valid_mask = np.isfinite(alpha_t)
    return alpha_t, valid_mask

def find_crossover_time(lag_times, alpha_t, valid_mask, stability_window=20, stability_thresh=0.08):
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < stability_window + 5:
        return lag_times[valid_indices[0]], valid_indices[0]
    for i in range(stability_window, len(valid_indices)):
        window_alphas = alpha_t[valid_indices[i - stability_window:i]]
        if np.std(window_alphas) < stability_thresh:
            return lag_times[valid_indices[i - stability_window]], valid_indices[i - stability_window]
    mid = len(valid_indices) // 2
    return lag_times[valid_indices[mid]], valid_indices[mid]

def fit_powerlaw_asymptotic(lag_times, ensemble_tamsd, crossover_idx, max_lag):
    end_idx = min(int(max_lag * 0.8), len(lag_times) - 1)
    log_t_cross = np.log(lag_times[crossover_idx])
    log_t_end = np.log(lag_times[end_idx])
    log_t_mid = 0.5 * (log_t_cross + log_t_end)
    fit_start = int(np.searchsorted(np.log(lag_times), log_t_mid))
    fit_start = max(fit_start, crossover_idx)
    fit_end = end_idx
    if fit_end <= fit_start + 2:
        fit_start = crossover_idx
        fit_end = end_idx
    lt = np.log(lag_times[fit_start:fit_end + 1])
    lm = np.log(ensemble_tamsd[fit_start:fit_end + 1])
    valid = np.isfinite(lm)
    if np.sum(valid) < 3:
        return np.nan, np.nan, lag_times[fit_start], lag_times[fit_end], fit_start, fit_end
    coeffs = np.polyfit(lt[valid], lm[valid], 1)
    return coeffs[0], coeffs[1], lag_times[fit_start], lag_times[fit_end], fit_start, fit_end

def bootstrap_alpha(stack, lag_times, crossover_idx, max_lag, n_boot=500):
    n_traj = stack.shape[0]
    end_idx = min(int(max_lag * 0.8), len(lag_times) - 1)
    log_t_cross = np.log(lag_times[crossover_idx])
    log_t_end = np.log(lag_times[end_idx])
    log_t_mid = 0.5 * (log_t_cross + log_t_end)
    fit_start = int(np.searchsorted(np.log(lag_times), log_t_mid))
    fit_start = max(fit_start, crossover_idx)
    fit_end = end_idx
    if fit_end <= fit_start + 2:
        fit_start = crossover_idx
        fit_end = end_idx
    lt = np.log(lag_times[fit_start:fit_end + 1])
    alpha_boot = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, size=n_traj)
        boot_tamsd = np.mean(stack[idx], axis=0)
        lm = np.log(boot_tamsd[fit_start:fit_end + 1])
        valid = np.isfinite(lm)
        if np.sum(valid) < 3:
            alpha_boot[b] = np.nan
            continue
        coeffs = np.polyfit(lt[valid], lm[valid], 1)
        alpha_boot[b] = coeffs[0]
    return alpha_boot

def process_group(data, group_field, group_val, id_field, dt, label, max_lag_frac=0.8, n_boot=500):
    xs, ys, times = extract_trajectories(data, group_field, group_val, id_field)
    T = len(times)
    max_lag = int(T * max_lag_frac)
    lag_times = np.arange(1, max_lag + 1) * dt
    stack, ensemble_tamsd = compute_all_tamsds(xs, ys, max_lag)
    alpha_t, valid_mask = compute_local_alpha(lag_times, ensemble_tamsd, window=15)
    crossover_time, crossover_idx = find_crossover_time(lag_times, alpha_t, valid_mask)
    alpha_fit, log_D, fit_t_min, fit_t_max, fit_start, fit_end = fit_powerlaw_asymptotic(lag_times, ensemble_tamsd, crossover_idx, max_lag)
    alpha_boot = bootstrap_alpha(stack, lag_times, crossover_idx, max_lag, n_boot=n_boot)
    alpha_boot_clean = alpha_boot[np.isfinite(alpha_boot)]
    alpha_emp = np.median(alpha_boot_clean) if len(alpha_boot_clean) > 0 else alpha_fit
    ci_low = np.percentile(alpha_boot_clean, 2.5) if len(alpha_boot_clean) > 0 else np.nan
    ci_high = np.percentile(alpha_boot_clean, 97.5) if len(alpha_boot_clean) > 0 else np.nan
    print('  ' + label + ':')
    print('    N_traj=' + str(len(xs)) + ', T=' + str(T) + ' steps, max_lag=' + str(max_lag))
    print('    Crossover time: ' + str(round(crossover_time, 3)) + ' s (idx=' + str(crossover_idx) + ')')
    print('    Fit range: t_min=' + str(round(fit_t_min, 3)) + ' s, t_max=' + str(round(fit_t_max, 3)) + ' s')
    print('    alpha_fit (OLS)=' + str(round(alpha_fit, 4)))
    print('    alpha_emp (bootstrap median)=' + str(round(alpha_emp, 4)) + ' 95% CI=[' + str(round(ci_low, 4)) + ', ' + str(round(ci_high, 4)) + ']')
    return {'label': label, 'group_val': group_val, 'lag_times': lag_times, 'ensemble_tamsd': ensemble_tamsd, 'individual_tamsds': stack, 'alpha_t': alpha_t, 'valid_mask': valid_mask, 'crossover_time': crossover_time, 'crossover_idx': crossover_idx, 'alpha_fit': alpha_fit, 'log_D': log_D, 'fit_t_min': fit_t_min, 'fit_t_max': fit_t_max, 'fit_start': fit_start, 'fit_end': fit_end, 'alpha_boot': alpha_boot, 'alpha_emp': alpha_emp, 'ci_low': ci_low, 'ci_high': ci_high}

if __name__ == '__main__':
    pv, lw = load_datasets()
    pv_dt, lw_dt, n_boot = 0.05, 0.1, 500
    nv_configs = [5, 10, 20, 40]
    beta_configs = [1.2, 1.5, 1.8, 2.5]
    results = []
    for nv in nv_configs:
        results.append(process_group(pv, 'n_vortices', nv, 'trajectory_id', pv_dt, 'N=' + str(nv), n_boot=n_boot))
    for b in beta_configs:
        results.append(process_group(lw, 'beta', b, 'trajectory_id', lw_dt, 'beta=' + str(b), n_boot=n_boot))
    print('Pipeline complete.')