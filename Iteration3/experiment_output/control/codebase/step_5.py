# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy import stats
from scipy.stats import levy_stable
from step_2 import reshape_trajectories

def compute_increments_at_lag(x2d, y2d, lag_steps):
    dx = x2d[:, lag_steps:] - x2d[:, :-lag_steps]
    dy = y2d[:, lag_steps:] - y2d[:, :-lag_steps]
    return dx.ravel(), dy.ravel()

def fit_ccdf_tail(dx, dy, top_frac=0.10):
    dr = np.sqrt(dx**2 + dy**2)
    dr_sorted = np.sort(dr)
    n = len(dr_sorted)
    threshold_idx = int(n * (1.0 - top_frac))
    tail_vals = dr_sorted[threshold_idx:]
    if len(tail_vals) < 20:
        return np.nan, np.nan, np.nan, np.nan
    ranks = np.arange(len(tail_vals), 0, -1, dtype=np.float64)
    ccdf = ranks / n
    log_r = np.log(tail_vals)
    log_c = np.log(ccdf)
    valid = np.isfinite(log_r) & np.isfinite(log_c) & (tail_vals > 0) & (ccdf > 0)
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r_val, p_val, se = stats.linregress(log_r[valid], log_c[valid])
    mu = -slope
    return mu, se, tail_vals[valid], ccdf[valid]

def fit_levy_stable_distribution(dx, dy, max_samples=50000):
    combined = (dx + dy) / np.sqrt(2.0)
    combined = combined[np.isfinite(combined)]
    if len(combined) > max_samples:
        rng = np.random.default_rng(42)
        combined = rng.choice(combined, size=max_samples, replace=False)
    try:
        params = levy_stable.fit(combined, floc=0)
        alpha_stable = params[0]
        beta_skew = params[1]
        scale = params[3]
        return alpha_stable, beta_skew, scale
    except Exception:
        return np.nan, np.nan, np.nan

def process_group_increments(x2d, y2d, lag_steps_list, dt, group_label):
    results = {}
    for lag in lag_steps_list:
        dx, dy = compute_increments_at_lag(x2d, y2d, lag)
        mu, mu_se, tail_r, tail_ccdf = fit_ccdf_tail(dx, dy, top_frac=0.10)
        lag_time = lag * dt
        results[lag] = {
            'dx': dx, 'dy': dy,
            'mu': mu, 'mu_se': mu_se,
            'tail_r': tail_r if tail_r is not None else np.array([]),
            'tail_ccdf': tail_ccdf if tail_ccdf is not None else np.array([]),
            'lag_time': lag_time
        }
    alpha_stable, beta_skew, scale = fit_levy_stable_distribution(*compute_increments_at_lag(x2d, y2d, 1))
    results['levy_stable'] = {'alpha_stable': alpha_stable, 'beta_skew': beta_skew, 'scale': scale}
    return results

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    dt_pv, dt_lw = 0.05, 0.1
    n_steps_pv, n_steps_lw = 500, 600
    lag_steps_list = [1, 5, 10, 20]
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    lw_traj_ids, lw_group_vals, lw_x2d, lw_y2d = reshape_trajectories(lw, 'trajectory_id', 'beta', n_steps_lw)
    n_vortices_list = sorted(np.unique(pv_group_vals).tolist())
    beta_list = sorted(np.unique(lw_group_vals).tolist())
    pv_results, lw_results = {}, {}
    for nv in n_vortices_list:
        mask = pv_group_vals == nv
        pv_results[nv] = process_group_increments(pv_x2d[mask], pv_y2d[mask], lag_steps_list, dt_pv, "PV_N" + str(int(nv)))
    for beta in beta_list:
        mask = lw_group_vals == beta
        lw_results[beta] = process_group_increments(lw_x2d[mask], lw_y2d[mask], lag_steps_list, dt_lw, "LW_beta" + str(beta))
    save_dict = {}
    for nv in n_vortices_list:
        nv_key = str(int(nv))
        for lag in lag_steps_list:
            lag_key = "lag" + str(lag)
            save_dict["pv_dx_" + nv_key + "_" + lag_key] = pv_results[nv][lag]['dx']
            save_dict["pv_dy_" + nv_key + "_" + lag_key] = pv_results[nv][lag]['dy']
            save_dict["pv_mu_" + nv_key + "_" + lag_key] = np.array([pv_results[nv][lag]['mu']])
        ls = pv_results[nv]['levy_stable']
        save_dict["pv_alpha_stable_" + nv_key] = np.array([ls['alpha_stable']])
    for beta in beta_list:
        beta_key = str(beta).replace('.', 'p')
        for lag in lag_steps_list:
            lag_key = "lag" + str(lag)
            save_dict["lw_dx_" + beta_key + "_" + lag_key] = lw_results[beta][lag]['dx']
            save_dict["lw_dy_" + beta_key + "_" + lag_key] = lw_results[beta][lag]['dy']
            save_dict["lw_mu_" + beta_key + "_" + lag_key] = np.array([lw_results[beta][lag]['mu']])
        ls = lw_results[beta]['levy_stable']
        save_dict["lw_alpha_stable_" + beta_key] = np.array([ls['alpha_stable']])
    save_dict["lag_steps"] = np.array(lag_steps_list)
    np.savez(os.path.join(data_dir, "increment_results.npz"), **save_dict)
    print("Saved results to data/increment_results.npz")