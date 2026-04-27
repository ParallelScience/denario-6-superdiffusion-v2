# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.special import digamma
from scipy.stats import levy_stable, gaussian_kde
from step_2 import reshape_trajectories

def knn_mi_vectorized(x, y, k=5):
    n = len(x)
    if n < 2 * k + 2:
        return np.nan
    xy = np.column_stack([x, y])
    dists_xy = np.abs(xy[:, None, :] - xy[None, :, :]).max(axis=2)
    np.fill_diagonal(dists_xy, np.inf)
    kth_dists = np.partition(dists_xy, k - 1, axis=1)[:, k - 1]
    eps = kth_dists
    zero_mask = eps == 0
    if zero_mask.any():
        kth_dists2 = np.partition(dists_xy, k, axis=1)[:, k]
        eps = np.where(zero_mask, kth_dists2, eps)
    nx = np.sum(np.abs(x[:, None] - x[None, :]) < eps[:, None], axis=1) - 1
    ny = np.sum(np.abs(y[:, None] - y[None, :]) < eps[:, None], axis=1) - 1
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return float(mi)

def compute_mi_for_increments(dx_all, dy_all, lag_steps_list, max_samples=3000, k=5, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    dr = np.sqrt(dx_all**2 + dy_all**2)
    n_tracers, n_steps = dr.shape
    mi_results = {}
    for lag in lag_steps_list:
        if lag >= n_steps:
            mi_results[lag] = np.nan
            continue
        x_pairs = dr[:, :n_steps - lag].ravel()
        y_pairs = dr[:, lag:].ravel()
        valid = np.isfinite(x_pairs) & np.isfinite(y_pairs) & (x_pairs > 0) & (y_pairs > 0)
        x_v, y_v = x_pairs[valid], y_pairs[valid]
        if len(x_v) > max_samples:
            idx = rng.choice(len(x_v), size=max_samples, replace=False)
            x_v, y_v = x_v[idx], y_v[idx]
        if len(x_v) < 20:
            mi_results[lag] = np.nan
            continue
        mi_results[lag] = knn_mi_vectorized(np.log(x_v), np.log(y_v), k=k)
    return mi_results

def compute_kl_divergence(dx, dy, alpha_stable, beta_skew, scale, n_grid=500, max_samples=20000, rng_seed=42):
    if not np.isfinite(alpha_stable) or alpha_stable <= 0 or alpha_stable > 2:
        return np.nan
    rng = np.random.default_rng(rng_seed)
    combined = (dx + dy) / np.sqrt(2.0)
    combined = combined[np.isfinite(combined)]
    if len(combined) > max_samples:
        combined = rng.choice(combined, size=max_samples, replace=False)
    if len(combined) < 50:
        return np.nan
    p5, p95 = np.percentile(combined, [2, 98])
    grid = np.linspace(p5, p95, n_grid)
    try:
        kde = gaussian_kde(combined, bw_method='silverman')
        p_emp = np.clip(kde(grid), 1e-300, None)
        p_levy = np.clip(levy_stable.pdf(grid, alpha_stable, beta_skew, loc=0, scale=scale), 1e-300, None)
        p_emp /= np.trapz(p_emp, grid)
        p_levy /= np.trapz(p_levy, grid)
        return float(np.trapz(p_emp * np.log(p_emp / p_levy), grid))
    except Exception:
        return np.nan

if __name__ == '__main__':
    data_dir = "data/"
    inc_data = np.load(os.path.join(data_dir, "increment_results.npz"))
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    pv = np.load(pv_path, allow_pickle=False)
    n_steps_pv = 500
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    n_vortices_list = [5, 10, 20, 40]
    lag_steps_list = [1, 5, 10, 20]
    for nv in n_vortices_list:
        nv_key = str(int(nv))
        dx_lag1 = inc_data["pv_dx_" + nv_key + "_lag1"]
        dy_lag1 = inc_data["pv_dy_" + nv_key + "_lag1"]
        n_tr = int(np.sum(pv_group_vals == nv))
        mi_vals = compute_mi_for_increments(dx_lag1.reshape(n_tr, -1), dy_lag1.reshape(n_tr, -1), lag_steps_list)
        print("N=" + nv_key + " MI: " + str(mi_vals))
        alpha = float(inc_data["pv_alpha_stable_" + nv_key][0])
        kl = compute_kl_divergence(dx_lag1, dy_lag1, alpha, 0, 1.0)
        print("N=" + nv_key + " KL: " + str(kl))