# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy import stats

def reshape_trajectories(data, id_field, group_field, n_steps, sort_field='time'):
    unique_pairs = []
    seen = set()
    for row in data:
        key = (float(row[id_field]), float(row[group_field]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)
    n_tracers = len(unique_pairs)
    traj_ids = np.array([p[0] for p in unique_pairs])
    group_vals = np.array([p[1] for p in unique_pairs])
    x2d = np.zeros((n_tracers, n_steps), dtype=np.float64)
    y2d = np.zeros((n_tracers, n_steps), dtype=np.float64)
    for i, (tid, gv) in enumerate(unique_pairs):
        mask = (data[id_field] == tid) & (data[group_field] == gv)
        rows = data[mask]
        sort_idx = np.argsort(rows[sort_field])
        rows = rows[sort_idx]
        x2d[i] = rows['x_true']
        y2d[i] = rows['y_true']
    return traj_ids, group_vals, x2d, y2d

def compute_tamsd_ensemble(x2d, y2d, max_lag_frac=0.5):
    n_tracers, n_steps = x2d.shape
    max_lag = int(n_steps * max_lag_frac)
    tamsd = np.zeros((n_tracers, max_lag), dtype=np.float64)
    for delta in range(1, max_lag + 1):
        dx = x2d[:, delta:] - x2d[:, :n_steps - delta]
        dy = y2d[:, delta:] - y2d[:, :n_steps - delta]
        sd = dx ** 2 + dy ** 2
        tamsd[:, delta - 1] = sd.mean(axis=1)
    ensemble_msd = tamsd.mean(axis=0)
    lag_indices = np.arange(1, max_lag + 1)
    return tamsd, ensemble_msd, lag_indices

def fit_powerlaw(lag_times, msd_values, fit_frac_start=0.05, fit_frac_end=0.8):
    n = len(lag_times)
    i_start = max(1, int(n * fit_frac_start))
    i_end = max(i_start + 2, int(n * fit_frac_end))
    lt = lag_times[i_start:i_end]
    mv = msd_values[i_start:i_end]
    valid = (lt > 0) & (mv > 0)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    log_t = np.log(lt[valid])
    log_m = np.log(mv[valid])
    slope, intercept, r_value, p_value, se = stats.linregress(log_t, log_m)
    return slope, se, intercept

def compute_velocities(x2d, y2d, dt):
    vx = np.diff(x2d, axis=1) / dt
    vy = np.diff(y2d, axis=1) / dt
    return vx, vy

def compute_vacf_fft(vx, vy):
    n_tracers, n_steps = vx.shape
    n_fft = 2 * n_steps
    vacf_sum = np.zeros(n_steps, dtype=np.float64)
    norm_sum = 0.0
    for i in range(n_tracers):
        for v_comp in [vx[i], vy[i]]:
            v_mean = v_comp.mean()
            v_centered = v_comp - v_mean
            fft_v = np.fft.rfft(v_centered, n=n_fft)
            power = fft_v * np.conj(fft_v)
            acf_full = np.fft.irfft(power)[:n_steps].real
            acf_full /= (n_steps - np.arange(n_steps))
            vacf_sum += acf_full
            norm_sum += acf_full[0]
    vacf = vacf_sum / norm_sum
    return vacf

def fit_vacf_decay(vacf, lag_times, fit_start_frac=0.05, fit_end_frac=0.4):
    n = len(lag_times)
    i_start = max(1, int(n * fit_start_frac))
    i_end = max(i_start + 2, int(n * fit_end_frac))
    lt = lag_times[i_start:i_end]
    cv = vacf[i_start:i_end]
    valid = (lt > 0) & (cv > 0)
    if valid.sum() < 3:
        return np.nan, np.nan
    log_t = np.log(lt[valid])
    log_c = np.log(cv[valid])
    slope, intercept, r_value, p_value, se = stats.linregress(log_t, log_c)
    return -slope, se

if __name__ == '__main__':
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv = np.load(pv_path, allow_pickle=False)
    dt_pv = 0.05
    n_steps_pv = 500
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    n_vortices_list = sorted(np.unique(pv_group_vals).tolist())
    pv_alpha_group = {}
    pv_alpha_group_se = {}
    pv_beta_vacf = {}
    pv_beta_vacf_se = {}
    for nv in n_vortices_list:
        mask = pv_group_vals == nv
        x2d_g = pv_x2d[mask]
        y2d_g = pv_y2d[mask]
        tamsd, ens_msd, lag_idx = compute_tamsd_ensemble(x2d_g, y2d_g, max_lag_frac=0.5)
        lag_t = lag_idx * dt_pv
        a_grp, se_grp, _ = fit_powerlaw(lag_t, ens_msd)
        pv_alpha_group[nv] = a_grp
        pv_alpha_group_se[nv] = se_grp
        vx, vy = compute_velocities(x2d_g, y2d_g, dt_pv)
        vacf = compute_vacf_fft(vx, vy)
        vacf_lag_t = np.arange(len(vacf)) * dt_pv
        bv, bv_se = fit_vacf_decay(vacf, vacf_lag_t)
        pv_beta_vacf[nv] = bv
        pv_beta_vacf_se[nv] = bv_se
    print("Point-Vortex: Ensemble-averaged TAMSD power-law fits (MSD ~ t^alpha):")
    print("N_vortices  alpha_grp     se_grp      beta_vacf  alpha_check(2-beta)")
    for nv in n_vortices_list:
        a = pv_alpha_group[nv]
        se = pv_alpha_group_se[nv]
        bv = pv_beta_vacf[nv]
        alpha_check = 2.0 - bv if not np.isnan(bv) else np.nan
        print(str(int(nv)) + " " + str(round(a, 4)) + " " + str(round(se, 4)) + " " + str(round(bv, 4)) + " " + str(round(alpha_check, 4)))