# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy import stats
from step_2 import reshape_trajectories, compute_tamsd_ensemble, compute_velocities, compute_vacf_fft, fit_powerlaw

def compute_lyapunov_pairs(x2d, y2d, dt):
    n_tracers, n_steps = x2d.shape
    total_time = (n_steps - 1) * dt
    pair_lambdas = []
    for i in range(n_tracers):
        for j in range(i + 1, n_tracers):
            dx = x2d[i] - x2d[j]
            dy = y2d[i] - y2d[j]
            sep = np.sqrt(dx ** 2 + dy ** 2)
            sep_init = sep[0]
            sep_final = sep[-1]
            if sep_init > 0 and sep_final > 0:
                lam = np.log(sep_final / sep_init) / total_time
                pair_lambdas.append(lam)
    pair_lambdas = np.array(pair_lambdas)
    if len(pair_lambdas) == 0:
        return np.nan, np.nan, pair_lambdas
    return pair_lambdas.mean(), pair_lambdas.std(ddof=1) if len(pair_lambdas) > 1 else 0.0, pair_lambdas

def compute_local_lyapunov_ow(W_2d, dt):
    absW = np.abs(W_2d)
    dW = np.diff(absW, axis=1) / dt
    mean_absW = absW[:, :-1].mean(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        local_lam = np.where(mean_absW > 0, np.abs(dW).mean(axis=1) / mean_absW, np.nan)
    return local_lam

def compute_ensemble_msd_vacf(x2d, y2d, dt, max_lag_frac=0.5):
    _, ens_msd, lag_idx = compute_tamsd_ensemble(x2d, y2d, max_lag_frac=max_lag_frac)
    lag_times = lag_idx * dt
    vx, vy = compute_velocities(x2d, y2d, dt)
    vacf = compute_vacf_fft(vx, vy)
    return lag_times, ens_msd, vacf

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = "/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy"
    dt_pv = 0.05
    n_steps_pv = 500
    max_lag_frac = 0.5
    pv = np.load(pv_path, allow_pickle=False)
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    n_vortices_list = sorted(np.unique(pv_group_vals).tolist())
    n_tracers_total = pv_x2d.shape[0]
    W_flat = np.load(os.path.join(data_dir, "okubo_weiss_results.npy"))
    W_2d = W_flat.reshape(n_tracers_total, n_steps_pv)
    lyapunov_results = {}
    msd_normalized = {}
    vacf_arrays = {}
    local_lam_results = {}
    print("=== LYAPUNOV EXPONENT ESTIMATION (from tracer pair divergence) ===")
    print("NOTE: Only 5 tracers per group => at most 10 pairs per group (small sample)")
    print("")
    for nv in n_vortices_list:
        mask = pv_group_vals == nv
        x2d_g = pv_x2d[mask]
        y2d_g = pv_y2d[mask]
        W_2d_g = W_2d[mask]
        lam_mean, lam_std, pair_lams = compute_lyapunov_pairs(x2d_g, y2d_g, dt_pv)
        n_pairs = len(pair_lams)
        tau_int = 1.0 / lam_mean if not np.isnan(lam_mean) and lam_mean > 0 else np.nan
        lyapunov_results[nv] = {'lambda_mean': lam_mean, 'lambda_std': lam_std, 'tau_int': tau_int, 'n_pairs': n_pairs, 'pair_lambdas': pair_lams}
        local_lam = compute_local_lyapunov_ow(W_2d_g, dt_pv)
        local_lam_results[nv] = local_lam
        lag_times, ens_msd, vacf = compute_ensemble_msd_vacf(x2d_g, y2d_g, dt_pv, max_lag_frac)
        lag_times_norm = lag_times / tau_int if not np.isnan(tau_int) and tau_int > 0 else lag_times.copy()
        msd_normalized[nv] = {'lag_times': lag_times, 'lag_times_norm': lag_times_norm, 'ens_msd': ens_msd}
        vacf_arrays[nv] = {'vacf': vacf, 'lag_times': np.arange(len(vacf)) * dt_pv, 'lag_times_norm': np.arange(len(vacf)) * dt_pv / (tau_int if not np.isnan(tau_int) and tau_int > 0 else 1.0)}
        print("N=" + str(int(nv)) + ":")
        print("  n_pairs=" + str(n_pairs) + " | lambda_mean=" + str(round(lam_mean, 5)) + " s^-1 | lambda_std=" + str(round(lam_std, 5)) + " s^-1")
        print("  tau_int=1/lambda=" + str(round(tau_int, 4)) + " s")
        print("  Per-pair lambdas (s^-1): " + str([round(float(l), 5) for l in pair_lams]))
        print("  Local OW Lyapunov-like exponents (s^-1): mean=" + str(round(float(np.nanmean(local_lam)), 5)) + " | std=" + str(round(float(np.nanstd(local_lam)), 5)) + " | per-tracer=" + str([round(float(v), 5) for v in local_lam]))
        print("")
    print("=== NORMALIZED MSD SCALING COLLAPSE TEST ===")
    for nv in n_vortices_list:
        lt_norm = msd_normalized[nv]['lag_times_norm']
        ens_msd = msd_normalized[nv]['ens_msd']
        alpha, se, _ = fit_powerlaw(lt_norm, ens_msd)
        print("N=" + str(int(nv)) + " | alpha_norm=" + str(round(alpha, 4)) + " +/- " + str(round(se, 4)) + " | tau_int=" + str(round(lyapunov_results[nv]['tau_int'], 4)) + " s")
    print("")
    print("=== VACF NORMALIZATION CHECK ===")
    for nv in n_vortices_list:
        vacf = vacf_arrays[nv]['vacf']
        lt_norm = vacf_arrays[nv]['lag_times_norm']
        valid = (lt_norm > 0) & (vacf > 0)
        if valid.sum() >= 3:
            log_t = np.log(lt_norm[valid])
            log_v = np.log(vacf[valid])
            slope, intercept, r, p, se = stats.linregress(log_t[:int(len(log_t) * 0.4)], log_v[:int(len(log_t) * 0.4)])
            print("N=" + str(int(nv)) + " | VACF decay slope (norm time)=" + str(round(-slope, 4)) + " +/- " + str(round(se, 4)))
        else:
            print("N=" + str(int(nv)) + " | VACF: insufficient positive values for fit")
    save_dict = {}
    for nv in n_vortices_list:
        nv_key = str(int(nv))
        res = lyapunov_results[nv]
        save_dict['lambda_mean_' + nv_key] = np.array([res['lambda_mean']])
        save_dict['lambda_std_' + nv_key] = np.array([res['lambda_std']])
        save_dict['tau_int_' + nv_key] = np.array([res['tau_int']])
        save_dict['pair_lambdas_' + nv_key] = res['pair_lambdas']
        save_dict['local_lam_' + nv_key] = local_lam_results[nv]
        save_dict['lag_times_' + nv_key] = msd_normalized[nv]['lag_times']
        save_dict['lag_times_norm_' + nv_key] = msd_normalized[nv]['lag_times_norm']
        save_dict['ens_msd_' + nv_key] = msd_normalized[nv]['ens_msd']
        save_dict['vacf_' + nv_key] = vacf_arrays[nv]['vacf']
        save_dict['vacf_lag_times_' + nv_key] = vacf_arrays[nv]['lag_times']
        save_dict['vacf_lag_times_norm_' + nv_key] = vacf_arrays[nv]['lag_times_norm']
    np.savez(os.path.join(data_dir, "lyapunov_results.npz"), **save_dict)
    print("Lyapunov results saved to " + os.path.join(data_dir, "lyapunov_results.npz"))