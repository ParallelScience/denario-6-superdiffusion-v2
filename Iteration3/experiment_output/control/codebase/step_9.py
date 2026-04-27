# filename: codebase/step_9.py
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
from step_2 import reshape_trajectories, compute_tamsd_ensemble, fit_powerlaw
from step_5 import compute_increments_at_lag, fit_ccdf_tail

def compute_group_alpha_msd(x2d, y2d, dt, max_lag_frac=0.5):
    _, ens_msd, lag_idx = compute_tamsd_ensemble(x2d, y2d, max_lag_frac=max_lag_frac)
    lag_times = lag_idx * dt
    alpha, se, _ = fit_powerlaw(lag_times, ens_msd)
    return alpha, se

def compute_group_mu_tail(x2d, y2d, lag_steps=1, top_frac=0.10):
    dx, dy = compute_increments_at_lag(x2d, y2d, lag_steps)
    mu, mu_se, _, _ = fit_ccdf_tail(dx, dy, top_frac=top_frac)
    return mu, mu_se

def compute_mu_eff_theory(N, gamma_std=1.0, c_correction=1.5):
    mu_eff = 2.0 - c_correction * gamma_std / np.sqrt(N)
    mu_eff = np.clip(mu_eff, 0.5, 2.0)
    alpha_theory_corrected = 3.0 - mu_eff
    return mu_eff, alpha_theory_corrected

def composite_distance(alpha_pv, mu_pv, alpha_lw, mu_lw, w_alpha=0.6, w_mu=0.4):
    d_alpha = abs(alpha_pv - alpha_lw)
    d_mu = abs(mu_pv - mu_lw)
    return w_alpha * d_alpha + w_mu * d_mu

if __name__ == '__main__':
    data_dir = "data/"
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
    alpha_theory_lw = {1.2: 1.8, 1.5: 1.5, 1.8: 1.2, 2.5: 1.0}
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    lw_traj_ids, lw_group_vals, lw_x2d, lw_y2d = reshape_trajectories(lw, 'trajectory_id', 'beta', n_steps_lw)
    bayesian_alpha_pv = {5: {'map': 1.646, 'ci_low': 1.565, 'ci_high': 1.726}, 10: {'map': 1.796, 'ci_low': 1.706, 'ci_high': 1.887}, 20: {'map': 1.827, 'ci_low': 1.706, 'ci_high': 1.947}, 40: {'map': 1.867, 'ci_low': 1.746, 'ci_high': 1.987}}
    inc_data = np.load(os.path.join(data_dir, "increment_results.npz"))
    pv_alpha_msd, pv_alpha_msd_se, pv_mu, pv_mu_se = {}, {}, {}, {}
    for nv in n_vortices_list:
        mask = pv_group_vals == nv
        alpha_val, alpha_se = compute_group_alpha_msd(pv_x2d[mask], pv_y2d[mask], dt_pv)
        pv_alpha_msd[nv] = alpha_val
        pv_alpha_msd_se[nv] = alpha_se
        nv_key = str(int(nv))
        pv_mu[nv] = float(inc_data["pv_mu_" + nv_key + "_lag1"][0])
        pv_mu_se[nv] = 0.0
    lw_alpha_msd, lw_alpha_msd_se, lw_mu, lw_mu_se = {}, {}, {}, {}
    for beta in beta_list:
        mask = lw_group_vals == beta
        alpha_val, alpha_se = compute_group_alpha_msd(lw_x2d[mask], lw_y2d[mask], dt_lw)
        lw_alpha_msd[beta] = alpha_val
        lw_alpha_msd_se[beta] = alpha_se
        beta_key = str(beta).replace('.', 'p')
        lw_mu[beta] = float(inc_data["lw_mu_" + beta_key + "_lag1"][0])
        lw_mu_se[beta] = 0.0
    pv_alpha_stable, lw_alpha_stable = {}, {}
    for nv in n_vortices_list:
        nv_key = str(int(nv))
        pv_alpha_stable[nv] = float(inc_data["pv_alpha_stable_" + nv_key][0])
    for beta in beta_list:
        beta_key = str(beta).replace('.', 'p')
        lw_alpha_stable[beta] = float(inc_data["lw_alpha_stable_" + beta_key][0])
    mu_eff_theory, alpha_theory_corrected = {}, {}
    for nv in n_vortices_list:
        mu_eff, alpha_corr = compute_mu_eff_theory(nv)
        mu_eff_theory[nv] = mu_eff
        alpha_theory_corrected[nv] = alpha_corr
    mapping_table = []
    for i, nv in enumerate(n_vortices_list):
        alpha_pv = bayesian_alpha_pv[nv]['map']
        mu_pv_val = pv_mu[nv]
        best_beta, best_dist = None, np.inf
        for j, beta in enumerate(beta_list):
            dist = composite_distance(alpha_pv, mu_pv_val, lw_alpha_msd[beta], lw_mu[beta])
            if dist < best_dist:
                best_dist = dist
                best_beta = beta
        mapping_table.append({'n_vortices': nv, 'best_match_beta': best_beta, 'alpha_pv_bayesian': alpha_pv, 'alpha_pv_ci_low': bayesian_alpha_pv[nv]['ci_low'], 'alpha_pv_ci_high': bayesian_alpha_pv[nv]['ci_high'], 'alpha_pv_msd': pv_alpha_msd[nv], 'alpha_lw_msd': lw_alpha_msd[best_beta], 'alpha_lw_theory': alpha_theory_lw[best_beta], 'mu_pv': mu_pv_val, 'mu_lw': lw_mu[best_beta], 'alpha_theory_corrected': alpha_theory_corrected[nv], 'mu_eff_theory': mu_eff_theory[nv], 'composite_distance': best_dist, 'alpha_stable_pv': pv_alpha_stable[nv], 'alpha_stable_lw': lw_alpha_stable[best_beta]})
    print("=" * 100)
    print("EFFECTIVE THEORY MAPPING TABLE")
    print("=" * 100)
    for row in mapping_table:
        print(row)