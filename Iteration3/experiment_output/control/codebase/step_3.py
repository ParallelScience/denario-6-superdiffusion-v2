# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from step_2 import reshape_trajectories, compute_tamsd_ensemble

def compute_eb(tamsd_matrix):
    n_tracers = tamsd_matrix.shape[0]
    if n_tracers < 2:
        return np.full(tamsd_matrix.shape[1], np.nan)
    mean_tamsd = tamsd_matrix.mean(axis=0)
    var_tamsd = tamsd_matrix.var(axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        eb = np.where(mean_tamsd > 0, var_tamsd / (mean_tamsd ** 2), np.nan)
    return eb

def get_dominant_regime(is_trapped_2d):
    frac_trapped = is_trapped_2d.mean(axis=1)
    return frac_trapped >= 0.5

def print_eb_summary(label, eb_array, lag_times, short_idx, mid_idx, long_idx):
    def fmt(val):
        if np.isnan(val):
            return "NaN"
        return str(round(float(val), 5))
    print(label + " | EB(short, t=" + str(round(lag_times[short_idx], 2)) + "s)=" + fmt(eb_array[short_idx]) +
          " | EB(mid, t=" + str(round(lag_times[mid_idx], 2)) + "s)=" + fmt(eb_array[mid_idx]) +
          " | EB(long, t=" + str(round(lag_times[long_idx], 2)) + "s)=" + fmt(eb_array[long_idx]))

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = "/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy"
    lw_path = "/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy"
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    is_trapped_flat = np.load(os.path.join(data_dir, "tracer_regimes.npy"))
    dt_pv = 0.05
    n_steps_pv = 500
    max_lag_frac = 0.5
    pv_traj_ids, pv_group_vals, pv_x2d, pv_y2d = reshape_trajectories(pv, 'trajectory_id', 'n_vortices', n_steps_pv)
    n_vortices_list = sorted(np.unique(pv_group_vals).tolist())
    n_tracers_total = pv_x2d.shape[0]
    n_steps = pv_x2d.shape[1]
    is_trapped_2d = is_trapped_flat.reshape(n_tracers_total, n_steps)
    dominant_trapped = get_dominant_regime(is_trapped_2d)
    max_lag = int(n_steps * max_lag_frac)
    lag_indices = np.arange(1, max_lag + 1)
    lag_times_pv = lag_indices * dt_pv
    short_idx = max(0, int(max_lag * 0.05) - 1)
    mid_idx = max(0, int(max_lag * 0.25) - 1)
    long_idx = max_lag - 1
    eb_pv_all = {}
    eb_pv_trapped = {}
    eb_pv_chaotic = {}
    print("=== POINT-VORTEX ERGODICITY-BREAKING PARAMETER ===")
    for nv in n_vortices_list:
        mask_nv = pv_group_vals == nv
        x2d_g = pv_x2d[mask_nv]
        y2d_g = pv_y2d[mask_nv]
        dom_trap_g = dominant_trapped[mask_nv]
        tamsd_all, _, _ = compute_tamsd_ensemble(x2d_g, y2d_g, max_lag_frac=max_lag_frac)
        eb_all = compute_eb(tamsd_all)
        eb_pv_all[nv] = eb_all
        mask_trap = dom_trap_g
        mask_chao = ~dom_trap_g
        n_trap = mask_trap.sum()
        n_chao = mask_chao.sum()
        if n_trap >= 2:
            tamsd_trap, _, _ = compute_tamsd_ensemble(x2d_g[mask_trap], y2d_g[mask_trap], max_lag_frac=max_lag_frac)
            eb_trap = compute_eb(tamsd_trap)
        else:
            eb_trap = np.full(max_lag, np.nan)
        eb_pv_trapped[nv] = eb_trap
        if n_chao >= 2:
            tamsd_chao, _, _ = compute_tamsd_ensemble(x2d_g[mask_chao], y2d_g[mask_chao], max_lag_frac=max_lag_frac)
            eb_chao = compute_eb(tamsd_chao)
        else:
            eb_chao = np.full(max_lag, np.nan)
        eb_pv_chaotic[nv] = eb_chao
        print("N=" + str(int(nv)) + " | n_tracers=" + str(int(mask_nv.sum())) + " | n_trapped=" + str(int(n_trap)) + " | n_chaotic=" + str(int(n_chao)))
        print_eb_summary("  ALL    N=" + str(int(nv)), eb_all, lag_times_pv, short_idx, mid_idx, long_idx)
        print_eb_summary("  TRAPPED N=" + str(int(nv)), eb_trap, lag_times_pv, short_idx, mid_idx, long_idx)
        print_eb_summary("  CHAOTIC N=" + str(int(nv)), eb_chao, lag_times_pv, short_idx, mid_idx, long_idx)
    dt_lw = 0.1
    n_steps_lw = 600
    lw_traj_ids, lw_group_vals, lw_x2d, lw_y2d = reshape_trajectories(lw, 'trajectory_id', 'beta', n_steps_lw)
    beta_list = sorted(np.unique(lw_group_vals).tolist())
    max_lag_lw = int(n_steps_lw * max_lag_frac)
    lag_indices_lw = np.arange(1, max_lag_lw + 1)
    lag_times_lw = lag_indices_lw * dt_lw
    short_idx_lw = max(0, int(max_lag_lw * 0.05) - 1)
    mid_idx_lw = max(0, int(max_lag_lw * 0.25) - 1)
    long_idx_lw = max_lag_lw - 1
    eb_lw_all = {}
    print("=== LEVY WALK ERGODICITY-BREAKING PARAMETER ===")
    for beta in beta_list:
        mask_b = lw_group_vals == beta
        x2d_b = lw_x2d[mask_b]
        y2d_b = lw_y2d[mask_b]
        tamsd_b, _, _ = compute_tamsd_ensemble(x2d_b, y2d_b, max_lag_frac=max_lag_frac)
        eb_b = compute_eb(tamsd_b)
        eb_lw_all[beta] = eb_b
        alpha_th = round(3.0 - beta, 2) if beta < 2.0 else 1.0
        print("beta=" + str(round(beta, 2)) + " (alpha_theory=" + str(alpha_th) + ") | n_tracers=" + str(int(mask_b.sum())))
        print_eb_summary("  LW beta=" + str(round(beta, 2)), eb_b, lag_times_lw, short_idx_lw, mid_idx_lw, long_idx_lw)
    np.savez(os.path.join(data_dir, "eb_results.npz"), lag_times_pv=lag_times_pv, lag_times_lw=lag_times_lw, n_vortices_list=np.array(n_vortices_list), beta_list=np.array(beta_list), eb_pv_all_5=eb_pv_all[5], eb_pv_all_10=eb_pv_all[10], eb_pv_all_20=eb_pv_all[20], eb_pv_all_40=eb_pv_all[40], eb_pv_trapped_5=eb_pv_trapped[5], eb_pv_trapped_10=eb_pv_trapped[10], eb_pv_trapped_20=eb_pv_trapped[20], eb_pv_trapped_40=eb_pv_trapped[40], eb_pv_chaotic_5=eb_pv_chaotic[5], eb_pv_chaotic_10=eb_pv_chaotic[10], eb_pv_chaotic_20=eb_pv_chaotic[20], eb_pv_chaotic_40=eb_pv_chaotic[40], eb_lw_12=eb_lw_all[1.2], eb_lw_15=eb_lw_all[1.5], eb_lw_18=eb_lw_all[1.8], eb_lw_25=eb_lw_all[2.5])
    print("EB results saved to " + os.path.join(data_dir, "eb_results.npz"))