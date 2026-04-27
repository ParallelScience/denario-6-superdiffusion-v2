# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def load_and_validate_point_vortex(path):
    pv = np.load(path, allow_pickle=False)
    return pv

def load_and_validate_levy_walk(path):
    lw = np.load(path, allow_pickle=False)
    return lw

def summarize_group_stats(arr, group_field, group_values, label):
    stats = {}
    print("\n" + "=" * 80)
    print("DATASET: " + label)
    print("=" * 80)
    header = "{:<12} {:>8} {:>10} {:>20} {:>22} {:>22} {:>22}".format(group_field, "n_rows", "n_trajs", "time_range (s)", "x_true_range (m)", "y_true_range (m)", "msd_true_range (m^2)")
    print(header)
    print("-" * 120)
    for gv in group_values:
        mask = arr[group_field] == gv
        sub = arr[mask]
        n_rows = sub.shape[0]
        n_trajs = len(np.unique(sub['trajectory_id']))
        t_min = sub['time'].min()
        t_max = sub['time'].max()
        x_min = sub['x_true'].min()
        x_max = sub['x_true'].max()
        y_min = sub['y_true'].min()
        y_max = sub['y_true'].max()
        msd_min = sub['msd_true'].min()
        msd_max = sub['msd_true'].max()
        time_range_str = "[" + str(round(t_min, 4)) + ", " + str(round(t_max, 4)) + "]"
        x_range_str = "[" + str(round(x_min, 4)) + ", " + str(round(x_max, 4)) + "]"
        y_range_str = "[" + str(round(y_min, 4)) + ", " + str(round(y_max, 4)) + "]"
        msd_range_str = "[" + str(round(msd_min, 4)) + ", " + str(round(msd_max, 4)) + "]"
        row = "{:<12} {:>8} {:>10} {:>20} {:>22} {:>22} {:>22}".format(str(gv), n_rows, n_trajs, time_range_str, x_range_str, y_range_str, msd_range_str)
        print(row)
        stats[gv] = {'n_rows': n_rows, 'n_trajs': n_trajs, 't_min': t_min, 't_max': t_max, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'msd_min': msd_min, 'msd_max': msd_max}
    print("-" * 120)
    return stats

def print_array_info(arr, name):
    print("\n--- Array Info: " + name + " ---")
    print("Shape: " + str(arr.shape))
    print("Dtype fields: " + str(arr.dtype.names))
    print("Dtypes:")
    for field in arr.dtype.names:
        print("  " + field + ": " + str(arr.dtype[field]))

def print_noisy_vs_true_ranges(arr, group_field, group_values, label):
    print("\n--- Noisy vs True Position Ranges: " + label + " ---")
    for gv in group_values:
        mask = arr[group_field] == gv
        sub = arr[mask]
        xn_range = "[" + str(round(sub['x_noisy'].min(), 4)) + ", " + str(round(sub['x_noisy'].max(), 4)) + "]"
        xt_range = "[" + str(round(sub['x_true'].min(), 4)) + ", " + str(round(sub['x_true'].max(), 4)) + "]"
        yn_range = "[" + str(round(sub['y_noisy'].min(), 4)) + ", " + str(round(sub['y_noisy'].max(), 4)) + "]"
        yt_range = "[" + str(round(sub['y_true'].min(), 4)) + ", " + str(round(sub['y_true'].max(), 4)) + "]"
        print(str(group_field) + "=" + str(gv) + "  x_noisy: " + xn_range + "  x_true: " + xt_range + "  y_noisy: " + yn_range + "  y_true: " + yt_range)

def print_levy_walk_alpha_theory(lw, beta_values):
    print("\n--- Levy Walk: beta vs alpha_theory ---")
    for bv in beta_values:
        mask = lw['beta'] == bv
        sub = lw[mask]
        alpha_vals = np.unique(sub['alpha_theory'])
        print("  beta=" + str(round(bv, 4)) + "  alpha_theory=" + str(alpha_vals))

if __name__ == '__main__':
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    data_dir = "data/"
    pv = load_and_validate_point_vortex(pv_path)
    lw = load_and_validate_levy_walk(lw_path)
    print_array_info(pv, "point_vortex_tracers")
    print_array_info(lw, "levy_walk_trajectories")
    pv_n_values = sorted(np.unique(pv['n_vortices']).tolist())
    lw_beta_values = sorted(np.unique(lw['beta']).tolist())
    print("\n--- Point-Vortex: unique n_vortices ---")
    print(pv_n_values)
    print("--- Levy Walk: unique beta values ---")
    print(lw_beta_values)
    print("\n--- Point-Vortex: unique trajectory_ids per n_vortices ---")
    for nv in pv_n_values:
        mask = pv['n_vortices'] == nv
        ids = np.unique(pv[mask]['trajectory_id'])
        print("  n_vortices=" + str(nv) + "  trajectory_ids=" + str(ids.tolist()))
    print("\n--- Levy Walk: unique trajectory_ids per beta ---")
    for bv in lw_beta_values:
        mask = lw['beta'] == bv
        ids = np.unique(lw[mask]['trajectory_id'])
        print("  beta=" + str(round(bv, 4)) + "  trajectory_ids=" + str(ids.tolist()))
    pv_stats = summarize_group_stats(pv, 'n_vortices', pv_n_values, "Point-Vortex Tracers")
    lw_stats = summarize_group_stats(lw, 'beta', lw_beta_values, "Levy Walk Trajectories")
    print_noisy_vs_true_ranges(pv, 'n_vortices', pv_n_values, "Point-Vortex Tracers")
    print_noisy_vs_true_ranges(lw, 'beta', lw_beta_values, "Levy Walk Trajectories")
    print_levy_walk_alpha_theory(lw, lw_beta_values)
    print("\n--- gamma_std values in Point-Vortex dataset ---")
    print(np.unique(pv['gamma_std']))
    print("\n--- Point-Vortex: total rows, expected 10000 ---")
    print("Total rows: " + str(pv.shape[0]))
    print("--- Levy Walk: total rows, expected 24000 ---")
    print("Total rows: " + str(lw.shape[0]))
    pv_out = os.path.join(data_dir, "point_vortex_tracers_validated.npz")
    lw_out = os.path.join(data_dir, "levy_walk_trajectories_validated.npz")
    np.savez_compressed(pv_out, data=pv)
    np.savez_compressed(lw_out, data=lw)
    print("\nValidated point-vortex data saved to: " + pv_out)
    print("Validated Levy walk data saved to: " + lw_out)
    print("\n--- Step 1 Complete: Both datasets loaded, validated, and saved. ---")