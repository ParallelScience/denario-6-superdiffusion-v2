# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def load_and_validate_datasets(pv_path, lw_path):
    pv = np.load(pv_path, allow_pickle=False)
    lw = np.load(lw_path, allow_pickle=False)
    return pv, lw

def print_dataset_summary(pv, lw):
    print("=== POINT-VORTEX DATASET SUMMARY ===")
    print("Shape:", pv.shape)
    print("Unique n_vortices:", np.unique(pv['n_vortices']))
    print("Rows per n_vortices:", {int(n): int(np.sum(pv['n_vortices'] == n)) for n in np.unique(pv['n_vortices'])})
    print("Time range: [" + str(pv['time'].min()) + ", " + str(pv['time'].max()) + "] s")
    print()
    print("=== LEVY WALK DATASET SUMMARY ===")
    print("Shape:", lw.shape)
    print("Unique beta:", np.unique(lw['beta']))
    print("Rows per beta:", {float(b): int(np.sum(lw['beta'] == b)) for b in np.unique(lw['beta'])})
    print("Time range: [" + str(lw['time'].min()) + ", " + str(lw['time'].max()) + "] s")
    print()

def reshape_pv_trajectories(pv, n_steps=500):
    unique_pairs = []
    seen = set()
    for row in pv:
        key = (int(row['trajectory_id']), int(row['n_vortices']))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)
    n_tracers = len(unique_pairs)
    tracer_ids = np.array([p[0] for p in unique_pairs], dtype=np.int32)
    n_vortices_arr = np.array([p[1] for p in unique_pairs], dtype=np.int32)
    x_true_2d = np.zeros((n_tracers, n_steps), dtype=np.float64)
    y_true_2d = np.zeros((n_tracers, n_steps), dtype=np.float64)
    for i, (tid, nv) in enumerate(unique_pairs):
        mask = (pv['trajectory_id'] == tid) & (pv['n_vortices'] == nv)
        rows = pv[mask]
        sort_idx = np.argsort(rows['time'])
        rows = rows[sort_idx]
        x_true_2d[i] = rows['x_true']
        y_true_2d[i] = rows['y_true']
    return tracer_ids, n_vortices_arr, x_true_2d, y_true_2d

def compute_velocities_pv(x_true_2d, y_true_2d, dt=0.05):
    vx = np.gradient(x_true_2d, dt, axis=1)
    vy = np.gradient(y_true_2d, dt, axis=1)
    return vx, vy

def compute_okubo_weiss(vx, vy, dt=0.05):
    dvx_dt = np.gradient(vx, dt, axis=1)
    dvy_dt = np.gradient(vy, dt, axis=1)
    s1 = dvx_dt
    s2 = (dvy_dt + dvx_dt) / 2.0
    omega = (dvy_dt - dvx_dt) / 2.0
    W = s1 ** 2 + s2 ** 2 - omega ** 2
    return W

if __name__ == '__main__':
    data_dir = "data/"
    pv_path = '/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy'
    lw_path = '/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy'
    pv, lw = load_and_validate_datasets(pv_path, lw_path)
    print_dataset_summary(pv, lw)
    tracer_ids, n_vortices_arr, x_true_2d, y_true_2d = reshape_pv_trajectories(pv)
    vx, vy = compute_velocities_pv(x_true_2d, y_true_2d)
    W = compute_okubo_weiss(vx, vy)
    is_trapped = W < 0
    n_vortices_list = sorted(np.unique(n_vortices_arr).tolist())
    for nv in n_vortices_list:
        mask = n_vortices_arr == nv
        frac = is_trapped[mask].mean()
        mean_w = W[mask].mean()
        print("N=" + str(nv) + ": Trapped fraction=" + str(round(frac, 3)) + ", Mean W=" + str(round(mean_w, 5)))
    np.save(os.path.join(data_dir, "okubo_weiss_results.npy"), W)
    np.save(os.path.join(data_dir, "tracer_regimes.npy"), is_trapped)