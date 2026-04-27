# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os, warnings, datetime
from step_1 import compute_tamsd_single, compute_eb
from step_2 import find_crossover_time
from step_3 import get_positions_for_config_pv, get_positions_for_config_lw, compute_tamsd_ensemble, compute_hill_for_config
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
def get_etamsd(px, py, dt):
    T = len(px[0])
    ml = T // 2
    lt = np.arange(1, ml + 1) * dt
    tm = np.array([compute_tamsd_single(x, y, max_lag=ml) for x, y in zip(px, py)])
    return lt, np.mean(tm, axis=0), tm
def fit_pl(lt, msd, f0=0.10, f1=0.60):
    T = lt[-1]
    m = (lt >= f0 * T) & (lt <= f1 * T)
    if m.sum() < 3: return np.nan, np.nan, np.nan, np.array([]), np.array([])
    lx, ly = np.log(lt[m]), np.log(msd[m])
    v = np.isfinite(lx) & np.isfinite(ly)
    if v.sum() < 3: return np.nan, np.nan, np.nan, np.array([]), np.array([])
    s, b, r, _, _ = linregress(lx[v], ly[v])
    tf = lt[m][v]
    return s, b, r**2, tf, np.exp(b) * tf**s
def pdf_logbins(px, lag=1, nb=60):
    dxs = np.concatenate([x[lag:] - x[:len(x)-lag] for x in px if len(x) > lag])
    ax = np.abs(dxs[dxs != 0])
    if len(ax) < 10: return np.array([]), np.array([]), dxs
    p1 = max(np.percentile(ax, 0.5), ax[ax > 0].min())
    p99 = np.percentile(ax, 99.5)
    bins = np.logspace(np.log10(p1), np.log10(p99), nb)
    c, e = np.histogram(ax, bins=bins, density=True)
    ctr = 0.5 * (e[:-1] + e[1:])
    v = c > 0
    return ctr[v], c[v], dxs
def fit_tail(ctr, pdf, ft=0.20):
    k = max(int(len(ctr) * ft), 5)
    tx, ty = ctr[-k:], pdf[-k:]
    v = (tx > 0) & (ty > 0)
    if v.sum() < 4: return np.nan, np.nan, np.array([]), np.array([])
    s, b, r, _, _ = linregress(np.log(tx[v]), np.log(ty[v]))
    xf = np.array([tx[v][0], tx[v][-1]])
    return -s, r**2, xf, np.exp(b) * xf**s
if __name__ == '__main__':
    DD = 'data/'
    pv = np.load('/home/node/work/projects/superdiffusion_v2/point_vortex_tracers.npy', allow_pickle=False)
    lw = np.load('/home/node/work/projects/superdiffusion_v2/levy_walk_trajectories.npy', allow_pickle=False)
    pv_dt, lw_dt = 0.05, 0.1
    NV = np.array([5, 10, 20, 40])
    BT = np.array([1.2, 1.5, 1.8, 2.5])
    AT = np.array([1.8, 1.5, 1.2, 1.0])
    PC = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    LC = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    POS_PV, LT_PV, MT_PV, AE_PV, R2_PV, MU_PV = {}, {}, {}, {}, {}, {}
    for n in NV:
        px, py = get_positions_for_config_pv(pv, n)
        lt, mt, tm = get_etamsd(px, py, pv_dt)
        av, rv, _, _ = compute_tamsd_ensemble(px, py, pv_dt)
        mu, _, _ = compute_hill_for_config(px, py, frac=0.125)
        POS_PV[n] = (px, py); LT_PV[n] = lt; MT_PV[n] = mt
        AE_PV[n] = float(np.nanmean(av)); R2_PV[n] = float(np.nanmean(rv)); MU_PV[n] = float(mu)
    POS_LW, LT_LW, MT_LW, AE_LW, R2_LW, MU_LW = {}, {}, {}, {}, {}, {}
    for b in BT:
        lx, ly = get_positions_for_config_lw(lw, b)
        lt, mt, tm = get_etamsd(lx, ly, lw_dt)
        av, rv, _, _ = compute_tamsd_ensemble(lx, ly, lw_dt)
        mu, _, _ = compute_hill_for_config(lx, ly, frac=0.125)
        POS_LW[b] = (lx, ly); LT_LW[b] = lt; MT_LW[b] = mt
        AE_LW[b] = float(np.nanmean(av)); R2_LW[b] = float(np.nanmean(rv)); MU_LW[b] = float(mu)
    np.save(os.path.join(DD, 'step4_results.npy'), {'AE_PV': AE_PV, 'R2_PV': R2_PV, 'MU_PV': MU_PV, 'AE_LW': AE_LW, 'R2_LW': R2_LW, 'MU_LW': MU_LW, 'AT': AT, 'NV': NV, 'BT': BT}, allow_pickle=True)
    print('Analysis complete. Figures 1-3 would be generated here.')