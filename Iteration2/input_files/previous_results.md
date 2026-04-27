**Code Explanation:**

This script generates and saves all 5 figures. The plotting code is split across two separate Python files to avoid truncation: this file (step_4.py) handles data loading, computation, printing, and Figures 1-3, while step_4b.py handles Figures 4-5. Both files are complete and self-contained.

**Python Code:**

```python
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
    """Ensemble TAMSD. Returns lag_times (s), mean_tamsd (m^2), tamsd_matrix (m^2)."""
    T = len(px[0])
    ml = T // 2
    lt = np.arange(1, ml + 1) * dt
    tm = np.array([compute_tamsd_single(x, y, max_lag=ml) for x, y in zip(px, py)])
    return lt, np.mean(tm, axis=0), tm


def fit_pl(lt, msd, f0=0.10, f1=0.60):
    """Power-law fit to TAMSD. Returns alpha, intercept, r2, t_fit (s), msd_fit (m^2)."""
    T = lt[-1]
    m = (lt >= f0 * T) & (lt <= f1 * T)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    lx, ly = np.log(lt[m]), np.log(msd[m])
    v = np.isfinite(lx) & np.isfinite(ly)
    if v.sum() < 3:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    s, b, r, _, _ = linregress(lx[v], ly[v])
    tf = lt[m][v]
    return s, b, r**2, tf, np.exp(b) * tf**s


def pdf_logbins(px, lag=1, nb=60):
    """Log-binned PDF of |dx|. Returns centers (m), pdf (1/m), dx_signed (m)."""
    dxs = np.concatenate([x[lag:] - x[:len(x)-lag] for x in px if len(x) > lag])
    ax = np.abs(dxs[dxs != 0])
    if len(ax) < 10:
        return np.array([]), np.array([]), dxs
    p1 = max(np.percentile(ax, 0.5), ax[ax > 0].min())
    p99 = np.percentile(ax, 99.5)
    bins = np.logspace(np.log10(p1), np.log10(p99), nb)
    c, e = np.histogram(ax, bins=bins, density=True)
    ctr = 0.5 * (e[:-1] + e[1:])
    v = c > 0
    return ctr[v], c[v], dxs


def fit_tail(ctr, pdf, ft=0.20):
    """Power-law tail fit. Returns mu (dimensionless), r2, x_fit (m), y_fit (1/m)."""
    k = max(int(len(ctr) * ft), 5)
    tx, ty = ctr[-k:], pdf[-k:]
    v = (tx > 0) & (ty > 0)
    if v.sum() < 4:
        return np.nan, np.nan, np.array([]), np.array([])
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

    POS_PV, LT_PV, MT_PV, TM_PV, AE_PV, R2_PV, MU_PV = {}, {}, {}, {}, {}, {}, {}
    for n in NV:
        px, py = get_positions_for_config_pv(pv, n)
        lt, mt, tm = get_etamsd(px, py, pv_dt)
        av, rv, _, _ = compute_tamsd_ensemble(px, py, pv_dt)
        mu, _, _ = compute_hill_for_config(px, py, frac=0.125)
        POS_PV[n] = (px, py); LT_PV[n] = lt; MT_PV[n] = mt; TM_PV[n] = tm
        AE_PV[n] = float(np.nanmean(av)); R2_PV[n] = float(np.nanmean(rv)); MU_PV[n] = float(mu)

    POS_LW, LT_LW, MT_LW, TM_LW, AE_LW, R2_LW, MU_LW = {}, {}, {}, {}, {}, {}, {}
    for b in BT:
        lx, ly = get_positions_for_config_lw(lw, b)
        lt, mt, tm = get_etamsd(lx, ly, lw_dt)
        av, rv, _, _ = compute_tamsd_ensemble(lx, ly, lw_dt)
        mu, _, _ = compute_hill_for_config(lx, ly, frac=0.125)
        POS_LW[b] = (lx, ly); LT_LW[b] = lt; MT_LW[b] = mt; TM_LW[b] = tm
        AE_LW[b] = float(np.nanmean(av)); R2_LW[b] = float(np.nanmean(rv)); MU_LW[b] = float(mu)

    lag_step = max(1, 500 // 80)
    lag_idx = np.arange(1, 250, lag_step)
    CF_A, CF_D, CF_CL, CF_CH, CF_TC, CF_LT = {}, {}, {}, {}, {}, {}
    for n in NV:
        d = np.load(os.path.join(DD, 'pv_results_n' + str(n) + '.npz'))
        CF_A[n] = d['alpha']; CF_D[n] = d['D_alpha']
        CF_CL[n] = d['ci_low']; CF_CH[n] = d['ci_high']
        CF_TC[n] = d['tau_c_boot']; CF_LT[n] = lag_idx * pv_dt

    print("=== Key Results ===")
    print("\n--- PV TAMSD ---")
    for n in NV:
        print("  N=" + str(n) + " alpha=" + str(round(AE_PV[n], 4)) +
              " R2=" + str(round(R2_PV[n], 4)) + " mu=" + str(round(MU_PV[n], 4)))
    print("\n--- LW TAMSD ---")
    for i, b in enumerate(BT):
        print("  beta=" + str(b) + " ath=" + str(AT[i]) +
              " alpha=" + str(round(AE_LW[b], 4)) +
              " R2=" + str(round(R2_LW[b], 4)) + " mu=" + str(round(MU_LW[b], 4)))
    print("\n--- CF Crossover Times ---")
    TAU_C = {}
    for n in NV:
        tc = find_crossover_time(CF_A[n], CF_LT[n])
        TAU_C[n] = tc
        bv = CF_TC[n][np.isfinite(CF_TC[n])]
        lo = float(np.percentile(bv, 2.5)) if len(bv) > 0 else np.nan
        hi = float(np.percentile(bv, 97.5)) if len(bv) > 0 else np.nan
        fv = CF_A[n][np.isfinite(CF_A[n])]
        aa = float(np.nanmedian(fv[-5:])) if len(fv) >= 5 else np.nan
        print("  N=" + str(n) + " tau_c=" + str(round(tc, 3) if np.isfinite(tc) else 'nan') +
              " 95CI=[" + str(round(lo, 3) if np.isfinite(lo) else 'nan') + "," +
              str(round(hi, 3) if np.isfinite(hi) else 'nan') + "]" +
              " alpha_asymp=" + str(round(aa, 4) if np.isfinite(aa) else 'nan'))
    print("\n--- Effective Theory Mapping ---")
    MAPPING = {}
    for n in NV:
        best_b, best_d = None, float('inf')
        for b in BT:
            dist = np.sqrt((AE_PV[n] - AE_LW[b])**2 + (MU_PV[n] - MU_LW[b])**2)
            if dist < best_d:
                best_d = dist; best_b = b
        MAPPING[n] = best_b
        ath = 3.0 - best_b if best_b < 2 else 1.0
        apred = 3.0 - MU_PV[n] if MU_PV[n] < 2 else 1.0
        print("  N=" + str(n) + " best_beta=" + str(best_b) +
              " ath=" + str(round(ath, 2)) + " dist=" + str(round(best_d, 4)) +
              " alpha_theory_pred=" + str(round(apred, 4)))

    np.save(os.path.join(DD, 'step4_results.npy'), {
        'AE_PV': AE_PV, 'R2_PV': R2_PV, 'MU_PV': MU_PV,
        'AE_LW': AE_LW, 'R2_LW': R2_LW, 'MU_LW': MU_LW,
        'TAU_C': TAU_C, 'MAPPING': MAPPING,
        'AT': AT, 'NV': NV, 'BT': BT
    }, allow_pickle=True)

    print("\n=== Figure 1: TAMSD Comparison ===")
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, configs, LT, MT, colors, lp, title in [
        (axes[0], NV, LT_PV, MT_PV, PC, 'N=', 'Point-Vortex: Ensemble TAMSD'),
        (axes[1], BT, LT_LW, MT_LW, LC, 'b=', 'Levy Walk: Ensemble TAMSD')
    ]:
        for i, cfg in enumerate(configs):
            lt, mt = LT[cfg], MT[cfg]
            v = mt > 0
            ax.loglog(lt[v], mt[v], color=colors[i], lw=1.8, label=lp + str(cfg))
            al, _, r2, tf, mf = fit_pl(lt, mt)
            if len(tf) > 0 and np.isfinite(al):
                ax.loglog(tf, mf, '--', color=colors[i], lw=1.2)
                mid = len(tf) // 2
                ax.annotate('a=' + str(round(al, 2)) + ' R2=' + str(round(r2, 2)),
                            xy=(tf[mid], mf[mid]), fontsize=6.5, color=colors[i],
                            xytext=(5, 3), textcoords='offset points')
        ax.set_xlabel('Lag time (s)')
        ax.set_ylabel('TAMSD (m^2)')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(DD, 'tamsd_comparison_1_' + ts + '.png')
    fig1.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Figure 1 saved to " + p1)

    print("=== Figure 2: Displacement PDFs ===")
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    for ax, configs, POS, colors, lp, title in [
        (axes2[0], NV, POS_PV, PC, 'N=', 'Point-Vortex: Displacement PDF'),
        (axes2[1], BT, POS_LW, LC, 'b=', 'Levy Walk: Displacement PDF')
    ]:
        inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
        for i, cfg in enumerate(configs):
            px, py = POS[cfg]
            ctr, pdf, dxs = pdf_logbins(px, lag=1, nb=60)
            if len(ctr) == 0:
                continue
            ax.loglog(ctr, pdf, color=colors[i], lw=1.5, alpha=0.85, label=lp + str(cfg))
            mu, r2, xf, yf = fit_tail(ctr, pdf, ft=0.20)
            if len(xf) > 0 and np.isfinite(mu):
                ax.loglog(xf, yf, '--', color=colors[i], lw=1.2)
                ax.annotate('mu=' + str(round(mu, 2)), xy=(xf[-1], yf[-1]),
                            fontsize=7, color=colors[i], xytext=(-35