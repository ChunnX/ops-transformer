#!/usr/bin/env python3
"""Generate QAR/QRS performance analysis plots."""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PEAK = 56.0
OUTDIR = os.path.expanduser('~/')

# ── data ────────────────────────────────────────────────────
def load_data():
    d = {}

    # Server 8p QAR
    d['srv_qar'] = {
        'V': [0.031,0.063,0.125,0.219,0.406,0.711,0.766,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [15.50,16.00,17.81,23.17,34.20,41.24,47.11,90.39,171.65,247.08,306.94,448.70,582.76],
        'label': 'Server 8p', 'color': '#d62728', 'marker': 's'
    }
    # Server 8p QRS
    d['srv_qrs'] = {
        'V': [0.020,0.051,0.098,0.195,0.387,0.677,0.773,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [14.76,15.50,16.88,22.74,34.97,41.49,51.79,88.49,176.15,270.81,356.43,580.86,761.46],
        'label': 'Server 8p QRS', 'color': '#d62728', 'marker': 's', 'ls': '--'
    }
    # POD 8p QAR
    d['pod_qar'] = {
        'V': [0.031,0.063,0.125,0.219,0.406,0.711,0.766,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [12.69,13.29,15.33,20.87,26.43,39.34,40.74,80.06,143.42,200.65,259.78,357.86,470.55],
        'label': 'POD 8p', 'color': '#1f77b4', 'marker': 'o'
    }
    # POD 8p QRS
    d['pod_qrs'] = {
        'V': [0.020,0.051,0.098,0.195,0.387,0.677,0.773,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [12.33,13.27,14.82,20.73,27.53,41.60,43.76,83.53,147.36,205.06,284.83,478.31,638.52],
        'label': 'POD 8p QRS', 'color': '#1f77b4', 'marker': 'o', 'ls': '--'
    }
    # 标卡 4p QAR
    d['std_qar'] = {
        'V': [0.016,0.047,0.109,0.203,0.391,0.684,0.766,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [7.60,8.44,10.98,14.98,23.25,33.79,36.10,71.96,128.24,183.60,235.38,346.77,478.02],
        'label': 'Std 4p', 'color': '#ff7f0e', 'marker': '^'
    }
    # 标卡 4p QRS
    d['std_qrs'] = {
        'V': [0.016,0.051,0.098,0.195,0.387,0.677,0.773,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [7.63,8.76,10.84,15.23,23.85,35.99,38.20,73.94,129.45,185.53,241.91,380.82,519.79],
        'label': 'Std 4p QRS', 'color': '#ff7f0e', 'marker': '^', 'ls': '--'
    }
    # EVB 2p QAR
    d['evb_qar'] = {
        'V': [0.016,0.047,0.102,0.195,0.383,0.670,0.766,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [6.53,6.76,8.58,11.96,19.61,29.16,31.75,55.07,107.33,158.08,208.61,310.17,412.74],
        'label': 'EVB 2p', 'color': '#2ca02c', 'marker': 'D'
    }
    # EVB 2p QRS
    d['evb_qrs'] = {
        'V': [0.016,0.051,0.098,0.191,0.387,0.677,0.773,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [6.42,7.06,8.45,12.06,20.02,29.61,32.61,55.23,107.71,158.91,210.15,314.24,426.02],
        'label': 'EVB 2p QRS', 'color': '#2ca02c', 'marker': 'D', 'ls': '--'
    }
    # 4p MXFP QAR
    d['mx_qar'] = {
        'V': [0.016,0.047,0.109,0.203,0.391,0.684,0.766,1.5,3.0,4.5,6.0,9.0,12.0],
        'T': [7.80,8.57,11.17,15.14,23.50,34.12,36.80,72.04,131.43,186.58,236.02,347.57,448.08],
        'label': 'MXFP 4p', 'color': '#9467bd', 'marker': 'P'
    }
    return d

DATA = load_data()

# ── model lines ────────────────────────────────────────────
def model_qar(V, N):
    return (-3.0 + 2.9*N) + V * (35.6 + 0.4*N) / 1000

def model_qrs(V, N):
    t = (1.9 + 1.2*N) + V * (32.1 + 1.7*N) / 1000
    if N >= 8:
        t += 14.1 * np.maximum(0, np.array(V) - 6) / 1000
    return t

def model_srv_qar(V):
    return 18.4 + V * 47.7

def model_srv_qrs(V):
    v = np.array(V)
    t = np.where(v <= 6, 2.3 + 59.1*v, -41.3 + 67.5*v)
    return t

# ── style ──────────────────────────────────────────────────
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'font.family': 'sans-serif'})

v_fine = np.logspace(-2, 1.2, 200)

# ============================================================
# FIGURE 1: T_kernel vs V — QAR all envs
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
for key in ['evb_qar', 'std_qar', 'pod_qar', 'srv_qar', 'mx_qar']:
    d = DATA[key]
    ax.plot(d['V'], d['T'], marker=d['marker'], color=d['color'], ls='',
            label=d['label'], markersize=6, alpha=0.9)

# fits
for N, color, label in [(2, '#2ca02c', ''), (4, '#ff7f0e', ''), (8, '#1f77b4', '')]:
    ax.plot(v_fine, model_qar(v_fine, N), color=color, lw=1.2, alpha=0.5, ls='--')
ax.plot(v_fine, model_srv_qar(v_fine), color='#d62728', lw=1.5, ls='--', alpha=0.7, label='Server fit')
ax.plot(v_fine, model_qar(v_fine, 8), color='#1f77b4', lw=1.5, ls='--', alpha=0.7, label='POD/E VB/Std fits')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('V (MB)'); ax.set_ylabel('T_kernel (us)')
ax.set_title('QAR: T_kernel vs Payload')
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(OUTDIR + 'fig1_t_vs_v_qar.png')
plt.close(fig)

# ============================================================
# FIGURE 2: Util vs V — QAR all envs
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
for key in ['evb_qar', 'std_qar', 'pod_qar', 'srv_qar', 'mx_qar']:
    d = DATA[key]
    util = [v/t*1000/PEAK*100 for v, t in zip(d['V'], d['T'])]
    ax.plot(d['V'], util, marker=d['marker'], color=d['color'], ls='',
            label=d['label'], markersize=6, alpha=0.9)

# model Util curves
for N, color in [(2, '#2ca02c'), (4, '#ff7f0e'), (8, '#1f77b4')]:
    t_pred = model_qar(v_fine, N)
    util_pred = v_fine / t_pred * 1000 / PEAK * 100
    ax.plot(v_fine, util_pred, color=color, lw=1, alpha=0.4, ls='--')
t_pred = model_srv_qar(v_fine)
util_pred = v_fine / t_pred * 1000 / PEAK * 100
ax.plot(v_fine, util_pred, color='#d62728', lw=1.2, alpha=0.5, ls='--')

ax.axhline(100, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.text(0.03, 98, '56 GB/s ceiling', fontsize=8, color='gray', alpha=0.7)

ax.set_xscale('log')
ax.set_xlabel('V (MB)'); ax.set_ylabel('Util (%)')
ax.set_title('QAR: Util vs Payload')
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(OUTDIR + 'fig2_util_vs_v_qar.png')
plt.close(fig)

# ============================================================
# FIGURE 3: QAR vs QRS — Server & POD
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, env, qar_k, qrs_k, title in [
    (ax1, 'Server', 'srv_qar', 'srv_qrs', 'Server 8p'),
    (ax2, 'POD',   'pod_qar', 'pod_qrs', 'POD 8p'),
]:
    # QAR
    dq = DATA[qar_k]; uq = [v/t*1000/PEAK*100 for v,t in zip(dq['V'], dq['T'])]
    ax.plot(dq['V'], uq, marker=dq['marker'], color='#1f77b4', ls='-', lw=1.2,
            label='QAR', markersize=6, alpha=0.9)
    # QRS
    ds = DATA[qrs_k]; us = [v/t*1000/PEAK*100 for v,t in zip(ds['V'], ds['T'])]
    ax.plot(ds['V'], us, marker='^', color='#d62728', ls='--', lw=1.2,
            label='QRS', markersize=5, alpha=0.9)

    # model lines
    vv = np.logspace(-2, 1.2, 200)
    if env == 'Server':
        ax.plot(vv, vv/model_srv_qar(vv)*1000/PEAK*100, color='#1f77b4', lw=0.8, alpha=0.3)
        ax.plot(vv, vv/model_srv_qrs(vv)*1000/PEAK*100, color='#d62728', lw=0.8, alpha=0.3, ls='--')
    else:
        ax.plot(vv, vv/model_qar(vv,8)*1000/PEAK*100, color='#1f77b4', lw=0.8, alpha=0.3)
        ax.plot(vv, vv/model_qrs(vv,8)*1000/PEAK*100, color='#d62728', lw=0.8, alpha=0.3, ls='--')

    # all2all penalty zone
    if env == 'Server':
        ax.axvspan(6, 12, alpha=0.06, color='red')
        ax.text(7.5, 22, 'all2all\nzone', fontsize=7, color='#d62728', alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('V (MB)'); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax1.set_ylabel('Util (%)')
fig.suptitle('QAR vs QRS: Util Comparison', fontsize=14, y=1.01)
fig.savefig(OUTDIR + 'fig3_qar_vs_qrs.png')
plt.close(fig)

# ============================================================
# FIGURE 4: Model validation — T_predicted vs T_actual
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.5))

all_pred = []; all_act = []
for key, N, fn in [
    ('evb_qar', 2, model_qar), ('std_qar', 4, model_qar),
    ('pod_qar', 8, model_qar), ('srv_qar', 8, model_srv_qar),
    ('evb_qrs', 2, model_qrs), ('std_qrs', 4, model_qrs),
    ('pod_qrs', 8, model_qrs), ('srv_qrs', 8, model_srv_qrs),
]:
    d = DATA[key]
    V = np.array(d['V'])
    if fn == model_srv_qar:
        tp = model_srv_qar(V)
    elif fn == model_srv_qrs:
        tp = model_srv_qrs(V)
    elif fn == model_qar:
        tp = model_qar(V, N)
    else:
        tp = model_qrs(V, N)
    c = d['color']; m = d['marker']
    ax.plot(tp, d['T'], marker=m, color=c, ls='', label=d['label'], markersize=5, alpha=0.85)
    all_pred.extend(tp); all_act.extend(d['T'])

lims = [0, max(max(all_pred), max(all_act)) * 1.05]
ax.plot(lims, lims, 'k-', lw=0.8, alpha=0.3, label='y=x')
ax.plot(lims, [x*1.10 for x in lims], 'gray', lw=0.6, ls=':', alpha=0.4, label='+10%')
ax.plot(lims, [x*0.90 for x in lims], 'gray', lw=0.6, ls=':', alpha=0.4, label='-10%')
ax.fill_between(lims, [x*0.90 for x in lims], [x*1.10 for x in lims],
                alpha=0.04, color='green')

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('T_predicted (us)'); ax.set_ylabel('T_actual (us)')
ax.set_title('Model Validation: Predicted vs Actual')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)
ax.set_aspect('equal')
fig.savefig(OUTDIR + 'fig4_model_validation.png')
plt.close(fig)

print(f'4 figures saved to {OUTDIR}')
for f in ['fig1_t_vs_v_qar.png', 'fig2_util_vs_v_qar.png', 'fig3_qar_vs_qrs.png', 'fig4_model_validation.png']:
    print(f'  {OUTDIR}{f}')
