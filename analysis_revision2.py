#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_revision2.py
====================================================================
Revision-1 analyses for the Safety Science major revision of:
  "The Safety Culture-Injury Paradox Explained by Occupational
   Confounding" (Safety Science, decision: major revision,
   resubmission due 2026-08-14).

Implements the NEW analyses requested by the reviewers:

  Reviewer 1, comment 4 (top-coded firm-size category):
    A. Population-weighted share of each size category (wt1) --
       quantifies how much of the establishment population the
       open-ended >=100 category represents.
    B. S9: primary model with the open-ended top size category
       mapped to 1,000 and 2,000 workers (extends S6-S8).
    C. S10: within the >=100-worker stratum, effect-measure
       modification of the safety-culture-injury association by
       two observable markers of very large scale:
         (i)  electrical contract capacity >= 1,000 kW (elec_cap == 6)
         (ii) presence of a dedicated in-house safety department
              (saf_dept_yn == 1)
       Interaction tests + marker-stratum-specific IRRs.
    D. S11: primary model excluding the >=100 category entirely
       (1-99 workers only, where size measurement is not top-coded).

  Reviewer 1, comment 5 (within-2024 temporal overlap / reactive
  safety-posture upgrading):
    E. Reactive-upgrading diagnostic: OLS of the standardized overall
       safety-culture score on injury-history group (never injured
       2022-2024 / prior-only 2022-23 / new-2024-only / persistent),
       adjusting for industry fixed effects and size category.
       If an early-2024 accident inflated late-2024 culture reports,
       the new-2024-only group should score HIGHER than never-injured
       establishments conditional on structure.

Input : z_revision_1/input/{analytic_sample.csv, Dataset1.csv} when present,
        else the canonical project paths (output/pre_output/, Data_kosha/)
Output: paper/Safety Science/z_revision_1/output/
        (+ run_revision2_<timestamp> under output/analysis_output/)

Run:
  /Users/y3korea/miniforge3/bin/python analysis_revision2.py
====================================================================
"""
import os, sys, json, hashlib, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ------------------------------------------------------------------
# 0. Paths (portable: Colab or local)
# ------------------------------------------------------------------
try:
    import google.colab  # noqa
    if not os.path.isdir('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')
except Exception:
    pass
CANDIDATES = [
    '/content/drive/MyDrive/완석_구글자료/연구자료/20260313_kosha',
    '/Users/y3korea/Library/CloudStorage/GoogleDrive-y3korea@gmail.com/내 드라이브/완석_구글자료/연구자료/20260313_kosha',
]
BASE = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if BASE:
    CODE = os.path.join(BASE, 'Code_kosha', '2_code')
    REV = os.path.join(CODE, 'paper', 'Safety Science', 'z_revision_1')
else:
    # Standalone / cloned-repo mode: run from any working directory that
    # holds analytic_sample.csv and Dataset1.csv (directly or in ./input/).
    # Outputs are then written to ./revision1_outputs/.
    BASE = CODE = os.getcwd()
    REV = os.path.join(os.getcwd(), 'revision1_outputs')

def _first_existing(*paths):
    return next((p for p in paths if os.path.isfile(p)), paths[-1])

# Prefer the self-contained revision folder (z_revision_1/input), then the
# current working directory, then the canonical project locations.
INPUT = _first_existing(
    os.path.join(REV, 'input', 'analytic_sample.csv'),
    os.path.join(os.getcwd(), 'input', 'analytic_sample.csv'),
    os.path.join(os.getcwd(), 'analytic_sample.csv'),
    os.path.join(CODE, 'output', 'pre_output', 'analytic_sample.csv'))
RAW1 = _first_existing(
    os.path.join(REV, 'input', 'Dataset1.csv'),
    os.path.join(os.getcwd(), 'input', 'Dataset1.csv'),
    os.path.join(os.getcwd(), 'Dataset1.csv'),
    os.path.join(BASE, 'Data_kosha', '작업환경 실태조사', '7차_작업환경 실태조사',
                 '[CSV] 제7차 작업환경실태조사 데이터(CSV)', 'Dataset1.csv'))
TS = datetime.now().strftime('%Y%m%d_%H%M')
OUT_RUN = os.path.join(CODE, 'output', 'analysis_output', f'run_revision2_{TS}')
OUT_REV = os.path.join(REV, 'output')
for d in (OUT_RUN, OUT_REV):
    os.makedirs(d, exist_ok=True)

def banner(msg):
    print('\n' + '=' * 70 + f'\n{msg}\n' + '=' * 70)

def save_csv(dfx, name):
    for d in (OUT_RUN, OUT_REV):
        dfx.to_csv(os.path.join(d, name), index=False)

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
banner('1. LOAD')
df = pd.read_csv(INPUT, encoding='utf-8-sig')
sha = hashlib.sha256(open(INPUT, 'rb').read()).hexdigest()[:16]
print(f'analytic_sample.csv  N = {len(df):,}  SHA-256[:16] = {sha}')
raw = pd.read_csv(RAW1, encoding='utf-8-sig', low_memory=False)
raw = raw.rename(columns={raw.columns[0]: 'id'})
df = df.rename(columns={df.columns[0]: 'id'})
mg = raw[['id', 'elec_cap', 'ptnr_yn', 'saf_dept_yn']].copy()
mg = mg.rename(columns={'saf_dept_yn': 'saf_dept_yn_raw'})
df = df.merge(mg, on='id', how='left', validate='1:1')
assert len(df) == 20262, f'merge changed N: {len(df)}'
print(f'merged elec_cap/ptnr_yn from Dataset1: non-missing elec_cap = {df.elec_cap.notna().sum():,}')

df['industry'] = df['industry'].astype(int)
IND = pd.get_dummies(df['industry'], prefix='ind', drop_first=True).astype(float)
exposure = df['n_workers'].values

def fit_nb(y, X, expo, **kw):
    return sm.GLM(np.asarray(y).astype(int), X,
                  family=sm.families.NegativeBinomial(),
                  exposure=np.asarray(expo)).fit(**kw)

def irr_row(m, idx):
    c, s, p = m.params[idx], m.bse[idx], m.pvalues[idx]
    return (round(np.exp(c), 3), round(np.exp(c - 1.96 * s), 3),
            round(np.exp(c + 1.96 * s), 3), round(p, 4))

RES = {'_meta': {'timestamp': TS, 'N': int(len(df)), 'input_sha256_16': sha}}

# ==================================================================
# A. Population-weighted size-category shares (R1.4)
# ==================================================================
banner('A. POPULATION-WEIGHTED SIZE SHARES (wt1)')
lab = {1: '1-4', 2: '5-19', 3: '20-49', 4: '50-99', 5: '>=100'}
rows = []
wtot = df['wt1'].sum()
for sz in [1, 2, 3, 4, 5]:
    sub = df[df['r_wrk_tot'] == sz]
    rows.append({'Size category': f'{lab[sz]} workers',
                 'n (sample)': len(sub),
                 'Sample %': round(100 * len(sub) / len(df), 1),
                 'Weighted population estimate': int(round(sub['wt1'].sum())),
                 'Population %': round(100 * sub['wt1'].sum() / wtot, 2)})
df_pop = pd.DataFrame(rows)
save_csv(df_pop, 'table_size_population_shares.csv')
print(df_pop.to_string(index=False))
RES['pop_shares'] = df_pop.to_dict('records')

# ==================================================================
# B. S9: top size category mapped to 1,000 / 2,000 workers (R1.4)
# ==================================================================
banner('B. S9 OFFSET EXTENSION: top category = 1,000 / 2,000 workers')
baseX = sm.add_constant(pd.concat([df[['sc_total_z', 'log_prior', 'size_cat']].reset_index(drop=True),
                                   IND.reset_index(drop=True)], axis=1).astype(float).values)
MID = {1: 2.5, 2: 12.0, 3: 34.5, 4: 74.5}
s9 = []
for top in [200.0, 1000.0, 2000.0]:
    nw = df['r_wrk_tot'].map({**MID, 5: top}).values
    m = fit_nb(df['vic_2024_appr'], baseX, nw)
    irr, lo, hi, p = irr_row(m, 1)
    s9.append({'Specification': f'Top category = {int(top)} workers',
               'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
               'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
df_s9 = pd.DataFrame(s9)
save_csv(df_s9, 'table_s9_top1000.csv')
print(df_s9[['Specification', 'IRR (95% CI)', 'p']].to_string(index=False))
RES['s9_offset_ext'] = df_s9.to_dict('records')

# ==================================================================
# C. S10: within >=100 stratum, moderation by very-large-scale markers
# ==================================================================
banner('C. S10 WITHIN >=100 STRATUM: scale-marker moderation')
top = df[df['r_wrk_tot'] == 5].copy()
print(f'>=100 stratum n = {len(top):,}')

def moderation(sub, marker, marker_name):
    """NB model within stratum: SC*marker interaction + industry FE + log_prior."""
    sub = sub.dropna(subset=[marker]).copy()
    ind_d = pd.get_dummies(sub['industry'], prefix='ind', drop_first=True).astype(float)
    sub['sc_x_m'] = sub['sc_total_z'] * sub[marker]
    X = sm.add_constant(pd.concat(
        [sub[['sc_total_z', marker, 'sc_x_m', 'log_prior']].reset_index(drop=True),
         ind_d.reset_index(drop=True)], axis=1).astype(float).values)
    m = fit_nb(sub['vic_2024_appr'], X, sub['n_workers'].values)
    # SC effect when marker == 0
    irr0, lo0, hi0, p0 = irr_row(m, 1)
    # SC effect when marker == 1 (linear combination b1 + b3)
    b = m.params[1] + m.params[3]
    V = m.cov_params()
    se = np.sqrt(V[1, 1] + V[3, 3] + 2 * V[1, 3])
    irr1, lo1, hi1 = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    p1 = 2 * stats.norm.sf(abs(b / se))
    p_int = m.pvalues[3]
    n0, n1 = int((sub[marker] == 0).sum()), int((sub[marker] == 1).sum())
    out = [
        {'Moderator': marker_name, 'Stratum': 'Marker absent', 'n': n0,
         'IRR': round(irr0, 3), 'CI_lo': round(lo0, 3), 'CI_hi': round(hi0, 3), 'p': round(p0, 3),
         'p_interaction': round(p_int, 3),
         'IRR (95% CI)': f'{irr0:.2f} ({lo0:.2f}-{hi0:.2f})'},
        {'Moderator': marker_name, 'Stratum': 'Marker present', 'n': n1,
         'IRR': round(irr1, 3), 'CI_lo': round(lo1, 3), 'CI_hi': round(hi1, 3), 'p': round(p1, 3),
         'p_interaction': round(p_int, 3),
         'IRR (95% CI)': f'{irr1:.2f} ({lo1:.2f}-{hi1:.2f})'},
    ]
    return out

# marker 1: electrical contract capacity >= 1,000 kW (elec_cap == 6; 8/9 = unknown/refused -> NA)
top['very_large_kw'] = np.where(top['elec_cap'].isin([8, 9]) | top['elec_cap'].isna(), np.nan,
                                (top['elec_cap'] == 6).astype(float))
# marker 2: dedicated in-house safety department (1 = yes, 2 = no)
top['safety_dept'] = np.where(top['saf_dept_yn_raw'].isin([1, 2]),
                              (top['saf_dept_yn_raw'] == 1).astype(float), np.nan)

s10 = []
s10 += moderation(top, 'very_large_kw', 'Electrical capacity >=1,000 kW')
s10 += moderation(top, 'safety_dept', 'Dedicated safety department')
df_s10 = pd.DataFrame(s10)
save_csv(df_s10, 'table_s10_toplevel_moderation.csv')
print(df_s10[['Moderator', 'Stratum', 'n', 'IRR (95% CI)', 'p', 'p_interaction']].to_string(index=False))
RES['s10_moderation'] = df_s10.to_dict('records')

# ==================================================================
# D. S11: exclude the >=100 category entirely (R1.4)
# ==================================================================
banner('D. S11: 1-99 WORKERS ONLY (drop top-coded category)')
sub99 = df[df['r_wrk_tot'] <= 4]
ind99 = pd.get_dummies(sub99['industry'], prefix='ind', drop_first=True).astype(float)
X99 = sm.add_constant(pd.concat([sub99[['sc_total_z', 'log_prior', 'size_cat']].reset_index(drop=True),
                                 ind99.reset_index(drop=True)], axis=1).astype(float).values)
m99 = fit_nb(sub99['vic_2024_appr'], X99, sub99['n_workers'].values)
irr, lo, hi, p = irr_row(m99, 1)
df_s11 = pd.DataFrame([{'Analysis': 'S11: 1-99 workers only (top-coded category excluded)',
                        'n': len(sub99), 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
                        'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'}])
save_csv(df_s11, 'table_s11_exclude_top.csv')
print(df_s11[['Analysis', 'n', 'IRR (95% CI)', 'p']].to_string(index=False))
RES['s11_exclude_top'] = df_s11.to_dict('records')

# ==================================================================
# E. Reactive-upgrading diagnostic (R1.5)
# ==================================================================
banner('E. REACTIVE-UPGRADING DIAGNOSTIC (R1.5)')
# groups from APPROVED counts: prior = 2022-23, new = 2024
prior_any = (df['vic_prior'] > 0)
y2024_any = (df['vic_2024_appr'] > 0)
df['hist_group'] = np.select(
    [~prior_any & ~y2024_any, prior_any & ~y2024_any, ~prior_any & y2024_any, prior_any & y2024_any],
    ['never', 'prior_only', 'new_2024_only', 'persistent'], default='never')
print(df['hist_group'].value_counts().to_string())

size_d = pd.get_dummies(df['size_cat'], prefix='sz', drop_first=True).astype(float)
Xg = pd.concat([pd.get_dummies(df['hist_group'], prefix='g')[['g_prior_only', 'g_new_2024_only', 'g_persistent']]
                .astype(float).reset_index(drop=True),
                size_d.reset_index(drop=True), IND.reset_index(drop=True)], axis=1)
Xg = sm.add_constant(Xg.astype(float))
m_ols = sm.OLS(df['sc_total_z'].values, Xg.values).fit(cov_type='HC1')
names = list(Xg.columns)
rows = []
raw_means = df.groupby('hist_group')['sc_total_z'].mean()
for g, label in [('g_prior_only', 'Prior injuries only (2022-23, none in 2024)'),
                 ('g_new_2024_only', 'New 2024 injury only (none in 2022-23)'),
                 ('g_persistent', 'Injuries in both periods')]:
    i = names.index(g)
    b, se, p = m_ols.params[i], m_ols.bse[i], m_ols.pvalues[i]
    key = g.replace('g_', '')
    rows.append({'Group (vs never injured 2022-2024)': label,
                 'n': int((df['hist_group'] == key).sum()),
                 'Unadjusted mean SC (z)': round(float(raw_means[key]), 3),
                 'Adjusted difference in SC (z, SD units)': round(b, 3),
                 'CI_lo': round(b - 1.96 * se, 3), 'CI_hi': round(b + 1.96 * se, 3),
                 'p': round(p, 4),
                 'Diff (95% CI)': f'{b:+.3f} ({b - 1.96 * se:+.3f} to {b + 1.96 * se:+.3f})'})
df_re = pd.DataFrame(rows)
nev = df[df['hist_group'] == 'never']
print(f"never group: n = {len(nev):,}, unadjusted mean SC z = {nev['sc_total_z'].mean():.3f}")
save_csv(df_re, 'table_reactive_upgrading.csv')
print(df_re[['Group (vs never injured 2022-2024)', 'n', 'Diff (95% CI)', 'p']].to_string(index=False))
RES['reactive_upgrading'] = {'never_n': int(len(nev)),
                             'never_mean_z': round(float(nev['sc_total_z'].mean()), 3),
                             'rows': df_re.to_dict('records')}

# ------------------------------------------------------------------
# Save headline JSON
# ------------------------------------------------------------------
for d in (OUT_RUN, OUT_REV):
    with open(os.path.join(d, 'results_revision2.json'), 'w', encoding='utf-8') as f:
        json.dump(RES, f, ensure_ascii=False, indent=2)
banner(f'DONE -> {OUT_REV}')

# ==================================================================
# F. REPRODUCE THE REVISION FIGURE SET (renumbered)
#    Figure_1  = sequential-adjustment cascade   (main text Fig. 1)
#    Figure_2  = unadjusted vs adjusted forest   (main text Fig. 2)
#    Figure_3  = ML permutation importance       (main text Fig. 3)
#    Figure_S1 = DAG                             (supplement)
#    Figure_S2 = RCS dose-response               (supplement)
#    Figure_S3 = between-sub-sector scatter      (supplement)
#    graphical_abstract
#    -> z_revision_1/output/figures/ (PNG 350 dpi + PDF)
#    Ported unchanged (seed 42, same specs) from analysis_revision.py
#    so every uploaded image is reproducible from this one notebook.
# ==================================================================
banner('F. REPRODUCE REVISION FIGURES')
import importlib, subprocess
for _pkg in ('adjustText',):
    if importlib.util.find_spec(_pkg) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg], check=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = os.path.join(OUT_REV, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
      'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'grey': '#9AA7B0', 'ink': '#22303C'}
DIMCOL = [CB['blue'], CB['orange'], CB['green'], CB['red'], CB['purple'], CB['sky']]
matplotlib.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 350, 'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})
ROPE_LO, ROPE_HI = 1 / 1.10, 1.10

def savefig(fig, stem, dpi=350):
    fig.savefig(os.path.join(FIG_DIR, stem + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem} (png + pdf)')

SC_DIMS = {
    'sc_mgmt':  ['mgt_emph_saf', 'mgt_prior_saf', 'mgt_value_saf'],
    'sc_comm':  ['saf_disc_opp', 'saf_open_disc', 'saf_feed_reg', 'saf_sug_sys', 'saf_sug_resp'],
    'sc_train': ['saf_tr_opp', 'saf_tr_effect'],
    'sc_sys':   ['saf_sys_proc', 'saf_proc_effect', 'saf_equip_avail'],
    'sc_empow': ['work_ref_unsaf', 'work_vol_saf'],
}
SC_ITEMS = [i for items in SC_DIMS.values() for i in items]
DIM_LABEL = {'sc_mgmt': 'A. Management Commitment', 'sc_comm': 'B. Safety Communication',
             'sc_train': 'C. Safety Training', 'sc_sys': 'D. Safety Systems',
             'sc_empow': 'E. Worker Empowerment'}
KSIC = {10: 'Food', 11: 'Beverages', 13: 'Textiles', 14: 'Apparel', 15: 'Leather',
        16: 'Wood', 17: 'Paper', 18: 'Printing', 19: 'Coke/petroleum', 20: 'Chemicals',
        21: 'Pharma', 22: 'Rubber/plastics', 23: 'Non-metallic min.', 24: 'Basic metals',
        25: 'Fabricated metal', 26: 'Electronics', 27: 'Medical/precision', 28: 'Electrical eq.',
        29: 'Machinery', 30: 'Motor vehicles', 31: 'Other transport eq.', 32: 'Furniture',
        33: 'Other mfg', 34: 'Machinery repair'}
EXP = [(k + '_z', v) for k, v in DIM_LABEL.items()] + [('sc_total_z', 'Overall (15 items)')]

# --- adjusted + unadjusted models for the 6 exposures ---------------
adj_rows, unadj_rows = [], []
for var, label in EXP:
    Xa = sm.add_constant(pd.concat([df[[var, 'log_prior', 'size_cat']].reset_index(drop=True),
                                    IND.reset_index(drop=True)], axis=1).astype(float).values)
    ma = fit_nb(df['vic_2024_appr'], Xa, exposure)
    irr, lo, hi, p = irr_row(ma, 1)
    adj_rows.append({'Exposure': label, 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
                     'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
    Xu = sm.add_constant(df[[var]].astype(float).values)
    mu = fit_nb(df['vic_2024_appr'], Xu, exposure)
    uirr, ulo, uhi, up = irr_row(mu, 1)
    unadj_rows.append({'Exposure': label, 'IRR': uirr, 'CI_lo': ulo, 'CI_hi': uhi, 'p': up})
df_adj = pd.DataFrame(adj_rows); df_unadj = pd.DataFrame(unadj_rows)
ov = df_adj.iloc[-1]; ovu = df_unadj.iloc[-1]

# --- Figure_1: sequential-adjustment cascade ------------------------
casc_specs = [('Crude (offset only)', []),
              ('+ Industry fixed effects', ['IND']),
              ('+ Firm size', ['IND', 'size_cat']),
              ('+ Prior injuries (full model)', ['IND', 'size_cat', 'log_prior'])]
casc = []
for label, extra in casc_specs:
    cols = [df['sc_total_z'].reset_index(drop=True)]
    for e in extra:
        cols.append(IND.reset_index(drop=True) if e == 'IND' else df[e].reset_index(drop=True))
    X = sm.add_constant(pd.concat(cols, axis=1).astype(float).values)
    m = fit_nb(df['vic_2024_appr'], X, exposure)
    irr, lo, hi, p = irr_row(m, 1)
    casc.append({'step': label, 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p})
df_casc = pd.DataFrame(casc)
print(df_casc[['step', 'IRR', 'CI_lo', 'CI_hi']].to_string(index=False))
fig, ax = plt.subplots(figsize=(9.5, 5.6))
yp = np.arange(len(df_casc))[::-1]
ax.axvspan(ROPE_LO, ROPE_HI, color=CB['green'], alpha=0.08, zorder=0, label='±10% practical-equivalence')
ax.plot(df_casc['IRR'], yp, '-', color=CB['grey'], lw=1.4, zorder=1)
for i, (_, r) in enumerate(df_casc.iterrows()):
    col = CB['red'] if i == 0 else (CB['ink'] if i == len(df_casc) - 1 else CB['blue'])
    ax.errorbar(r['IRR'], yp[i], xerr=[[r['IRR'] - r['CI_lo']], [r['CI_hi'] - r['IRR']]],
                fmt='o', color=col, markersize=11, capsize=5, lw=2.2, zorder=3)
    ax.text(r['CI_hi'] + 0.012, yp[i], f"{r['IRR']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})",
            va='center', fontsize=10, color=CB['ink'])
ax.axvline(1, color=CB['ink'], ls='--', lw=1, alpha=0.7)
ax.set_yticks(yp); ax.set_yticklabels(df_casc['step'])
ax.set_xlim(0.92, 1.27)
ax.set_xlabel('Overall safety-culture IRR per 1-SD (with 95% CI)')
ax.set_title('How the safety-culture–injury association attenuates with confounder adjustment\n'
             'Crude 1.18 → 1.01 (null) after industry, firm size and prior injuries')
ax.legend(loc='lower right')
savefig(fig, 'Figure_1')

# --- Figure_2: unadjusted vs adjusted forest ------------------------
fig, ax = plt.subplots(figsize=(11, 5.8))
yp = np.arange(len(df_adj)) * 2
ax.axvspan(ROPE_LO, ROPE_HI, color=CB['green'], alpha=0.07, zorder=0)
for i, (_, r) in enumerate(df_adj.iterrows()):
    u = df_unadj.iloc[i]
    ax.errorbar(u['IRR'], yp[i] + 0.34, xerr=[[u['IRR'] - u['CI_lo']], [u['CI_hi'] - u['IRR']]],
                fmt='s', color=CB['red'], markersize=8, capsize=4, lw=1.8, label='Unadjusted' if i == 0 else '', zorder=3)
    ax.errorbar(r['IRR'], yp[i] - 0.34, xerr=[[r['IRR'] - r['CI_lo']], [r['CI_hi'] - r['IRR']]],
                fmt='o', color=CB['ink'], markersize=8, capsize=4, lw=1.8, label='Adjusted' if i == 0 else '', zorder=3)
ax.axvline(1, color=CB['ink'], ls='--', lw=1, alpha=0.7)
ax.set_yticks(yp); ax.set_yticklabels(df_adj['Exposure'])
ax.set_xlabel('Incidence rate ratio (IRR) per 1-SD safety culture')
ax.set_title('Confounding resolution: unadjusted vs adjusted\n(industry + firm size + prior injuries; green = ±10% equivalence)')
ax.legend(loc='upper right')
savefig(fig, 'Figure_2')

# --- Figure_3: ML permutation importance ----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
Xrf = df[SC_ITEMS + ['log_prior', 'size_cat']].copy()
for ind_code in df['industry'].value_counts().head(8).index:
    Xrf[f'ind_{ind_code}'] = (df['industry'] == ind_code).astype(int)
yrf = df['any_acc_2024'].values
Xtr, Xte, ytr, yte = train_test_split(Xrf, yrf, test_size=0.30, random_state=42, stratify=yrf)
rf = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(Xtr, ytr)
auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
print(f'  held-out ROC-AUC: {auc:.3f}')
pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=42, n_jobs=-1, scoring='roc_auc')
fi = pd.DataFrame({'Feature': Xrf.columns, 'Importance': pi.importances_mean,
                   'Importance_sd': pi.importances_std}).sort_values('Importance', ascending=False)
LBL = {**{k: k for k in fi['Feature']}, 'log_prior': 'Prior injuries (log)', 'size_cat': 'Firm size'}
topn = fi.head(18).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
clist = [CB['red'] if f in SC_ITEMS else CB['grey'] for f in topn['Feature']]
ax.barh(range(len(topn)), topn['Importance'], xerr=topn['Importance_sd'], color=clist,
        error_kw=dict(ecolor=CB['ink'], lw=0.9, capsize=2), zorder=3)
ax.set_yticks(range(len(topn)))
ax.set_yticklabels([LBL.get(f, f).replace('ind_', 'Industry ') for f in topn['Feature']])
ax.set_xlabel('Permutation importance (mean decrease in held-out ROC-AUC ± SD)')
ax.set_title(f'Machine-learning feature importance (random forest; held-out AUC = {auc:.2f})\n'
             'Red = safety-culture items; grey = structural controls')
ax.legend(handles=[mpatches.Patch(color=CB['red'], label='Safety-culture items'),
                   mpatches.Patch(color=CB['grey'], label='Structural controls')], loc='lower right')
savefig(fig, 'Figure_3')

# --- Figure_S1: DAG --------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-0.15, 7)
NODES, CTR = {}, {}
def node(key, x, y, txt, fc):
    pch = FancyBboxPatch((x - 1.15, y - 0.45), 2.3, 0.9, boxstyle='round,pad=0.04',
                         fc=fc, ec=CB['ink'], lw=1.4, zorder=3)
    ax.add_patch(pch)
    ax.text(x, y, txt, ha='center', va='center', fontsize=10.5, fontweight='bold', zorder=4, color=CB['ink'])
    NODES[key] = pch; CTR[key] = (x, y)
def arrow(k1, k2, color, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(CTR[k1], CTR[k2], arrowstyle='-|>', mutation_scale=15, lw=1.7,
                                 color=color, ls=ls, shrinkA=4, shrinkB=5,
                                 patchA=NODES[k1], patchB=NODES[k2],
                                 connectionstyle=f'arc3,rad={rad}', zorder=2))
node('exp', 2.0, 1.2, 'Safety culture\n(exposure)', '#AED9F0')
node('out', 8.0, 1.2, '2024 injuries\n(outcome)', '#F6B8AE')
node('ind', 2.2, 5.7, 'Industry\n(KSIC 2-digit)', '#FBE3C2')
node('siz', 5.0, 6.2, 'Firm size', '#FBE3C2')
node('pri', 7.8, 5.7, 'Prior injuries\n(2022–2023)', '#FBE3C2')
arrow('exp', 'out', CB['blue'])
ax.text(5.0, 0.45, 'effect of interest (≈ null after adjustment)', ha='center', va='center',
        fontsize=9.5, color=CB['blue'], style='italic')
for src in ['ind', 'siz', 'pri']:
    arrow(src, 'exp', CB['grey'], rad=0.05)
    arrow(src, 'out', CB['grey'], rad=-0.05)
ax.text(5.0, 6.85, 'Confounders (minimally sufficient adjustment set)', ha='center', fontsize=10.5, fontweight='bold', color=CB['grey'])
ax.set_title('Directed acyclic graph: safety culture → manufacturing injuries')
savefig(fig, 'Figure_S1')

# --- Figure_S2: RCS dose-response ------------------------------------
from patsy import dmatrix
basis = dmatrix('cr(sc_total, df=4) - 1', df, return_type='dataframe')
DI = basis.design_info
covars = pd.concat([df[['log_prior', 'size_cat']].reset_index(drop=True),
                    IND.reset_index(drop=True)], axis=1).astype(float)
X_sp = sm.add_constant(pd.concat([basis.reset_index(drop=True), covars], axis=1).astype(float).values)
m_sp = fit_nb(df['vic_2024_appr'], X_sp, exposure, maxiter=5000)
X_lin = sm.add_constant(pd.concat([df[['sc_total']].reset_index(drop=True), covars], axis=1).astype(float).values)
m_lin = fit_nb(df['vic_2024_appr'], X_lin, exposure, maxiter=5000)
LR = 2 * (m_sp.llf - m_lin.llf); df_nl = basis.shape[1] - 1
p_nl = stats.chi2.sf(LR, df_nl)
print(f'  RCS non-linearity LR = {LR:.2f}, df = {df_nl}, p = {p_nl:.3f}')
grid = np.linspace(df['sc_total'].min(), df['sc_total'].max(), 100)
gb = np.asarray(dmatrix(DI, pd.DataFrame({'sc_total': grid}), return_type='dataframe'))
cov_means = np.concatenate([[covars['log_prior'].mean(), covars['size_cat'].mean()], IND.mean().values])
ref_basis = np.asarray(dmatrix(DI, pd.DataFrame({'sc_total': [df['sc_total'].mean()]}), return_type='dataframe'))
Xg = np.column_stack([np.ones(len(grid)), gb, np.tile(cov_means, (len(grid), 1))])
Xref = np.concatenate([[1.0], ref_basis.ravel(), cov_means])
L = Xg - Xref
beta, V = m_sp.params, m_sp.cov_params()
log_irr = L @ beta
se = np.sqrt(np.einsum('ij,jk,ik->i', L, V, L))
irr_g = np.exp(log_irr); lo_g = np.exp(log_irr - 1.96 * se); hi_g = np.exp(log_irr + 1.96 * se)
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(grid, lo_g, hi_g, alpha=0.18, color='#3498db', label='95% CI')
ax.plot(grid, irr_g, color='#2c3e50', lw=2.5, label='Adjusted IRR')
ax.axhline(1, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Safety culture score (1-5)'); ax.set_ylabel('IRR (reference = sample mean)')
ax.set_title('Dose-response: safety culture -> 2024 injuries\n'
             f'(restricted cubic spline, 4 knots, adjusted; non-linearity p = {p_nl:.2f})')
ax2 = ax.twinx()
ax2.hist(df['sc_total'], bins=40, color='#bdc3c7', alpha=0.35, zorder=0)
ax2.set_ylabel('Establishments (histogram)', color='#7f8c8d')
ax2.tick_params(axis='y', colors='#7f8c8d'); ax2.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)
ax.legend(loc='upper center'); ax.grid(alpha=0.3)
savefig(fig, 'Figure_S2')

# --- Figure_S3: between-sub-sector scatter ----------------------------
g = df.groupby('industry').agg(sc=('sc_total', 'mean'), vic=('vic_2024_appr', 'sum'),
                               wk=('n_workers', 'sum'), n=('id', 'count')).reset_index()
g['rate'] = 100 * g['vic'] / g['wk']
r_ind = np.corrcoef(g['sc'], g['rate'])[0, 1]
g = g.sort_values('rate', ascending=False).reset_index(drop=True)
g['num'] = np.arange(1, len(g) + 1)
Xg2 = sm.add_constant(g['sc'].values)
ols = sm.OLS(g['rate'].values, Xg2).fit()
xs = np.linspace(g['sc'].min(), g['sc'].max(), 80)
pr = ols.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.fill_between(xs, pr['mean_ci_lower'], pr['mean_ci_upper'], color=CB['blue'], alpha=0.10, zorder=1)
ax.plot(xs, pr['mean'], color=CB['blue'], lw=1.8, zorder=2)
sizes = 40 + 700 * g['n'] / g['n'].max()
ax.scatter(g['sc'], g['rate'], s=sizes, c=g['rate'], cmap='YlOrRd', edgecolor=CB['ink'], lw=0.8, alpha=0.85, zorder=3)
ax.margins(x=0.05, y=0.10)
nums = [ax.text(r['sc'], r['rate'], str(int(r['num'])), fontsize=8, fontweight='bold',
                ha='center', va='center', color=CB['ink'], zorder=6,
                bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec=CB['grey'], lw=0.4, alpha=0.9))
        for _, r in g.iterrows()]
try:
    from adjustText import adjust_text
    adjust_text(nums, x=g['sc'].to_numpy(), y=g['rate'].to_numpy(), ax=ax,
                expand=(1.4, 1.6), force_text=(0.5, 0.7), force_static=(0.5, 0.8), min_arrow_len=0,
                arrowprops=dict(arrowstyle='-', color=CB['grey'], lw=0.5, alpha=0.7))
except Exception as e:
    print('  [warn] adjustText unavailable -> static numbers:', e)
gg = g.sort_values('num')
items = [f"{int(r['num']):>2}. {KSIC.get(int(r['industry']), int(r['industry']))}" for _, r in gg.iterrows()]
half = (len(items) + 1) // 2
left, right = items[:half], items[half:] + [''] * (2 * half - len(items))
legend = "\n".join(f"{l:<24}{rr}" for l, rr in zip(left, right))
ax.text(0.015, 0.985, legend, transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
        family='DejaVu Sans Mono', linespacing=1.4,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=CB['grey'], lw=0.6, alpha=0.93))
ax.set_xlabel('Sub-sector mean safety culture (1–5)'); ax.set_ylabel('Crude injury rate (per 100 workers)')
ax.set_title(f'Between-sub-sector confounding within manufacturing\n'
             f'(r = {r_ind:.2f}; bubble size ∝ no. of establishments; band = 95% CI of fit)')
savefig(fig, 'Figure_S3')

# --- graphical abstract ----------------------------------------------
fig = plt.figure(figsize=(13.28, 5.31)); gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
axL = fig.add_subplot(gs[0, 0]); yy = np.arange(len(df_adj))[::-1]
axL.axvspan(ROPE_LO, ROPE_HI, color=CB['green'], alpha=0.08, zorder=0)
for i, (_, r) in enumerate(df_adj.iterrows()):
    u = df_unadj.iloc[i]
    axL.errorbar(u['IRR'], yy[i] + 0.18, xerr=[[u['IRR'] - u['CI_lo']], [u['CI_hi'] - u['IRR']]],
                 fmt='s', color=CB['red'], markersize=6, capsize=3, lw=1.4)
    axL.errorbar(r['IRR'], yy[i] - 0.18, xerr=[[r['IRR'] - r['CI_lo']], [r['CI_hi'] - r['IRR']]],
                 fmt='o', color=CB['ink'], markersize=6, capsize=3, lw=1.4)
axL.axvline(1, color=CB['ink'], ls='--', alpha=0.7); axL.set_yticks(yy)
axL.set_yticklabels([e.split('(')[0].strip() for e in df_adj['Exposure']], fontsize=9)
axL.set_xlabel('IRR per 1-SD safety culture'); axL.set_xlim(0.85, 1.32)
axL.legend(handles=[mpatches.Patch(color=CB['red'], label='Unadjusted'),
                    mpatches.Patch(color=CB['ink'], label='Adjusted')], loc='lower right', fontsize=8)
axL.set_title('The safety culture–injury paradox', fontsize=12)
axR = fig.add_subplot(gs[0, 1]); axR.axis('off')
axR.text(0.5, 0.92, '20,262 Korean manufacturing establishments (WES-7)', ha='center', fontsize=12, fontweight='bold')
axR.text(0.5, 0.6, f"Unadjusted:  IRR {ovu['IRR']:.2f} ({ovu['CI_lo']:.2f}–{ovu['CI_hi']:.2f})\n"
         r"$\Downarrow$  adjust industry + size + prior injuries" + "\n"
         f"Adjusted:  IRR {ov['IRR']:.2f} ({ov['CI_lo']:.2f}–{ov['CI_hi']:.2f}),  p = {ov['p']:.2f}",
         ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.6', fc='#F4F6F7', ec=CB['ink']))
axR.text(0.5, 0.16, 'The apparent positive association is confounding\nby occupational structure — not a real effect.',
         ha='center', va='center', fontsize=11, color=CB['ink'])
savefig(fig, 'graphical_abstract')

RES['figures'] = {'dir': FIG_DIR,
                  'cascade': [list(r.values()) for r in casc],
                  'ml_auc': round(float(auc), 3),
                  'rcs': {'LR': round(float(LR), 2), 'p': round(float(p_nl), 3)},
                  'industry_r': round(float(r_ind), 2)}
with open(os.path.join(OUT_REV, 'results_revision2.json'), 'w', encoding='utf-8') as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
banner('FIGURES DONE -> ' + FIG_DIR)
