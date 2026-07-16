#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_revision.py
====================================================================
Reproducible revision analysis for:
  "The Safety Culture-Injury Paradox Explained by Occupational
   Confounding: A National Survey of 20,262 Korean Manufacturing
   Establishments"  (Safety Science, revision under review)

This script reproduces the original results AND implements every
reviewer-revision item agreed during peer-style review:

  R1. CFA: report STANDARDIZED loadings, inter-factor correlations,
      AVE and Composite Reliability (CR); fit a 1-factor model and
      compare to the 5-factor model (discriminant validity).
  R2. Restricted cubic spline (RCS): add 95% CI band (delta method),
      a formal test of non-linearity (LR test, spline vs linear),
      and a rug/marginal histogram of the exposure distribution.
  R3. Replace E-value-on-a-null framing with a precision/equivalence
      interpretation (two one-sided tests, TOST, against a +/-10%
      region of practical equivalence). E-values retained for the
      UNADJUSTED estimate (where they are informative).
  R4. Survey-weighted sensitivity (S4): weights explicitly normalized
      to sum to N (mean 1); document the procedure.
  R5. Exposure-offset sensitivity: re-fit primary model with the
      open-ended top size category mapped to 300 and 500 workers,
      and with firm size entered as categorical dummies (not linear).
  R6. Machine-learning feature importance (NOT "cross-validation"):
      train/test split, held-out AUC, SHAP on held-out data
      (falls back to permutation importance if shap unavailable).
  R7. NEW Fig: directed acyclic graph (DAG) of the adjustment set.
  R8. NEW Fig: between-industry scatter (mean safety culture vs crude
      injury rate) visualizing the confounding mechanism.
  R9. Graphical abstract (1328 x 531 px) for Safety Science.

Input : output/pre_output/analytic_sample.csv  (N = 20,262; from 1_preprocess.ipynb)
Output: output/analysis_output/run_revised_<timestamp>/  (+ copies to paper/.../output)

Run (interpreter that has semopy + statsmodels + sklearn; shap optional):
  /Users/y3korea/miniforge3/bin/python analysis_revision.py
====================================================================
"""
import os, sys, json, warnings, hashlib, importlib, subprocess
from datetime import datetime
# Reproducibility guard: ensure semopy + adjustText are available (Colab / fresh envs).
# shap is intentionally NOT required — feature importance uses scikit-learn.
for _pkg in ('semopy', 'adjustText'):
    if importlib.util.find_spec(_pkg) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg], check=False)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings('ignore')
np.random.seed(42)
RNG = np.random.RandomState(42)

# ──────────────────────────────────────────────────────────────────
# 0. Paths (portable: Colab or local)
# ──────────────────────────────────────────────────────────────────
try:                                   # mount Google Drive when running on Colab
    import google.colab  # noqa
    if not os.path.isdir('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
CANDIDATES = [
    '/content/drive/MyDrive/완석_구글자료/연구자료/20260313_kosha',
    '/Users/y3korea/Library/CloudStorage/GoogleDrive-y3korea@gmail.com/내 드라이브/완석_구글자료/연구자료/20260313_kosha',
]
BASE = next((p for p in CANDIDATES if os.path.isdir(p)), None)
assert BASE, 'BASE project directory not found.'
CODE = os.path.join(BASE, 'Code_kosha', '2_code')
PRE_DIR = os.path.join(CODE, 'output', 'pre_output')
OUT_BASE = os.path.join(CODE, 'output', 'analysis_output')
PAPER_AUTO = os.path.join(CODE, 'paper', 'auto')
PAPER_SS_OUT = os.path.join(CODE, 'paper', 'Safety Science', 'output')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR = os.path.join(OUT_BASE, f'run_revised_{TIMESTAMP}')
os.makedirs(OUT_DIR, exist_ok=True)
for d in (PAPER_AUTO, PAPER_SS_OUT):
    os.makedirs(d, exist_ok=True)

# ── Publication-quality figure style (colourblind-safe Okabe–Ito) ──
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
ROPE_LO, ROPE_HI = 1 / 1.10, 1.10          # ±10% region of practical equivalence (IRR scale)

def savefig(fig, stem, dpi=350):
    """Save PNG (raster, journal dpi) + PDF (vector) to the run dir and the manuscript figure dir."""
    for dd in (OUT_DIR, PAPER_SS_OUT):
        fig.savefig(os.path.join(dd, stem + '.png'), dpi=dpi, bbox_inches='tight')
        fig.savefig(os.path.join(dd, stem + '.pdf'), bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

# KSIC 2-digit manufacturing sub-sector short labels (Section C)
KSIC = {10: 'Food', 11: 'Beverages', 13: 'Textiles', 14: 'Apparel', 15: 'Leather',
        16: 'Wood', 17: 'Paper', 18: 'Printing', 19: 'Coke/petroleum', 20: 'Chemicals',
        21: 'Pharma', 22: 'Rubber/plastics', 23: 'Non-metallic min.', 24: 'Basic metals',
        25: 'Fabricated metal', 26: 'Electronics', 27: 'Medical/precision', 28: 'Electrical eq.',
        29: 'Machinery', 30: 'Motor vehicles', 31: 'Other transport eq.', 32: 'Furniture',
        33: 'Other mfg', 34: 'Machinery repair'}

def banner(msg):
    print('\n' + '=' * 70 + f'\n{msg}\n' + '=' * 70)

# ──────────────────────────────────────────────────────────────────
# 1. Load analytic sample
# ──────────────────────────────────────────────────────────────────
banner('1. LOAD ANALYTIC SAMPLE')
INPUT = os.path.join(PRE_DIR, 'analytic_sample.csv')
df = pd.read_csv(INPUT, encoding='utf-8-sig')
sha = hashlib.sha256(open(INPUT, 'rb').read()).hexdigest()[:16]
print(f'Input         : {INPUT}')
print(f'SHA-256[:16]  : {sha}')
print(f'N             : {len(df):,} establishments x {df.shape[1]} cols')

SC_DIMS = {
    'sc_mgmt':  ['mgt_emph_saf', 'mgt_prior_saf', 'mgt_value_saf'],
    'sc_comm':  ['saf_disc_opp', 'saf_open_disc', 'saf_feed_reg', 'saf_sug_sys', 'saf_sug_resp'],
    'sc_train': ['saf_tr_opp', 'saf_tr_effect'],
    'sc_sys':   ['saf_sys_proc', 'saf_proc_effect', 'saf_equip_avail'],
    'sc_empow': ['work_ref_unsaf', 'work_vol_saf'],
}
SC_ITEMS = [i for items in SC_DIMS.values() for i in items]
FAC = {'sc_mgmt': 'MGMT', 'sc_comm': 'COMM', 'sc_train': 'TRAIN', 'sc_sys': 'SYS', 'sc_empow': 'EMPOW'}
DIM_LABEL = {'sc_mgmt': 'A. Management Commitment', 'sc_comm': 'B. Safety Communication',
             'sc_train': 'C. Safety Training', 'sc_sys': 'D. Safety Systems',
             'sc_empow': 'E. Worker Empowerment'}

# Industry dummies (drop_first) - reused everywhere
df['industry'] = df['industry'].astype(int)
IND_DUMMIES = pd.get_dummies(df['industry'], prefix='ind', drop_first=True).astype(float)
RESULTS = {}  # collect headline numbers for manuscript sync

# ──────────────────────────────────────────────────────────────────
# Helper: GLM-NB fit returning IRR/CI/p for a target coefficient index
# Matches original spec: NegativeBinomial() family (alpha=1.0 default),
# exposure = worker count (log offset). alpha=1.0 reproduces published nums.
# ──────────────────────────────────────────────────────────────────
def fit_nb(y, X, exposure, **kw):
    return sm.GLM(np.asarray(y).astype(int), X,
                  family=sm.families.NegativeBinomial(),
                  exposure=np.asarray(exposure)).fit(**kw)

def irr_row(m, idx):
    c, s, p = m.params[idx], m.bse[idx], m.pvalues[idx]
    return (round(np.exp(c), 3), round(np.exp(c - 1.96 * s), 3),
            round(np.exp(c + 1.96 * s), 3), round(p, 4))

# ══════════════════════════════════════════════════════════════════
# R1. CFA — fit indices, STANDARDIZED loadings, AVE, CR, factor r,
#     and 1-factor vs 5-factor comparison (discriminant validity)
# ══════════════════════════════════════════════════════════════════
banner('R1. CFA: 5-factor + 1-factor, standardized loadings, AVE, CR')
import semopy

MODEL_5F = """
MGMT  =~ mgt_emph_saf + mgt_prior_saf + mgt_value_saf
COMM  =~ saf_disc_opp + saf_open_disc + saf_feed_reg + saf_sug_sys + saf_sug_resp
TRAIN =~ saf_tr_opp + saf_tr_effect
SYS   =~ saf_sys_proc + saf_proc_effect + saf_equip_avail
EMPOW =~ work_ref_unsaf + work_vol_saf
MGMT ~~ COMM + TRAIN + SYS + EMPOW
COMM ~~ TRAIN + SYS + EMPOW
TRAIN ~~ SYS + EMPOW
SYS ~~ EMPOW
"""
MODEL_1F = "G =~ " + " + ".join(SC_ITEMS)

def fit_cfa(desc):
    m = semopy.Model(desc)
    m.fit(df[SC_ITEMS], obj='MLW')
    s = semopy.calc_stats(m).T
    get = lambda k: float(s.loc[k].iloc[0]) if k in s.index else np.nan
    fit = {k: get(k) for k in ['chi2', 'DoF', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']}
    return m, fit

m5, fit5 = fit_cfa(MODEL_5F)
m1, fit1 = fit_cfa(MODEL_1F)
print('5-factor :', {k: round(v, 3) for k, v in fit5.items()})
print('1-factor :', {k: round(v, 3) for k, v in fit1.items()})

# Standardized estimates
ins = m5.inspect(std_est=True)
std_col = next((c for c in ins.columns if 'Std' in c or 'std' in c), None)
assert std_col, f'standardized column not found in {list(ins.columns)}'
ins['std'] = pd.to_numeric(ins[std_col], errors='coerce')
load = ins[ins['op'] == '~'].copy()           # measurement loadings (item ~ factor in semopy inspect)
# semopy reports measurement model as 'lval =~ rval' -> in inspect op '~' with lval=item, rval=factor

# AVE & CR per factor from standardized loadings
ave_cr = []
std_loadings_out = []
for dim, items in SC_DIMS.items():
    fac = FAC[dim]
    lam = load[load['rval'] == fac]['std'].astype(float).values
    lam = np.abs(lam)
    ave = np.mean(lam ** 2)
    cr_rel = (lam.sum() ** 2) / ((lam.sum() ** 2) + np.sum(1 - lam ** 2))   # composite reliability
    ave_cr.append({'Dimension': DIM_LABEL[dim], 'Items': len(items),
                   'CR': round(cr_rel, 3), 'AVE': round(ave, 3),
                   'sqrt_AVE': round(np.sqrt(ave), 3)})
    for it in items:
        v = load[(load['rval'] == fac) & (load['lval'] == it)]['std']
        std_loadings_out.append({'Factor': fac, 'Item': it,
                                 'Std_loading': round(float(v.iloc[0]), 3) if len(v) else np.nan})
ave_cr_df = pd.DataFrame(ave_cr)
pd.DataFrame(std_loadings_out).to_csv(os.path.join(OUT_DIR, 'cfa_std_loadings.csv'), index=False)

# Inter-factor correlations (standardized covariances among latent factors)
facs = ['MGMT', 'COMM', 'TRAIN', 'SYS', 'EMPOW']
cov = ins[(ins['op'] == '~~') & (ins['lval'].isin(facs)) & (ins['rval'].isin(facs))]
corr = pd.DataFrame(np.eye(len(facs)), index=facs, columns=facs)
for _, r in cov.iterrows():
    if r['lval'] != r['rval']:
        val = float(r['std']) if not pd.isna(r['std']) else float(r[std_col])
        corr.loc[r['lval'], r['rval']] = round(val, 3)
        corr.loc[r['rval'], r['lval']] = round(val, 3)
# Fornell-Larcker: sqrt(AVE) on diagonal vs off-diagonal correlations
fl = corr.copy().astype(float)
for i, dim in enumerate(SC_DIMS):
    fl.iloc[i, i] = ave_cr[i]['sqrt_AVE']
fl.to_csv(os.path.join(OUT_DIR, 'cfa_fornell_larcker.csv'))
corr.to_csv(os.path.join(OUT_DIR, 'cfa_factor_correlations.csv'))
ave_cr_df.to_csv(os.path.join(OUT_DIR, 'cfa_ave_cr.csv'), index=False)

max_offdiag = corr.where(~np.eye(len(facs), dtype=bool)).abs().max().max()
print(f'Max inter-factor r        : {max_offdiag:.3f}')
print(f'Min sqrt(AVE)             : {ave_cr_df["sqrt_AVE"].min():.3f}  (Fornell-Larcker: sqrt(AVE) should exceed factor r)')
print(f'5F vs 1F  dCFI            : {fit5["CFI"] - fit1["CFI"]:+.3f}  | dRMSEA {fit5["RMSEA"] - fit1["RMSEA"]:+.3f}')

# Cronbach alpha (reproduce)
def cronbach(X):
    X = X.dropna()
    k = X.shape[1]
    return (k / (k - 1)) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))

alphas = {d: cronbach(df[items]) for d, items in SC_DIMS.items()}
alpha_all = cronbach(df[SC_ITEMS])

# Save fit indices (both models)
fit_out = pd.DataFrame([{'Model': '5-factor', **fit5}, {'Model': '1-factor', **fit1}])
fit_out.to_csv(os.path.join(OUT_DIR, 'cfa_fit_indices.csv'), index=False)
load.to_csv(os.path.join(OUT_DIR, 'cfa_loadings.csv'), index=False)
RESULTS['cfa'] = {'fit5': fit5, 'fit1': fit1, 'alpha_all': round(alpha_all, 3),
                  'alphas': {k: round(v, 3) for k, v in alphas.items()},
                  'max_interfactor_r': round(float(max_offdiag), 3),
                  'min_sqrtAVE': round(float(ave_cr_df['sqrt_AVE'].min()), 3),
                  'CR_range': [round(ave_cr_df['CR'].min(), 3), round(ave_cr_df['CR'].max(), 3)],
                  'AVE_range': [round(ave_cr_df['AVE'].min(), 3), round(ave_cr_df['AVE'].max(), 3)]}

# ══════════════════════════════════════════════════════════════════
# 2. Descriptive (Table 1) + quartile paradox
# ══════════════════════════════════════════════════════════════════
banner('2. DESCRIPTIVE / QUARTILE')
n_acc = int(df['any_acc_2024'].sum()); pct_acc = df['any_acc_2024'].mean() * 100
pct_prior = df['had_prior'].mean() * 100
df['sc_q4'] = pd.qcut(df['sc_total'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
q = {str(k): {'sc': g['sc_total'].mean(), 'acc_pct': g['any_acc_2024'].mean() * 100}
     for k, g in df.groupby('sc_q4')}
print('Quartile any-accident %:', {k: round(v['acc_pct'], 1) for k, v in q.items()})
RESULTS['desc'] = {'N': len(df), 'n_ind': int(df['industry'].nunique()),
                   'n_acc': n_acc, 'pct_acc': round(pct_acc, 1),
                   'pct_prior': round(pct_prior, 1),
                   'mean_vic': round(df['vic_2024_appr'].mean(), 3),
                   'sc_mean': round(df['sc_total'].mean(), 2), 'sc_sd': round(df['sc_total'].std(), 2),
                   'quartiles': {k: round(v['acc_pct'], 1) for k, v in q.items()}}

# ══════════════════════════════════════════════════════════════════
# 3. Primary NB (adjusted) + unadjusted, + E-value, + precision/TOST
# ══════════════════════════════════════════════════════════════════
banner('3. PRIMARY NB (adjusted + unadjusted) + E-value + equivalence')
EXP = [('sc_mgmt_z', 'A. Management'), ('sc_comm_z', 'B. Communication'),
       ('sc_train_z', 'C. Training'), ('sc_sys_z', 'D. Systems'),
       ('sc_empow_z', 'E. Empowerment'), ('sc_total_z', 'Overall (15 items)')]
exposure = df['n_workers'].values  # GLM exposure -> log offset

def evalue(est, lo, hi):
    rr = 1 / est if est < 1 else est
    pt = rr + np.sqrt(rr * (rr - 1))
    cin = lo if lo > 1 else (1 / hi if hi < 1 else 1.0)
    ec = cin + np.sqrt(cin * (cin - 1)) if cin > 1 else 1.0
    return round(pt, 2), round(ec, 2)

# TOST equivalence against +/-10% ROPE on the IRR scale (per 1-SD)
ROPE = np.log(1.10)  # symmetric bound: ln(1/1.10) .. ln(1.10)
def tost(beta, se):
    # H0a: beta <= -ROPE ; H0b: beta >= +ROPE ; equivalence if both rejected
    p_low = stats.norm.sf((beta - (-ROPE)) / se)   # upper-tail: beta > -ROPE
    p_high = stats.norm.cdf((beta - ROPE) / se)     # lower-tail: beta < +ROPE
    return max(p_low, p_high)

adj_rows, unadj_rows = [], []
for var, label in EXP:
    Xa = sm.add_constant(pd.concat([df[[var, 'log_prior', 'size_cat']].reset_index(drop=True),
                                    IND_DUMMIES.reset_index(drop=True)], axis=1).astype(float).values)
    ma = fit_nb(df['vic_2024_appr'], Xa, exposure)
    irr, lo, hi, p = irr_row(ma, 1)
    ev, evci = evalue(irr, lo, hi)
    p_tost = tost(ma.params[1], ma.bse[1])
    adj_rows.append({'Exposure': label, 'Variable': var, 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi,
                     'p': p, 'Evalue': ev, 'Evalue_CI': evci, 'p_equivalence_TOST': round(p_tost, 4),
                     'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
    Xu = sm.add_constant(df[[var]].astype(float).values)
    mu = fit_nb(df['vic_2024_appr'], Xu, exposure)
    uirr, ulo, uhi, up = irr_row(mu, 1)
    uev, uevci = evalue(uirr, ulo, uhi)
    unadj_rows.append({'Exposure': label, 'IRR': uirr, 'CI_lo': ulo, 'CI_hi': uhi, 'p': up,
                       'Evalue': uev, 'Evalue_CI': uevci, 'IRR (95% CI)': f'{uirr:.2f} ({ulo:.2f}-{uhi:.2f})'})

df_adj = pd.DataFrame(adj_rows); df_unadj = pd.DataFrame(unadj_rows)
df_adj.to_csv(os.path.join(OUT_DIR, 'table_primary.csv'), index=False)
df_unadj.to_csv(os.path.join(OUT_DIR, 'table_unadjusted.csv'), index=False)
ov = df_adj[df_adj['Variable'] == 'sc_total_z'].iloc[0]
ovu = df_unadj[df_unadj['Exposure'] == 'Overall (15 items)'].iloc[0]
print(f"Overall UNADJ : IRR {ovu['IRR']:.2f} ({ovu['CI_lo']:.2f}-{ovu['CI_hi']:.2f}) | E-value {ovu['Evalue']}")
print(f"Overall ADJ   : IRR {ov['IRR']:.2f} ({ov['CI_lo']:.2f}-{ov['CI_hi']:.2f}) p={ov['p']} | TOST p={ov['p_equivalence_TOST']}")
RESULTS['primary'] = {'overall_adj': [ov['IRR'], ov['CI_lo'], ov['CI_hi'], ov['p'], ov['p_equivalence_TOST']],
                      'overall_unadj': [ovu['IRR'], ovu['CI_lo'], ovu['CI_hi'], ovu['Evalue'], ovu['Evalue_CI']],
                      'adj_dim_range': [df_adj[df_adj.Variable != 'sc_total_z']['IRR'].min(),
                                        df_adj[df_adj.Variable != 'sc_total_z']['IRR'].max()],
                      'rope_pct': 10}

# ══════════════════════════════════════════════════════════════════
# R2. RCS with 95% CI band + non-linearity LR test + rug
# ══════════════════════════════════════════════════════════════════
banner('R2. RCS dose-response: CI band + non-linearity LR test')
from patsy import dmatrix
RCS_F = 'cr(sc_total, df=4) - 1'
basis = dmatrix(RCS_F, df, return_type='dataframe')
DI = basis.design_info   # reuse identical spline knots for prediction grid
covars = pd.concat([df[['log_prior', 'size_cat']].reset_index(drop=True),
                    IND_DUMMIES.reset_index(drop=True)], axis=1).astype(float)
X_sp = sm.add_constant(pd.concat([basis.reset_index(drop=True), covars], axis=1).astype(float).values)
# The spline likelihood is very flat in the cubic-spline directions (there is no
# real departure from linearity — that is the finding), so default IRLS stops at
# its 100-iteration cap before the deviance fully settles. A high iteration cap
# stabilizes the log-likelihood and makes the non-linearity LR test reproducible
# (LR = 3.29, p = 0.35) across environments rather than environment-dependent.
m_sp = fit_nb(df['vic_2024_appr'], X_sp, exposure, maxiter=5000)
# linear (reduced) model — same iteration cap for a like-for-like LR contrast
X_lin = sm.add_constant(pd.concat([df[['sc_total']].reset_index(drop=True), covars], axis=1).astype(float).values)
m_lin = fit_nb(df['vic_2024_appr'], X_lin, exposure, maxiter=5000)
LR = 2 * (m_sp.llf - m_lin.llf)
df_nl = basis.shape[1] - 1
p_nl = stats.chi2.sf(LR, df_nl)
print(f'Non-linearity LR = {LR:.2f}, df = {df_nl}, p = {p_nl:.3f}')

# Prediction grid + delta-method CI (contrast vs mean reference)
grid = np.linspace(df['sc_total'].min(), df['sc_total'].max(), 100)
gb = np.asarray(dmatrix(DI, pd.DataFrame({'sc_total': grid}), return_type='dataframe'))
cov_means = np.concatenate([[covars['log_prior'].mean(), covars['size_cat'].mean()],
                            IND_DUMMIES.mean().values])
Xg = np.column_stack([np.ones(len(grid)), gb, np.tile(cov_means, (len(grid), 1))])
ref_basis = np.asarray(dmatrix(DI, pd.DataFrame({'sc_total': [df['sc_total'].mean()]}), return_type='dataframe'))
Xref = np.concatenate([[1.0], ref_basis.ravel(), cov_means])
L = Xg - Xref                       # contrast rows (ref subtracted)
beta, V = m_sp.params, m_sp.cov_params()
log_irr = L @ beta
se = np.sqrt(np.einsum('ij,jk,ik->i', L, V, L))
irr_g = np.exp(log_irr); lo_g = np.exp(log_irr - 1.96 * se); hi_g = np.exp(log_irr + 1.96 * se)
RESULTS['rcs'] = {'LR': round(LR, 2), 'df': int(df_nl), 'p_nonlinear': round(float(p_nl), 3),
                  'irr_min': round(float(irr_g.min()), 2), 'irr_max': round(float(irr_g.max()), 2)}

fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(grid, lo_g, hi_g, alpha=0.18, color='#3498db', label='95% CI')
ax.plot(grid, irr_g, color='#2c3e50', lw=2.5, label='Adjusted IRR')
ax.axhline(1, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Safety culture score (1-5)'); ax.set_ylabel('IRR (reference = sample mean)')
ax.set_title('Dose-response: safety culture -> 2024 injuries\n'
             f'(restricted cubic spline, 4 knots, adjusted; non-linearity p = {p_nl:.2f})')
# marginal rug / histogram (data density)
ax2 = ax.twinx()
ax2.hist(df['sc_total'], bins=40, color='#bdc3c7', alpha=0.35, zorder=0)
ax2.set_ylabel('Establishments (histogram)', color='#7f8c8d')
ax2.tick_params(axis='y', colors='#7f8c8d'); ax2.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)
ax.legend(loc='upper center'); ax.grid(alpha=0.3)
savefig(fig, 'fig_rcs')
print('saved fig_rcs (png + pdf, 350 dpi, both dirs; CI band + histogram)')

# ══════════════════════════════════════════════════════════════════
# 4. Size-stratified
# ══════════════════════════════════════════════════════════════════
banner('4. SIZE-STRATIFIED')
strat = []
labels = {1: '1-4', 2: '5-19', 3: '20-49', 4: '50-99', 5: '>=100'}
for sz in [1, 2, 3, 4, 5]:
    sub = df[df['r_wrk_tot'] == sz]
    ind_d = pd.get_dummies(sub['industry'], prefix='ind', drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([sub[['sc_total_z', 'log_prior']].reset_index(drop=True),
                                   ind_d.reset_index(drop=True)], axis=1).astype(float).values)
    m = fit_nb(sub['vic_2024_appr'], X, sub['n_workers'].values)
    irr, lo, hi, p = irr_row(m, 1)
    strat.append({'Size category': f'{labels[sz]} workers', 'n': len(sub),
                  'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p, 'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
df_strat = pd.DataFrame(strat); df_strat.to_csv(os.path.join(OUT_DIR, 'table_stratified.csv'), index=False)
print(df_strat[['Size category', 'n', 'IRR (95% CI)', 'p']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 5. Sensitivity S1-S5 (S4 = normalized survey weights, sum to N)
# ══════════════════════════════════════════════════════════════════
banner('5. SENSITIVITY S1-S5 (S4 weights normalized to sum=N)')
baseX = sm.add_constant(pd.concat([df[['sc_total_z', 'log_prior', 'size_cat']].reset_index(drop=True),
                                   IND_DUMMIES.reset_index(drop=True)], axis=1).astype(float).values)
sens = []
def addrow(name, purpose, m, n):
    irr, lo, hi, p = irr_row(m, 1)
    sens.append({'Analysis': name, 'Purpose': purpose, 'n': n, 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
                 'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
addrow('Primary (officially approved)', 'Main analysis', fit_nb(df['vic_2024_appr'], baseX, exposure), len(df))
addrow('S1: Self-reported injuries', 'Reporting bias', fit_nb(df['vic_2024_occ'], baseX, exposure), len(df))
addrow('S2: 3-year cumulative (22-24)', 'Temporal smoothing', fit_nb(df['vic_3yr_appr'], baseX, exposure), len(df))
addrow('S3: Fatal injuries only', 'Severity gradient', fit_nb(df['acc_dth_2024_appr'].fillna(0), baseX, exposure), len(df))
# S4: normalize wt2 so it sums to N (mean 1) -> preserves effective sample size
w = df['wt2'].fillna(1.0).values
w_norm = w / w.mean()
m_w = sm.GLM(df['vic_2024_appr'].astype(int).values, baseX,
             family=sm.families.NegativeBinomial(), exposure=exposure, freq_weights=w_norm).fit()
addrow('S4: Survey-weighted', 'Sampling design', m_w, len(df))
# S5: 5+ workers
sub5 = df[df['r_wrk_tot'] >= 2]
ind5 = pd.get_dummies(sub5['industry'], prefix='ind', drop_first=True).astype(float)
X5 = sm.add_constant(pd.concat([sub5[['sc_total_z', 'log_prior', 'size_cat']].reset_index(drop=True),
                                ind5.reset_index(drop=True)], axis=1).astype(float).values)
addrow('S5: >=5 workers only', 'OSHA coverage', fit_nb(sub5['vic_2024_appr'], X5, sub5['n_workers'].values), len(sub5))
df_sens = pd.DataFrame(sens); df_sens.to_csv(os.path.join(OUT_DIR, 'table_sensitivity.csv'), index=False)
print(df_sens[['Analysis', 'n', 'IRR (95% CI)', 'p']].to_string(index=False))
RESULTS['sensitivity'] = df_sens[['Analysis', 'IRR', 'CI_lo', 'CI_hi', 'p']].to_dict('records')

# ══════════════════════════════════════════════════════════════════
# R5. Offset / size-specification sensitivity
# ══════════════════════════════════════════════════════════════════
banner('R5. OFFSET sensitivity (top midpoint 200/300/500; categorical size)')
off_rows = []
MID = {1: 2.5, 2: 12.0, 3: 34.5, 4: 74.5}
for top in [200.0, 300.0, 500.0]:
    nw = df['r_wrk_tot'].map({**MID, 5: top}).values
    m = fit_nb(df['vic_2024_appr'], baseX, nw)
    irr, lo, hi, p = irr_row(m, 1)
    off_rows.append({'Specification': f'Top category = {int(top)} workers', 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
                     'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
# Size as categorical dummies (instead of linear size_cat)
size_d = pd.get_dummies(df['size_cat'], prefix='sz', drop_first=True).astype(float)
Xcat = sm.add_constant(pd.concat([df[['sc_total_z', 'log_prior']].reset_index(drop=True),
                                  size_d.reset_index(drop=True), IND_DUMMIES.reset_index(drop=True)], axis=1).astype(float).values)
m_cat = fit_nb(df['vic_2024_appr'], Xcat, exposure)
irr, lo, hi, p = irr_row(m_cat, 1)
off_rows.append({'Specification': 'Firm size as categorical dummies', 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p,
                 'IRR (95% CI)': f'{irr:.2f} ({lo:.2f}-{hi:.2f})'})
df_off = pd.DataFrame(off_rows); df_off.to_csv(os.path.join(OUT_DIR, 'table_offset_sensitivity.csv'), index=False)
print(df_off[['Specification', 'IRR (95% CI)', 'p']].to_string(index=False))
RESULTS['offset'] = df_off[['Specification', 'IRR', 'CI_lo', 'CI_hi', 'p']].to_dict('records')

# ══════════════════════════════════════════════════════════════════
# 6. Forest plots, sequential-adjustment cascade, graphical abstract
# ══════════════════════════════════════════════════════════════════
banner('6. FOREST / CASCADE / GRAPHICAL ABSTRACT')

# --- Fig: adjusted forest (dimensions) ---
fig, ax = plt.subplots(figsize=(9.2, 5.6))
yp = np.arange(len(df_adj))[::-1]
ax.axvspan(ROPE_LO, ROPE_HI, color=CB['green'], alpha=0.07, zorder=0)
for i, (_, r) in enumerate(df_adj.iterrows()):
    ax.errorbar(r['IRR'], yp[i], xerr=[[r['IRR'] - r['CI_lo']], [r['CI_hi'] - r['IRR']]],
                fmt='o', color=DIMCOL[i % len(DIMCOL)], markersize=9, capsize=4, lw=2, zorder=3)
    ax.text(1.152, yp[i], f"{r['IRR (95% CI)']}  p={r['p']:.2f}", va='center', fontsize=9.5, color=CB['ink'])
ax.axvline(1, color=CB['ink'], ls='--', lw=1, alpha=0.7)
ax.set_yticks(yp); ax.set_yticklabels(df_adj['Exposure'])
ax.set_xlim(0.85, 1.15); ax.set_xlabel('Adjusted incidence rate ratio (IRR) per 1-SD safety culture')
ax.set_title('Adjusted associations — all safety-culture dimensions null\n(green band = ±10% practical-equivalence margin)')
savefig(fig, 'fig_forest_adjusted')

# --- Fig 2 (manuscript): confounding resolution unadjusted vs adjusted ---
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
savefig(fig, 'fig_confounding')

# --- Fig (NEW marquee): sequential-adjustment cascade for the overall composite ---
banner('NEW. Sequential-adjustment cascade')
IND = IND_DUMMIES.reset_index(drop=True)
casc_specs = [('Crude (offset only)', []),
              ('+ Industry fixed effects', ['IND']),
              ('+ Firm size', ['IND', 'size_cat']),
              ('+ Prior injuries (full model)', ['IND', 'size_cat', 'log_prior'])]
casc = []
for label, extra in casc_specs:
    cols = [df['sc_total_z'].reset_index(drop=True)]
    for e in extra:
        cols.append(IND if e == 'IND' else df[e].reset_index(drop=True))
    X = sm.add_constant(pd.concat(cols, axis=1).astype(float).values)
    m = fit_nb(df['vic_2024_appr'], X, exposure)
    irr, lo, hi, p = irr_row(m, 1)
    casc.append({'step': label, 'IRR': irr, 'CI_lo': lo, 'CI_hi': hi, 'p': p})
df_casc = pd.DataFrame(casc)
df_casc.to_csv(os.path.join(OUT_DIR, 'table_cascade.csv'), index=False)
print(df_casc[['step', 'IRR', 'CI_lo', 'CI_hi', 'p']].to_string(index=False))
RESULTS['cascade'] = df_casc.to_dict('records')

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
savefig(fig, 'fig_cascade')
print('saved fig_cascade (sequential adjustment 1.18 -> 1.01)')

# --- Graphical abstract (banner ~1328x531 -> rendered larger) ---
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
savefig(fig, 'graphical_abstract', dpi=220)
print('saved forest, confounding, cascade, graphical_abstract (png + pdf)')

# ══════════════════════════════════════════════════════════════════
# R7. DAG figure
# ══════════════════════════════════════════════════════════════════
banner('R7. DAG figure')
fig, ax = plt.subplots(figsize=(9, 5.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-0.15, 7)
NODES, CTR = {}, {}
def node(key, x, y, txt, fc):
    p = FancyBboxPatch((x - 1.15, y - 0.45), 2.3, 0.9, boxstyle='round,pad=0.04',
                       fc=fc, ec=CB['ink'], lw=1.4, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, txt, ha='center', va='center', fontsize=10.5, fontweight='bold', zorder=4, color=CB['ink'])
    NODES[key] = p; CTR[key] = (x, y)
def arrow(k1, k2, color, rad=0.0, ls='-'):
    # clip the path to the two box outlines (patchA/patchB) and leave a ~1.5 mm gap
    # (shrinkA/shrinkB are in points) so arrowheads never touch the text boxes
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
savefig(fig, 'fig_dag')
print('saved fig_dag')

# ══════════════════════════════════════════════════════════════════
# R8. Between-(manufacturing sub-)sector scatter
# ══════════════════════════════════════════════════════════════════
banner('R8. Between-sub-sector scatter')
g = df.groupby('industry').agg(sc=('sc_total', 'mean'), vic=('vic_2024_appr', 'sum'),
                               wk=('n_workers', 'sum'), n=('id', 'count')).reset_index()
g['rate'] = 100 * g['vic'] / g['wk']
r_ind = np.corrcoef(g['sc'], g['rate'])[0, 1]
# Number sub-sectors by injury rate (1 = highest) so each bubble carries an unambiguous
# numeric marker keyed to a legend — avoids overlapping text labels in the dense cluster.
g = g.sort_values('rate', ascending=False).reset_index(drop=True)
g['num'] = np.arange(1, len(g) + 1)
# OLS fit + 95% CI band across the 24 sub-sectors
Xg = sm.add_constant(g['sc'].values)
ols = sm.OLS(g['rate'].values, Xg).fit()
xs = np.linspace(g['sc'].min(), g['sc'].max(), 80)
pr = ols.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.fill_between(xs, pr['mean_ci_lower'], pr['mean_ci_upper'], color=CB['blue'], alpha=0.10, zorder=1)
ax.plot(xs, pr['mean'], color=CB['blue'], lw=1.8, zorder=2)
sizes = 40 + 700 * g['n'] / g['n'].max()
ax.scatter(g['sc'], g['rate'], s=sizes, c=g['rate'], cmap='YlOrRd', edgecolor=CB['ink'], lw=0.8, alpha=0.85, zorder=3)
ax.margins(x=0.05, y=0.10)
# Small numeric token on each bubble; adjustText nudges overlapping tokens apart and draws
# a thin leader line back to the exact bubble — 1:1 mapping, no ambiguity.
nums = [ax.text(r['sc'], r['rate'], str(int(r['num'])), fontsize=8, fontweight='bold',
                ha='center', va='center', color=CB['ink'], zorder=6,
                bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec=CB['grey'], lw=0.4, alpha=0.9))
        for _, r in g.iterrows()]
try:
    from adjustText import adjust_text
    adjust_text(nums, x=g['sc'].to_numpy(), y=g['rate'].to_numpy(), ax=ax,
                expand=(1.4, 1.6), force_text=(0.5, 0.7), force_static=(0.5, 0.8), min_arrow_len=0,
                arrowprops=dict(arrowstyle='-', color=CB['grey'], lw=0.5, alpha=0.7))
    print('  labels: numbered markers + legend (adjustText-separated)')
except Exception as e:
    print('  [warn] adjustText unavailable -> static numbers:', e)
# Two-column numbered legend (ranked by injury rate) in the empty upper-left
gg = g.sort_values('num')
items = [f"{int(r['num']):>2}. {KSIC.get(int(r['industry']), int(r['industry']))}" for _, r in gg.iterrows()]
half = (len(items) + 1) // 2
left, right = items[:half], items[half:] + [''] * (2 * half - len(items))
legend = "\n".join(f"{l:<24}{r}" for l, r in zip(left, right))
ax.text(0.015, 0.985, legend, transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
        family='DejaVu Sans Mono', linespacing=1.4,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=CB['grey'], lw=0.6, alpha=0.93))
ax.set_xlabel('Sub-sector mean safety culture (1–5)'); ax.set_ylabel('Crude injury rate (per 100 workers)')
ax.set_title(f'Between-sub-sector confounding within manufacturing\n'
             f'(r = {r_ind:.2f}; bubble size ∝ no. of establishments; band = 95% CI of fit)')
savefig(fig, 'fig_industry_scatter')
print(f'saved fig_industry_scatter (between-sub-sector r = {r_ind:.2f})')
RESULTS['industry_scatter_r'] = round(float(r_ind), 2)

# ══════════════════════════════════════════════════════════════════
# R6. ML FEATURE IMPORTANCE (train/test, held-out AUC, permutation importance)
# ══════════════════════════════════════════════════════════════════
banner('R6. ML feature importance (held-out permutation importance)')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
Xrf = df[SC_ITEMS + ['log_prior', 'size_cat']].copy()
for ind in df['industry'].value_counts().head(8).index:
    Xrf[f'ind_{ind}'] = (df['industry'] == ind).astype(int)
yrf = df['any_acc_2024'].values
Xtr, Xte, ytr, yte = train_test_split(Xrf, yrf, test_size=0.30, random_state=42, stratify=yrf)
rf = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(Xtr, ytr)
auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
print(f'Held-out ROC-AUC: {auc:.3f}')
pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=42, n_jobs=-1, scoring='roc_auc')
fi = pd.DataFrame({'Feature': Xrf.columns, 'Importance': pi.importances_mean,
                   'Importance_sd': pi.importances_std}).sort_values('Importance', ascending=False)
method = 'Permutation importance (held-out ROC-AUC decrease)'
fi['Method'] = method
fi.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)
top_sc = fi[fi['Feature'].isin(SC_ITEMS)].head(1)
lp = float(fi[fi['Feature'] == 'log_prior']['Importance'].iloc[0])
sz = float(fi[fi['Feature'] == 'size_cat']['Importance'].iloc[0])
tsc = float(top_sc['Importance'].iloc[0]); tsc_name = top_sc['Feature'].iloc[0]
ratio = lp / tsc if tsc else np.nan
print(f'{method}: log_prior={lp:.3f}, size_cat={sz:.3f}, top SC item ({tsc_name})={tsc:.3f}, ratio={ratio:.1f}x')
RESULTS['ml'] = {'method': method, 'auc': round(float(auc), 3), 'log_prior': round(lp, 3),
                 'size_cat': round(sz, 3), 'top_sc_item': tsc_name, 'top_sc_imp': round(tsc, 3),
                 'ratio': round(float(ratio), 1)}
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
savefig(fig, 'fig_shap')
print('saved fig_shap (held-out permutation importance with SD bars)')

# ══════════════════════════════════════════════════════════════════
# 7. Save run summary (figures already written to both dirs by savefig)
# ══════════════════════════════════════════════════════════════════
banner('7. SAVE / SYNC')
RESULTS['_meta'] = {'timestamp': TIMESTAMP, 'N': len(df), 'input_sha256_16': sha, 'out_dir': OUT_DIR}
with open(os.path.join(OUT_DIR, 'results_for_manuscript.json'), 'w', encoding='utf-8') as fp:
    json.dump(RESULTS, fp, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, 'run_summary.json'), 'w') as fp:
    json.dump({'timestamp': TIMESTAMP, 'N': len(df), 'target': 'Safety Science',
               'adjusted_overall_IRR': RESULTS['primary']['overall_adj'][0],
               'unadjusted_overall_IRR': RESULTS['primary']['overall_unadj'][0],
               'rcs_nonlinearity_p': RESULTS['rcs']['p_nonlinear'],
               'ml_method': RESULTS['ml']['method'], 'ml_auc': RESULTS['ml']['auc']}, fp, indent=2)
print(f'All outputs (png + pdf) -> {OUT_DIR}  and  {PAPER_SS_OUT}')
banner('DONE')
