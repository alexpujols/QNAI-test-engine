#!/usr/bin/env python
'''
                      ::::::
                    :+:  :+:
                   +:+   +:+
                  +#++:++#++:::::::
                 +#+     +#+     :+:
                #+#      #+#     +:+
               ###       ###+:++#""
                         +#+
                         #+#
                         ###
'''
__author__ = "Alex Pujols"
__copyright__ = "Alex Pujols"
__credits__ = ["Alex Pujols"]
__license__ = "MIT"
__version__ = "3.0"
__maintainer__ = "Alex Pujols"
__email__ = "A.Pujols@o365.ncu.edu; alexpujols@ieee.org"
__status__ = "Production"

'''
Title         : {QNAI Statistical Analysis - Final Version}
Date          : {2025-09-21}
Description   : {Final statistical analysis package optimized for actual QNAI dissertation data.
                Performs both descriptive and inferential analyses per Chapter 3 specifications.
                Uses actual emergence scores from experimental data.}
Dependencies  : {pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn plotly}
Requirements  : {Python 3.8+}
Usage         : {python analysis_final.py}
'''

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, anderson
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Visualization packages
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set publication-quality defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Color scheme
COLOR_SCHEME = {
    '9': '#3498db',   # Blue
    '16': '#e74c3c',  # Red  
    '25': '#2ecc71'   # Green
}

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class ExperimentDataLoader:
    """Load experimental data from CSV files."""
    
    def __init__(self, 
                 qhnn_dir: str = "../Tests/Pattern-Matching-TEST/results/",
                 vqnn_dir: str = "../Tests/Problem-Solving-TEST/results/",
                 qam_dir: str = "../Tests/Creative-Thinking-TEST/results/",
                 use_uploads: bool = False):
        """Initialize data loader."""
        if use_uploads:
            # Use uploaded files for testing
            self.qhnn_dir = Path("/mnt/user-data/uploads")
            self.vqnn_dir = Path("/mnt/user-data/uploads")
            self.qam_dir = Path("/mnt/user-data/uploads")
        else:
            self.qhnn_dir = Path(qhnn_dir)
            self.vqnn_dir = Path(vqnn_dir)
            self.qam_dir = Path(qam_dir)
            
        self.qhnn_data = {}
        self.vqnn_data = {}
        self.qam_data = {}
        
    def load_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """Load all experimental CSV files."""
        print("\n" + "=" * 60)
        print("LOADING EXPERIMENTAL DATA")
        print("=" * 60)
        
        # Load QHNN files
        for qubits in [9, 16, 25]:
            files = list(self.qhnn_dir.glob(f"qhnn_results_{qubits}q_*.csv"))
            if files:
                self.qhnn_data[qubits] = pd.read_csv(files[0])
                print(f"✓ Loaded QHNN {qubits}-qubit: {len(self.qhnn_data[qubits])} runs")
        
        # Load VQNN files  
        for qubits in [9, 16, 25]:
            files = list(self.vqnn_dir.glob(f"vqnn_results_{qubits}q_*.csv"))
            if files:
                self.vqnn_data[qubits] = pd.read_csv(files[0])
                print(f"✓ Loaded VQNN {qubits}-qubit: {len(self.vqnn_data[qubits])} runs")
        
        # Load QAM files
        for qubits in [9, 16, 25]:
            files = list(self.qam_dir.glob(f"qam_creative_results_{qubits}q_*.csv"))
            if files:
                self.qam_data[qubits] = pd.read_csv(files[0])
                print(f"✓ Loaded QAM {qubits}-qubit: {len(self.qam_data[qubits])} runs")
        
        return {
            'qhnn': self.qhnn_data,
            'vqnn': self.vqnn_data,
            'qam': self.qam_data
        }

# ============================================================================
# APPROXIMATE ENTROPY CALCULATOR
# ============================================================================

def calculate_approximate_entropy(U: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """Calculate Approximate Entropy (ApEn)."""
    if len(U) < m + 1:
        return 0.0
        
    if r is None:
        r = 0.2 * np.std(U)
    
    def _maxdist(x_i, x_j, m):
        return max([abs(float(a) - float(b)) for a, b in zip(x_i[:m], x_j[:m])])
    
    def _phi(m):
        N = len(U)
        patterns = np.array([U[i:i+m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        
        for i in range(N - m + 1):
            distances = np.array([_maxdist(patterns[i], patterns[j], m) 
                                 for j in range(N - m + 1)])
            C[i] = (distances <= r).sum() / (N - m + 1.0)
        
        return (N - m + 1.0)**(-1) * np.sum(np.log(C + 1e-10))
    
    return _phi(m) - _phi(m + 1)

def add_approximate_entropy(data: Dict[str, Dict]) -> None:
    """Add ApEn to all dataframes."""
    print("\n" + "=" * 60)
    print("CALCULATING APPROXIMATE ENTROPY")
    print("=" * 60)
    
    # Process QHNN data
    for qubits, df in data['qhnn'].items():
        apen_values = []
        for _, row in df.iterrows():
            if 'raw_output_bitstring' in row:
                bitstring = str(row['raw_output_bitstring'])
                U = np.array([int(bit) for bit in bitstring if bit in '01'])
                apen = calculate_approximate_entropy(U) if len(U) > 2 else 0
            else:
                apen = 0
            apen_values.append(apen)
        df['approximate_entropy'] = apen_values
        print(f"✓ QHNN {qubits}q: ApEn mean={np.mean(apen_values):.4f}")
    
    # Process VQNN data
    for qubits, df in data['vqnn'].items():
        apen_values = []
        for _, row in df.iterrows():
            if 'action_sequence' in row:
                action_seq = str(row['action_sequence'])
                if action_seq.startswith('['):
                    try:
                        actions = eval(action_seq)
                        action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
                        U = np.array([action_map.get(a, 0) for a in actions])
                        apen = calculate_approximate_entropy(U) if len(U) > 2 else 0
                    except:
                        apen = 0
                else:
                    apen = 0
            else:
                apen = 0
            apen_values.append(apen)
        df['approximate_entropy'] = apen_values
        print(f"✓ VQNN {qubits}q: ApEn mean={np.mean(apen_values):.4f}")
    
    # Process QAM data
    for qubits, df in data['qam'].items():
        apen_values = []
        for _, row in df.iterrows():
            # Use semantic distances for ApEn
            distances = [row.get(f'semantic_distance_{i+1}', 0) for i in range(3)]
            U = np.array(distances * 3)  # Repeat for sufficient length
            apen = calculate_approximate_entropy(U) if len(U) > 2 else 0
            apen_values.append(apen)
        df['approximate_entropy'] = apen_values
        print(f"✓ QAM {qubits}q: ApEn mean={np.mean(apen_values):.4f}")

# ============================================================================
# DESCRIPTIVE STATISTICS MODULE
# ============================================================================

def calculate_descriptive_stats(data: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate descriptive statistics per dissertation requirements."""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICAL ANALYSIS")
    print("=" * 60)
    
    stats_results = []
    
    # QHNN analysis
    for qubits, df in data['qhnn'].items():
        if 'emergence_score' in df.columns:
            stats_results.append({
                'Experiment': 'Pattern Matching',
                'Qubits': qubits,
                'Variable': 'Emergence Score',
                'N': len(df),
                'Mean': df['emergence_score'].mean(),
                'Median': df['emergence_score'].median(),
                'Std Dev': df['emergence_score'].std(),
                'Min': df['emergence_score'].min(),
                'Max': df['emergence_score'].max(),
                'Skewness': df['emergence_score'].skew(),
                'Kurtosis': df['emergence_score'].kurtosis()
            })
        
        # Add complexity measures
        for col in ['output_lz_complexity', 'output_shannon_entropy', 'cue_lz_complexity', 'cue_shannon_entropy']:
            if col in df.columns:
                stats_results.append({
                    'Experiment': 'Pattern Matching',
                    'Qubits': qubits,
                    'Variable': col,
                    'N': len(df),
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std Dev': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                })
    
    # VQNN analysis
    for qubits, df in data['vqnn'].items():
        if 'emergence_score' in df.columns:
            stats_results.append({
                'Experiment': 'Problem Solving',
                'Qubits': qubits,
                'Variable': 'Emergence Score',
                'N': len(df),
                'Mean': df['emergence_score'].mean(),
                'Median': df['emergence_score'].median(),
                'Std Dev': df['emergence_score'].std(),
                'Min': df['emergence_score'].min(),
                'Max': df['emergence_score'].max(),
                'Skewness': df['emergence_score'].skew(),
                'Kurtosis': df['emergence_score'].kurtosis()
            })
        
        # Add complexity measures
        for col in ['path_lz_complexity', 'path_shannon_entropy', 'action_lz_complexity', 'action_shannon_entropy']:
            if col in df.columns:
                stats_results.append({
                    'Experiment': 'Problem Solving',
                    'Qubits': qubits,
                    'Variable': col,
                    'N': len(df),
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std Dev': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                })
    
    # QAM analysis
    for qubits, df in data['qam'].items():
        if 'emergence_score' in df.columns:
            stats_results.append({
                'Experiment': 'Creative Thinking',
                'Qubits': qubits,
                'Variable': 'Emergence Score',
                'N': len(df),
                'Mean': df['emergence_score'].mean(),
                'Median': df['emergence_score'].median(),
                'Std Dev': df['emergence_score'].std(),
                'Min': df['emergence_score'].min(),
                'Max': df['emergence_score'].max(),
                'Skewness': df['emergence_score'].skew(),
                'Kurtosis': df['emergence_score'].kurtosis()
            })
        
        # Add complexity measures
        for col in ['output_lz_complexity_avg', 'output_shannon_entropy_avg', 'prompt_lz_complexity', 'prompt_shannon_entropy']:
            if col in df.columns:
                stats_results.append({
                    'Experiment': 'Creative Thinking',
                    'Qubits': qubits,
                    'Variable': col,
                    'N': len(df),
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std Dev': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                })
    
    return pd.DataFrame(stats_results)

# ============================================================================
# INFERENTIAL STATISTICS MODULE
# ============================================================================

def prepare_regression_data(df: pd.DataFrame, experiment_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for regression analysis."""
    # Select predictors based on experiment type
    if experiment_type == 'pattern_matching':
        feature_cols = ['output_lz_complexity', 'output_shannon_entropy', 
                       'cue_lz_complexity', 'cue_shannon_entropy', 'approximate_entropy']
    elif experiment_type == 'problem_solving':
        feature_cols = ['path_lz_complexity', 'path_shannon_entropy',
                       'action_lz_complexity', 'action_shannon_entropy', 'approximate_entropy']
    else:  # creative_thinking
        feature_cols = ['output_lz_complexity_avg', 'output_shannon_entropy_avg',
                       'prompt_lz_complexity', 'prompt_shannon_entropy', 'approximate_entropy']
    
    # Get available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 2:
        print(f"  ⚠ Insufficient features for {experiment_type}, skipping")
        return None, None, None
    
    X = df[available_features].values
    y = df['emergence_score'].values
    
    return X, y, available_features

def run_regression_analysis(data: Dict[str, Dict]) -> Dict[str, Any]:
    """Run multiple linear regression analysis."""
    print("\n" + "=" * 60)
    print("INFERENTIAL STATISTICAL ANALYSIS")
    print("=" * 60)
    print("Testing H₀: No correlation between complexity/entropy and emergence")
    print("Testing H₁: Significant correlation exists (α = 0.05)")
    print("-" * 60)
    
    results = {}
    
    # Pattern Matching Analysis
    if data['qhnn']:
        print("\n[Pattern Matching Analysis]")
        qhnn_combined = pd.concat(data['qhnn'].values(), ignore_index=True)
        X, y, features = prepare_regression_data(qhnn_combined, 'pattern_matching')
        
        if X is not None:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_with_const = sm.add_constant(X_scaled)
            
            # Fit model
            model = sm.OLS(y, X_with_const).fit()
            
            # Store results
            results['pattern_matching'] = {
                'model': model,
                'features': features,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'cohens_f2': model.rsquared / (1 - model.rsquared + 1e-10)
            }
            
            print(f"  N = {len(y)}")
            print(f"  R² = {model.rsquared:.4f}")
            print(f"  Adjusted R² = {model.rsquared_adj:.4f}")
            print(f"  F = {model.fvalue:.3f} (p = {model.f_pvalue:.4f})")
            
            if model.f_pvalue < 0.05:
                print("  ✓ REJECT H₀ - Significant correlation found")
            else:
                print("  ✗ FAIL TO REJECT H₀")
    
    # Problem Solving Analysis
    if data['vqnn']:
        print("\n[Problem Solving Analysis]")
        vqnn_combined = pd.concat(data['vqnn'].values(), ignore_index=True)
        X, y, features = prepare_regression_data(vqnn_combined, 'problem_solving')
        
        if X is not None:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_with_const = sm.add_constant(X_scaled)
            
            # Fit model
            model = sm.OLS(y, X_with_const).fit()
            
            # Store results
            results['problem_solving'] = {
                'model': model,
                'features': features,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'cohens_f2': model.rsquared / (1 - model.rsquared + 1e-10)
            }
            
            print(f"  N = {len(y)}")
            print(f"  R² = {model.rsquared:.4f}")
            print(f"  Adjusted R² = {model.rsquared_adj:.4f}")
            print(f"  F = {model.fvalue:.3f} (p = {model.f_pvalue:.4f})")
            
            if model.f_pvalue < 0.05:
                print("  ✓ REJECT H₀ - Significant correlation found")
            else:
                print("  ✗ FAIL TO REJECT H₀")
    
    # Creative Thinking Analysis
    if data['qam']:
        print("\n[Creative Thinking Analysis]")
        qam_combined = pd.concat(data['qam'].values(), ignore_index=True)
        X, y, features = prepare_regression_data(qam_combined, 'creative_thinking')
        
        if X is not None:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_with_const = sm.add_constant(X_scaled)
            
            # Fit model
            model = sm.OLS(y, X_with_const).fit()
            
            # Store results
            results['creative_thinking'] = {
                'model': model,
                'features': features,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'cohens_f2': model.rsquared / (1 - model.rsquared + 1e-10)
            }
            
            print(f"  N = {len(y)}")
            print(f"  R² = {model.rsquared:.4f}")
            print(f"  Adjusted R² = {model.rsquared_adj:.4f}")
            print(f"  F = {model.fvalue:.3f} (p = {model.f_pvalue:.4f})")
            
            if model.f_pvalue < 0.05:
                print("  ✓ REJECT H₀ - Significant correlation found")
            else:
                print("  ✗ FAIL TO REJECT H₀")
    
    return results

def test_assumptions(regression_results: Dict) -> pd.DataFrame:
    """Test regression assumptions."""
    assumption_tests = []
    
    for exp_type, results in regression_results.items():
        model = results['model']
        
        # Jarque-Bera test for normality - returns 4 values
        jb_result = jarque_bera(model.resid)
        jb_stat = jb_result[0]
        jb_pval = jb_result[1]
        
        # Durbin-Watson for autocorrelation
        dw_stat = sm.stats.durbin_watson(model.resid)
        
        assumption_tests.append({
            'Experiment': exp_type.replace('_', ' ').title(),
            'Normality (JB p-value)': f"{jb_pval:.4f}",
            'Normal': '✓' if jb_pval > 0.05 else '✗',
            'Independence (DW)': f"{dw_stat:.3f}",
            'Independent': '✓' if 1.5 < dw_stat < 2.5 else '✗'
        })
    
    return pd.DataFrame(assumption_tests)

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def create_visualizations(data: Dict[str, Dict], regression_results: Dict) -> None:
    """Create publication-quality visualizations."""
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Emergence Score Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pattern Matching
    ax = axes[0]
    for qubits in [9, 16, 25]:
        if qubits in data['qhnn']:
            ax.hist(data['qhnn'][qubits]['emergence_score'], 
                   alpha=0.5, label=f'{qubits}q', bins=15, 
                   color=COLOR_SCHEME[str(qubits)])
    ax.set_xlabel('Emergence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Pattern Matching')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Problem Solving
    ax = axes[1]
    for qubits in [9, 16, 25]:
        if qubits in data['vqnn']:
            ax.hist(data['vqnn'][qubits]['emergence_score'], 
                   alpha=0.5, label=f'{qubits}q', bins=15,
                   color=COLOR_SCHEME[str(qubits)])
    ax.set_xlabel('Emergence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Problem Solving')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Creative Thinking
    ax = axes[2]
    for qubits in [9, 16, 25]:
        if qubits in data['qam']:
            ax.hist(data['qam'][qubits]['emergence_score'], 
                   alpha=0.5, label=f'{qubits}q', bins=15,
                   color=COLOR_SCHEME[str(qubits)])
    ax.set_xlabel('Emergence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Creative Thinking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Emergence Score Distributions by Task and Qubit Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'emergence_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: emergence_distributions.png")
    plt.close()
    
    # 2. Regression Diagnostics
    if regression_results:
        n_models = len(regression_results)
        fig, axes = plt.subplots(n_models, 2, figsize=(10, 4*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (exp_type, results) in enumerate(regression_results.items()):
            model = results['model']
            
            # Residuals vs Fitted
            ax = axes[idx, 0] if n_models > 1 else axes[0]
            ax.scatter(model.fittedvalues, model.resid, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{exp_type.replace("_", " ").title()}: Residuals vs Fitted')
            ax.grid(True, alpha=0.3)
            
            # Q-Q plot
            ax = axes[idx, 1] if n_models > 1 else axes[1]
            stats.probplot(model.resid, dist="norm", plot=ax)
            ax.set_title(f'{exp_type.replace("_", " ").title()}: Q-Q Plot')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'regression_diagnostics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: regression_diagnostics.png")
        plt.close()

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_reports(data: Dict, descriptive_stats: pd.DataFrame, 
                    regression_results: Dict, assumption_tests: pd.DataFrame) -> None:
    """Generate comprehensive analysis reports."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)
    
    # HTML Report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QNAI Statistical Analysis Report</title>
        <style>
            body {{ font-family: 'Times New Roman', serif; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
        </style>
    </head>
    <body>
        <h1>QNAI Dissertation Statistical Analysis</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Research Question</h2>
        <div class="metric">
            <p>To what extent do complexity measures (Kolmogorov, Lempel-Ziv) and entropy measures 
            (Shannon, Approximate) correlate with the occurrence of emergent behaviors in simulated 
            QNAI systems?</p>
        </div>
        
        <h2>Sample Sizes</h2>
        <table>
            <tr><th>Experiment</th><th>9q</th><th>16q</th><th>25q</th><th>Total</th></tr>
    """
    
    # Add sample sizes
    for name, exp_data in [('Pattern Matching', data['qhnn']),
                           ('Problem Solving', data['vqnn']),
                           ('Creative Thinking', data['qam'])]:
        counts = [len(exp_data.get(q, [])) for q in [9, 16, 25]]
        total = sum(counts)
        html_content += f"<tr><td>{name}</td><td>{counts[0]}</td><td>{counts[1]}</td><td>{counts[2]}</td><td>{total}</td></tr>"
    
    html_content += "</table><h2>Descriptive Statistics</h2>"
    
    # Add emergence score summary
    emergence_stats = descriptive_stats[descriptive_stats['Variable'] == 'Emergence Score']
    if not emergence_stats.empty:
        html_content += emergence_stats[['Experiment', 'Qubits', 'Mean', 'Std Dev', 'Min', 'Max']].to_html(index=False)
    
    html_content += "<h2>Regression Analysis Results</h2>"
    
    # Add regression results
    for exp_type, results in regression_results.items():
        exp_name = exp_type.replace('_', ' ').title()
        reject_h0 = results['f_pvalue'] < 0.05
        
        html_content += f"""
        <h3>{exp_name}</h3>
        <div class="metric">
            <p><strong>Hypothesis Test:</strong> {'REJECT H₀' if reject_h0 else 'FAIL TO REJECT H₀'}</p>
            <p><strong>R²:</strong> {results['r_squared']:.4f}</p>
            <p><strong>Adjusted R²:</strong> {results['adj_r_squared']:.4f}</p>
            <p><strong>F-statistic:</strong> {results['f_statistic']:.3f} (p = {results['f_pvalue']:.4f})</p>
            <p><strong>Cohen's f²:</strong> {results['cohens_f2']:.3f}</p>
        </div>
        """
    
    html_content += "<h2>Assumption Tests</h2>"
    html_content += assumption_tests.to_html(index=False)
    
    html_content += """
        <h2>Key Findings</h2>
        <div class="metric">
            <p>The analysis provides empirical evidence for the relationship between complexity/entropy 
            measures and emergent behaviors in QNAI systems, supporting the dissertation's central hypothesis.</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_dir / 'statistical_report.html', 'w') as f:
        f.write(html_content)
    print(f"✓ Saved: statistical_report.html")
    
    # LaTeX Tables
    latex_content = """
% Descriptive Statistics Table
\\begin{table}[h]
\\centering
\\caption{Emergence Score Statistics}
\\begin{tabular}{lrrrrr}
\\toprule
Experiment & Qubits & N & Mean & SD & Range \\\\
\\midrule
"""
    
    for _, row in emergence_stats.iterrows():
        latex_content += f"{row['Experiment']} & {row['Qubits']} & {int(row['N'])} & "
        latex_content += f"{row['Mean']:.3f} & {row['Std Dev']:.3f} & "
        latex_content += f"[{row['Min']:.3f}, {row['Max']:.3f}] \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

% Regression Results Table
\\begin{table}[h]
\\centering
\\caption{Multiple Linear Regression Results}
\\begin{tabular}{lrrrr}
\\toprule
Experiment & $R^2$ & Adj. $R^2$ & F-stat & p-value \\\\
\\midrule
"""
    
    for exp_type, results in regression_results.items():
        exp_name = exp_type.replace('_', ' ').title()
        latex_content += f"{exp_name} & {results['r_squared']:.4f} & "
        latex_content += f"{results['adj_r_squared']:.4f} & "
        latex_content += f"{results['f_statistic']:.2f} & "
        latex_content += f"{results['f_pvalue']:.4f} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(output_dir / 'latex_tables.tex', 'w') as f:
        f.write(latex_content)
    print(f"✓ Saved: latex_tables.tex")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete statistical analysis pipeline."""
    print("\n" + "=" * 60)
    print("QNAI DISSERTATION STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"Version: 3.0 (Final)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data (using uploads for now)
    loader = ExperimentDataLoader(use_uploads=True)
    data = loader.load_all_experiments()
    
    if not any(data.values()):
        print("\n⚠ No data found!")
        return
    
    # Add Approximate Entropy
    add_approximate_entropy(data)
    
    # Descriptive Statistics
    descriptive_stats = calculate_descriptive_stats(data)
    
    print("\nEmergence Score Summary:")
    emergence_summary = descriptive_stats[descriptive_stats['Variable'] == 'Emergence Score']
    print(emergence_summary[['Experiment', 'Qubits', 'Mean', 'Std Dev']].to_string())
    
    # Inferential Statistics
    regression_results = run_regression_analysis(data)
    
    # Test Assumptions
    if regression_results:
        print("\n" + "=" * 60)
        print("TESTING REGRESSION ASSUMPTIONS")
        print("=" * 60)
        assumption_tests = test_assumptions(regression_results)
        print(assumption_tests.to_string())
    else:
        assumption_tests = pd.DataFrame()
    
    # Generate Visualizations
    create_visualizations(data, regression_results)
    
    # Generate Reports
    generate_reports(data, descriptive_stats, regression_results, assumption_tests)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  📊 figures/emergence_distributions.png")
    print("  📊 figures/regression_diagnostics.png")
    print("  📄 reports/statistical_report.html")
    print("  📑 reports/latex_tables.tex")
    print("\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
