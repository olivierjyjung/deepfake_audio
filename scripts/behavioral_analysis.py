#!/usr/bin/env python3
"""
Behavioral Data Analysis Script for Visual Priming & Deepfake Audio Perception Study
======================================================================================

This script analyzes survey response data (behavioral measures) from the experiment.

Dependencies:
    pip install pandas numpy scipy statsmodels matplotlib seaborn pingouin

Usage:
    python behavioral_analysis.py

Author: Generated for audio_ai experiment
Date: 2026-03-09
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Visualization (non-interactive backend for CLI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Note: pingouin not installed. Some advanced statistics unavailable.")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/Users/user/Desktop/audio_ai")
OUTPUT_DIR = DATA_DIR / "analysis_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Survey question mappings (Korean to English)
QUESTION_MAP = {
    'Q1': 'false_memory',          # Have you seen this idol perform this song?
    'Q2': 'false_memory_confidence',  # Confidence in Q1
    'Q3': 'audio_source_judgment',  # How was the audio created?
    'Q4': 'audio_judgment_confidence',  # Confidence in Q3
    'Q5': 'judgment_factors',      # Factors influencing judgment
    'Q6': 'likeability',           # Overall likeability
    'Q7': 'attribution_reason',    # Why this idol?
    'Q8': 'listening_frequency',   # Listening frequency
    'Q9': 'voice_familiarity',     # Voice familiarity
    'Q10': 'exposure_sources',     # Exposure sources
    'Q11': 'other_factors',        # Other factors (open-ended)
}

# Audio source options for Q3
AUDIO_SOURCE_OPTIONS = {
    1: 'deep_learning_ai',
    2: 'midi_synthesis',
    3: 'cover_artist_live',
    4: 'cover_artist_edited',
    5: 'real_artist_live',
    6: 'real_artist_edited',
}

# Experimental groups
GROUPS = {
    'A': 'Visual Priming',
    'B': 'Control (No Priming)',
}


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_behavioral_data(file_path: Path) -> pd.DataFrame:
    """
    Load behavioral data from CSV/Excel file.

    Args:
        file_path: Path to data file

    Returns:
        DataFrame with behavioral data
    """
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return df


def create_sample_data() -> pd.DataFrame:
    """
    Create sample behavioral data for demonstration.
    Based on experimental design: N=56, 2 groups.

    Returns:
        DataFrame with simulated behavioral data
    """
    np.random.seed(42)
    n_subjects = 56

    # Assign groups (roughly equal)
    groups = np.array(['A'] * 28 + ['B'] * 28)
    np.random.shuffle(groups)

    # Simulate data based on expected priming effects
    data = {
        'subject_id': [f'sub-{i+1:03d}' for i in range(n_subjects)],
        'group': groups,
        'idol': np.random.choice(['Hanni', 'Jennie'], n_subjects),
        'audio_type': np.random.choice(['real', 'deepfake'], n_subjects),
    }

    # Q1: False memory (binary) - expect higher 'Yes' in Group A
    # Group A: ~60% false memory, Group B: ~30%
    q1 = []
    for g in groups:
        if g == 'A':
            q1.append(np.random.choice([1, 0], p=[0.6, 0.4]))
        else:
            q1.append(np.random.choice([1, 0], p=[0.3, 0.7]))
    data['Q1_false_memory'] = q1

    # Q2: Confidence in Q1 (1-5)
    data['Q2_confidence'] = np.random.randint(1, 6, n_subjects)

    # Q3: Audio source judgment
    # Correct: AI options (1,2) for deepfake, Real options (5,6) for real
    q3 = []
    for i, audio in enumerate(data['audio_type']):
        if audio == 'deepfake':
            # Harder to detect for Group A (primed)
            if groups[i] == 'A':
                q3.append(np.random.choice([1, 2, 5, 6], p=[0.3, 0.1, 0.35, 0.25]))
            else:
                q3.append(np.random.choice([1, 2, 5, 6], p=[0.4, 0.15, 0.25, 0.2]))
        else:
            # Real audio - generally easier
            q3.append(np.random.choice([1, 2, 5, 6], p=[0.1, 0.05, 0.5, 0.35]))
    data['Q3_audio_judgment'] = q3

    # Q4: Confidence in Q3 (1-5)
    data['Q4_confidence'] = np.random.randint(1, 6, n_subjects)

    # Q6: Likeability (1-5)
    data['Q6_likeability'] = np.random.randint(2, 6, n_subjects)

    # Q8: Listening frequency (1-5)
    data['Q8_listening_freq'] = np.random.randint(1, 6, n_subjects)

    # Q9: Voice familiarity (1-5)
    data['Q9_voice_familiarity'] = np.random.randint(1, 6, n_subjects)

    return pd.DataFrame(data)


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deepfake detection accuracy.

    Args:
        df: DataFrame with behavioral data

    Returns:
        DataFrame with accuracy column added
    """
    df = df.copy()

    # Correct detection: AI judgment (1,2) for deepfake, Real judgment (5,6) for real
    def is_correct(row):
        if row['audio_type'] == 'deepfake':
            return row['Q3_audio_judgment'] in [1, 2]
        else:
            return row['Q3_audio_judgment'] in [5, 6]

    df['detection_correct'] = df.apply(is_correct, axis=1).astype(int)

    return df


# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def analyze_false_memory(df: pd.DataFrame) -> Dict:
    """
    Analyze false memory induction between groups.

    Uses chi-square test for independence.

    Args:
        df: DataFrame with behavioral data

    Returns:
        Dictionary with statistical results
    """
    # Create contingency table
    contingency = pd.crosstab(df['group'], df['Q1_false_memory'])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Effect size (Cramér's V)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    # Proportions by group
    props = df.groupby('group')['Q1_false_memory'].mean()

    results = {
        'test': 'Chi-square test of independence',
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'group_A_proportion': props.get('A', 0),
        'group_B_proportion': props.get('B', 0),
        'contingency_table': contingency,
    }

    return results


def analyze_detection_accuracy(df: pd.DataFrame) -> Dict:
    """
    Analyze deepfake detection accuracy between groups.

    Args:
        df: DataFrame with behavioral data

    Returns:
        Dictionary with statistical results
    """
    # Accuracy by group
    accuracy_by_group = df.groupby('group')['detection_correct'].agg(['mean', 'std', 'count'])

    # Independent samples t-test
    group_a = df[df['group'] == 'A']['detection_correct']
    group_b = df[df['group'] == 'B']['detection_correct']

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(group_a) - 1) * group_a.var() + (len(group_b) - 1) * group_b.var()) /
        (len(group_a) + len(group_b) - 2)
    )
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0

    results = {
        'test': 'Independent samples t-test',
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'group_A_accuracy': accuracy_by_group.loc['A', 'mean'] if 'A' in accuracy_by_group.index else 0,
        'group_B_accuracy': accuracy_by_group.loc['B', 'mean'] if 'B' in accuracy_by_group.index else 0,
        'accuracy_summary': accuracy_by_group,
    }

    return results


def analyze_confidence_accuracy_relationship(df: pd.DataFrame) -> Dict:
    """
    Analyze relationship between confidence and accuracy.

    Args:
        df: DataFrame with behavioral data

    Returns:
        Dictionary with correlation results
    """
    # Pearson correlation
    r, p_value = stats.pearsonr(df['Q4_confidence'], df['detection_correct'])

    # Point-biserial correlation (more appropriate for binary outcome)
    r_pb, p_pb = stats.pointbiserialr(df['detection_correct'], df['Q4_confidence'])

    results = {
        'pearson_r': r,
        'pearson_p': p_value,
        'pointbiserial_r': r_pb,
        'pointbiserial_p': p_pb,
    }

    return results


def analyze_familiarity_effect(df: pd.DataFrame) -> Dict:
    """
    Analyze effect of idol familiarity on detection accuracy.

    Args:
        df: DataFrame with behavioral data

    Returns:
        Dictionary with regression/correlation results
    """
    # Correlation between familiarity and accuracy
    r_listening, p_listening = stats.pointbiserialr(
        df['detection_correct'], df['Q8_listening_freq']
    )
    r_voice, p_voice = stats.pointbiserialr(
        df['detection_correct'], df['Q9_voice_familiarity']
    )

    results = {
        'listening_freq_correlation': r_listening,
        'listening_freq_p': p_listening,
        'voice_familiarity_correlation': r_voice,
        'voice_familiarity_p': p_voice,
    }

    return results


def compute_signal_detection_metrics(df: pd.DataFrame, group: str = None) -> Dict:
    """
    Compute Signal Detection Theory metrics (d' and criterion).

    Args:
        df: DataFrame with behavioral data
        group: Optional group filter

    Returns:
        Dictionary with SDT metrics
    """
    if group:
        df = df[df['group'] == group]

    # Hit: Correctly identify deepfake as AI
    # False Alarm: Incorrectly identify real as AI
    # Miss: Incorrectly identify deepfake as real
    # Correct Rejection: Correctly identify real as real

    deepfake_trials = df[df['audio_type'] == 'deepfake']
    real_trials = df[df['audio_type'] == 'real']

    hit_rate = (deepfake_trials['Q3_audio_judgment'].isin([1, 2])).mean()
    fa_rate = (real_trials['Q3_audio_judgment'].isin([1, 2])).mean()

    # Adjust for ceiling/floor effects
    n_deepfake = len(deepfake_trials)
    n_real = len(real_trials)

    if hit_rate == 1:
        hit_rate = 1 - 1 / (2 * n_deepfake)
    elif hit_rate == 0:
        hit_rate = 1 / (2 * n_deepfake)

    if fa_rate == 1:
        fa_rate = 1 - 1 / (2 * n_real)
    elif fa_rate == 0:
        fa_rate = 1 / (2 * n_real)

    # Compute d' and criterion
    d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate))

    results = {
        'hit_rate': hit_rate,
        'false_alarm_rate': fa_rate,
        'd_prime': d_prime,
        'criterion': criterion,
        'n_deepfake': n_deepfake,
        'n_real': n_real,
    }

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_false_memory_comparison(df: pd.DataFrame, save_path: Path = None):
    """
    Plot false memory rates by group.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    props = df.groupby('group')['Q1_false_memory'].mean() * 100

    bars = ax.bar(['Group A\n(Visual Priming)', 'Group B\n(Control)'],
                  [props.get('A', 0), props.get('B', 0)],
                  color=['#E24A33', '#348ABD'],
                  edgecolor='black', linewidth=1.5)

    ax.set_ylabel('False Memory Rate (%)')
    ax.set_title('False Memory Induction by Experimental Condition')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, [props.get('A', 0), props.get('B', 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_detection_accuracy(df: pd.DataFrame, save_path: Path = None):
    """
    Plot detection accuracy by group and audio type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Overall accuracy by group
    acc_by_group = df.groupby('group')['detection_correct'].mean() * 100

    bars1 = axes[0].bar(['Group A\n(Visual Priming)', 'Group B\n(Control)'],
                        [acc_by_group.get('A', 0), acc_by_group.get('B', 0)],
                        color=['#E24A33', '#348ABD'],
                        edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Detection Accuracy (%)')
    axes[0].set_title('Overall Deepfake Detection Accuracy')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(y=50, color='gray', linestyle='--', label='Chance level')
    axes[0].legend()

    for bar, val in zip(bars1, [acc_by_group.get('A', 0), acc_by_group.get('B', 0)]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Right: Accuracy by group and audio type
    acc_by_type = df.groupby(['group', 'audio_type'])['detection_correct'].mean().unstack() * 100

    x = np.arange(2)
    width = 0.35

    bars2 = axes[1].bar(x - width/2, acc_by_type.get('deepfake', [0, 0]),
                        width, label='Deepfake', color='#E24A33', alpha=0.8)
    bars3 = axes[1].bar(x + width/2, acc_by_type.get('real', [0, 0]),
                        width, label='Real', color='#348ABD', alpha=0.8)

    axes[1].set_ylabel('Detection Accuracy (%)')
    axes[1].set_title('Detection Accuracy by Audio Type')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Group A\n(Visual Priming)', 'Group B\n(Control)'])
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].axhline(y=50, color='gray', linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_confidence_accuracy(df: pd.DataFrame, save_path: Path = None):
    """
    Plot relationship between confidence and accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean accuracy by confidence level
    acc_by_conf = df.groupby('Q4_confidence')['detection_correct'].agg(['mean', 'std', 'count'])
    acc_by_conf['se'] = acc_by_conf['std'] / np.sqrt(acc_by_conf['count'])

    ax.errorbar(acc_by_conf.index, acc_by_conf['mean'] * 100,
                yerr=acc_by_conf['se'] * 100,
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10,
                color='#8B4513')

    ax.set_xlabel('Confidence Rating (1-5)')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_title('Relationship Between Confidence and Detection Accuracy')
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.axhline(y=50, color='gray', linestyle='--', label='Chance level')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_sdt_comparison(df: pd.DataFrame, save_path: Path = None):
    """
    Plot Signal Detection Theory metrics by group.
    """
    sdt_a = compute_signal_detection_metrics(df, 'A')
    sdt_b = compute_signal_detection_metrics(df, 'B')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # d' comparison
    bars1 = axes[0].bar(['Group A\n(Visual Priming)', 'Group B\n(Control)'],
                        [sdt_a['d_prime'], sdt_b['d_prime']],
                        color=['#E24A33', '#348ABD'],
                        edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel("d' (Sensitivity)")
    axes[0].set_title('Sensitivity (d\') by Condition')
    axes[0].axhline(y=0, color='gray', linestyle='--')

    for bar, val in zip(bars1, [sdt_a['d_prime'], sdt_b['d_prime']]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Criterion comparison
    bars2 = axes[1].bar(['Group A\n(Visual Priming)', 'Group B\n(Control)'],
                        [sdt_a['criterion'], sdt_b['criterion']],
                        color=['#E24A33', '#348ABD'],
                        edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Criterion (c)')
    axes[1].set_title('Response Bias (Criterion) by Condition')
    axes[1].axhline(y=0, color='gray', linestyle='--', label='No bias')
    axes[1].legend()

    for bar, val in zip(bars2, [sdt_a['criterion'], sdt_b['criterion']]):
        y_pos = bar.get_height() + 0.05 if bar.get_height() >= 0 else bar.get_height() - 0.15
        axes[1].text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_summary_report(df: pd.DataFrame, output_path: Path) -> str:
    """
    Generate a comprehensive summary report.

    Args:
        df: DataFrame with behavioral data
        output_path: Path to save report

    Returns:
        Report text
    """
    # Run all analyses
    false_memory_results = analyze_false_memory(df)
    detection_results = analyze_detection_accuracy(df)
    confidence_results = analyze_confidence_accuracy_relationship(df)
    familiarity_results = analyze_familiarity_effect(df)
    sdt_a = compute_signal_detection_metrics(df, 'A')
    sdt_b = compute_signal_detection_metrics(df, 'B')

    report = f"""
================================================================================
BEHAVIORAL DATA ANALYSIS REPORT
Visual Priming & Deepfake Audio Perception Study
================================================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
1. SAMPLE CHARACTERISTICS
--------------------------------------------------------------------------------
Total participants: {len(df)}
Group A (Visual Priming): {(df['group'] == 'A').sum()}
Group B (Control): {(df['group'] == 'B').sum()}

Audio conditions:
  - Deepfake trials: {(df['audio_type'] == 'deepfake').sum()}
  - Real trials: {(df['audio_type'] == 'real').sum()}

Idol distribution:
{df['idol'].value_counts().to_string()}

--------------------------------------------------------------------------------
2. FALSE MEMORY ANALYSIS
--------------------------------------------------------------------------------
Research Question: Does visual priming induce false memories?

Results:
  Group A (Visual Priming): {false_memory_results['group_A_proportion']*100:.1f}% reported false memory
  Group B (Control): {false_memory_results['group_B_proportion']*100:.1f}% reported false memory

Chi-square test:
  χ² = {false_memory_results['chi2']:.3f}
  df = {false_memory_results['dof']}
  p = {false_memory_results['p_value']:.4f} {'***' if false_memory_results['p_value'] < 0.001 else '**' if false_memory_results['p_value'] < 0.01 else '*' if false_memory_results['p_value'] < 0.05 else 'ns'}
  Cramér's V = {false_memory_results['cramers_v']:.3f}

Interpretation: {'Visual priming significantly increased false memory reports.' if false_memory_results['p_value'] < 0.05 else 'No significant difference in false memory between groups.'}

--------------------------------------------------------------------------------
3. DEEPFAKE DETECTION ACCURACY
--------------------------------------------------------------------------------
Research Question: Can listeners distinguish real from deepfake audio?

Overall accuracy:
  Group A (Visual Priming): {detection_results['group_A_accuracy']*100:.1f}%
  Group B (Control): {detection_results['group_B_accuracy']*100:.1f}%

Independent samples t-test:
  t = {detection_results['t_statistic']:.3f}
  p = {detection_results['p_value']:.4f} {'***' if detection_results['p_value'] < 0.001 else '**' if detection_results['p_value'] < 0.01 else '*' if detection_results['p_value'] < 0.05 else 'ns'}
  Cohen's d = {detection_results['cohens_d']:.3f}

Interpretation: {'Visual priming significantly reduced detection accuracy.' if detection_results['p_value'] < 0.05 and detection_results['group_A_accuracy'] < detection_results['group_B_accuracy'] else 'No significant difference in detection accuracy between groups.'}

--------------------------------------------------------------------------------
4. SIGNAL DETECTION THEORY ANALYSIS
--------------------------------------------------------------------------------

Group A (Visual Priming):
  Hit Rate: {sdt_a['hit_rate']:.3f}
  False Alarm Rate: {sdt_a['false_alarm_rate']:.3f}
  d' (Sensitivity): {sdt_a['d_prime']:.3f}
  Criterion (Bias): {sdt_a['criterion']:.3f}

Group B (Control):
  Hit Rate: {sdt_b['hit_rate']:.3f}
  False Alarm Rate: {sdt_b['false_alarm_rate']:.3f}
  d' (Sensitivity): {sdt_b['d_prime']:.3f}
  Criterion (Bias): {sdt_b['criterion']:.3f}

Interpretation:
  - d' > 0 indicates above-chance discrimination ability
  - Criterion < 0 indicates liberal bias (tendency to say "AI")
  - Criterion > 0 indicates conservative bias (tendency to say "real")

--------------------------------------------------------------------------------
5. CONFIDENCE-ACCURACY RELATIONSHIP
--------------------------------------------------------------------------------

Point-biserial correlation:
  r = {confidence_results['pointbiserial_r']:.3f}
  p = {confidence_results['pointbiserial_p']:.4f}

Interpretation: {'Higher confidence is associated with higher accuracy.' if confidence_results['pointbiserial_r'] > 0 and confidence_results['pointbiserial_p'] < 0.05 else 'Confidence is not significantly related to accuracy.' if confidence_results['pointbiserial_p'] >= 0.05 else 'Higher confidence is associated with lower accuracy (overconfidence).'}

--------------------------------------------------------------------------------
6. FAMILIARITY EFFECTS
--------------------------------------------------------------------------------

Listening frequency × Detection accuracy:
  r = {familiarity_results['listening_freq_correlation']:.3f}
  p = {familiarity_results['listening_freq_p']:.4f}

Voice familiarity × Detection accuracy:
  r = {familiarity_results['voice_familiarity_correlation']:.3f}
  p = {familiarity_results['voice_familiarity_p']:.4f}

--------------------------------------------------------------------------------
7. KEY FINDINGS SUMMARY
--------------------------------------------------------------------------------

1. False Memory Effect:
   {'✓ Visual priming significantly induced false memories' if false_memory_results['p_value'] < 0.05 else '✗ No significant false memory effect observed'}

2. Detection Accuracy:
   {'✓ Visual priming impaired deepfake detection' if detection_results['p_value'] < 0.05 and detection_results['group_A_accuracy'] < detection_results['group_B_accuracy'] else '✗ No significant effect on detection accuracy'}

3. Confidence Calibration:
   {'✓ Confidence predicts accuracy' if confidence_results['pointbiserial_p'] < 0.05 else '✗ Confidence not calibrated with accuracy'}

4. Familiarity Effect:
   {'✓ Familiarity affects detection' if familiarity_results['voice_familiarity_p'] < 0.05 else '✗ No significant familiarity effect'}

================================================================================
Note: Statistical significance: * p < .05, ** p < .01, *** p < .001
================================================================================
"""

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved: {output_path}")

    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Behavioral Data Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing behavioral data
    possible_files = list(DATA_DIR.glob('*.csv')) + list(DATA_DIR.glob('behavioral*.xlsx'))

    if possible_files:
        print(f"\nFound potential data files: {[f.name for f in possible_files]}")
        # Try to load first file
        try:
            df = load_behavioral_data(possible_files[0])
            print(f"Loaded data from: {possible_files[0]}")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using sample data for demonstration...")
            df = create_sample_data()
    else:
        print("\nNo behavioral data file found.")
        print("Creating sample data for demonstration...")
        df = create_sample_data()

    # Compute accuracy
    df = compute_accuracy(df)

    # Save processed data
    processed_path = OUTPUT_DIR / 'processed_behavioral_data.csv'
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved: {processed_path}")

    # Set plot style
    set_plot_style()

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_false_memory_comparison(df, FIGURES_DIR / 'false_memory_comparison.png')
    plot_detection_accuracy(df, FIGURES_DIR / 'detection_accuracy.png')
    plot_confidence_accuracy(df, FIGURES_DIR / 'confidence_accuracy.png')
    plot_sdt_comparison(df, FIGURES_DIR / 'sdt_comparison.png')

    # Generate report
    print("\nGenerating summary report...")
    report = generate_summary_report(df, OUTPUT_DIR / 'behavioral_analysis_report.txt')

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
