#!/usr/bin/env python3
"""
Real Data Analysis Script for Visual Priming & Deepfake Audio Perception Study
===============================================================================

Analyzes actual experimental data from Google Forms responses.

Author: Generated for audio_ai experiment
Date: 2026-03-09
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Non-interactive matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/Users/user/Desktop/audio_ai")
OUTPUT_DIR = DATA_DIR / "analysis_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Group conditions mapping (based on experimental design)
# Groups 1-8: Treatment (Visual Priming with idol-related content)
# Groups 9-16: Neutral (Control with unrelated content)
TREATMENT_GROUPS = [1, 2, 3, 4, 5, 6, 7, 8]
NEUTRAL_GROUPS = [9, 10, 11, 12, 13, 14, 15, 16]

# =============================================================================
# Data Loading
# =============================================================================

def load_all_data():
    """Load and combine all response data."""

    # Load response files
    df1 = pd.read_excel(DATA_DIR / "Group #1-8 (Responses).xlsx")
    df2 = pd.read_excel(DATA_DIR / "Group #9-16 (Responses).xlsx")

    # Load participant group assignments
    participants = pd.read_csv(DATA_DIR / "participant_group for Audio Task.csv")

    # Load error records
    errors = pd.read_excel(DATA_DIR / "Errors from Audio task.xlsx")

    print(f"Group 1-8 responses: {len(df1)}")
    print(f"Group 9-16 responses: {len(df2)}")
    print(f"Participant assignments: {len(participants)}")
    print(f"Error records: {len(errors)}")

    return df1, df2, participants, errors


def reshape_to_long_format(df, group_range="1-8"):
    """
    Reshape wide format data to long format.
    Each participant has 4 audio clip evaluations.
    """

    records = []

    for idx, row in df.iterrows():
        group_num = row['Group number']
        timestamp = row['Timestamp']

        # 4 audio clips per participant
        for clip_num in range(1, 5):
            suffix = "" if clip_num == 1 else f" {clip_num}"

            record = {
                'participant_idx': idx,
                'group_number': group_num,
                'timestamp': timestamp,
                'clip_number': clip_num,
                'group_range': group_range,
            }

            # Q1: False memory
            q1_col = f"1.해당 가수가 이 노래를 부른 영상을 본 적이 있습니까?{suffix}"
            if q1_col in df.columns:
                record['Q1_false_memory'] = row[q1_col]

            # Q2: Confidence in Q1
            q2_col = f"2. 위 질문(1번 문항)에 대한 답변에 얼마나 확신하십니까?{suffix}"
            if q2_col in df.columns:
                record['Q2_confidence'] = row[q2_col]

            # Q3: Audio source judgment
            q3_col = f"3. 방금 들은 노래는 어떤 방식으로 만들어졌다고 생각하십니까?(복수 선택 불가){suffix}"
            if q3_col in df.columns:
                record['Q3_audio_judgment'] = row[q3_col]

            # Q4: Confidence in Q3
            q4_col = f"4. 3번 문항에 대한 답변에 얼마나 확신하십니까?{suffix}"
            if q4_col in df.columns:
                record['Q4_confidence'] = row[q4_col]

            # Q5: Judgment factors
            factors = ['발음', '억양', '감정 표현', '음질', '박자감(리듬감)', '숨소리나 호흡의 자연스러운 정도']
            for factor in factors:
                col = f"5. 다음 항목들이 본인의 판단(4번 문항의 응답)에 어느 정도 영향을 미쳤는지 평가해주세요. [{factor}]{suffix}"
                if col in df.columns:
                    record[f'Q5_{factor}'] = row[col]

            # Q6: Likeability
            q6_col = f"6. 방금 들은 노래에 대한 전반적인 호감도를 선택해 주십시오.{suffix}"
            if q6_col in df.columns:
                record['Q6_likeability'] = row[q6_col]

            records.append(record)

        # Q7-Q9 are per-singer (2 singers), add to first 2 clips
        for singer_num in range(1, 3):
            suffix = "" if singer_num == 1 else " 2"
            clip_idx = (singer_num - 1) * 2  # 0 or 2

            q7_col = f"7.  평소 해당 가수의 목소리 또는 노래를 얼마나 자주 접합니까?{suffix}"
            if q7_col in df.columns and clip_idx < len(records):
                # Find the right record
                for rec in records[-4:]:  # Last 4 records for this participant
                    if rec['clip_number'] in [singer_num, singer_num + 2]:
                        rec['Q7_listening_freq'] = row[q7_col]

            q8_col = f"8. 본인이 해당 가수의 목소리(또는 노래 스타일)을 얼마나 잘 알고 있다고 생각합니까?{suffix}"
            if q8_col in df.columns:
                for rec in records[-4:]:
                    if rec['clip_number'] in [singer_num, singer_num + 2]:
                        rec['Q8_voice_familiarity'] = row[q8_col]

    return pd.DataFrame(records)


def parse_responses(df):
    """Parse and clean response values."""
    import re

    df = df.copy()

    def extract_number(x):
        """Extract number from string like '(4) 꽤 확신한다'."""
        if pd.isna(x):
            return np.nan
        match = re.search(r'\((\d+)\)', str(x))
        if match:
            return int(match.group(1))
        # Try just getting first digit
        match = re.search(r'(\d+)', str(x))
        if match:
            return int(match.group(1))
        return np.nan

    # Q1: False memory (예/아니오 -> 1/0)
    if 'Q1_false_memory' in df.columns:
        df['Q1_false_memory_binary'] = df['Q1_false_memory'].apply(
            lambda x: 1 if '예' in str(x) or '있' in str(x) else 0 if pd.notna(x) else np.nan
        )

    # Q2: Confidence in false memory (extract number)
    if 'Q2_confidence' in df.columns:
        df['Q2_confidence_num'] = df['Q2_confidence'].apply(extract_number)

    # Q3: Audio judgment - categorize as AI vs Real
    if 'Q3_audio_judgment' in df.columns:
        def categorize_judgment(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x)
            # Check the option number first
            if x_str.startswith('(1)') or x_str.startswith('(2)'):
                return 'AI'  # AI synthesis options
            elif x_str.startswith('(3)') or x_str.startswith('(4)'):
                return 'Other'  # Cover singer options
            elif x_str.startswith('(5)') or x_str.startswith('(6)'):
                return 'Real'  # Real artist options
            # Fallback to text-based detection
            x_lower = x_str.lower()
            if 'ai' in x_lower or '딥러닝' in x_lower or '합성' in x_lower:
                return 'AI'
            elif '실제' in x_lower:
                return 'Real'
            else:
                return 'Other'

        df['Q3_judgment_category'] = df['Q3_audio_judgment'].apply(categorize_judgment)
        df['Q3_judgment_num'] = df['Q3_audio_judgment'].apply(extract_number)

    # Q4: Confidence in audio judgment (extract number)
    if 'Q4_confidence' in df.columns:
        df['Q4_confidence_num'] = df['Q4_confidence'].apply(extract_number)

    # Q6: Likeability (extract number)
    if 'Q6_likeability' in df.columns:
        df['Q6_likeability_num'] = df['Q6_likeability'].apply(extract_number)

    # Q7, Q8: Familiarity (extract number)
    if 'Q7_listening_freq' in df.columns:
        df['Q7_listening_freq_num'] = df['Q7_listening_freq'].apply(extract_number)

    if 'Q8_voice_familiarity' in df.columns:
        df['Q8_voice_familiarity_num'] = df['Q8_voice_familiarity'].apply(extract_number)

    # Q5 factors (extract numbers)
    for col in df.columns:
        if col.startswith('Q5_') and '_num' not in col:
            df[col + '_num'] = df[col].apply(extract_number)

    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_false_memory_by_group(df):
    """Analyze false memory rates by experimental group."""

    if 'Q1_false_memory_binary' not in df.columns:
        print("Q1_false_memory_binary column not found")
        return None

    # Group-level analysis
    group_stats = df.groupby('group_number').agg({
        'Q1_false_memory_binary': ['mean', 'std', 'count', 'sum']
    }).round(3)

    group_stats.columns = ['mean', 'std', 'count', 'sum']
    group_stats['false_memory_rate'] = (group_stats['mean'] * 100).round(1)

    print("\n=== False Memory Rate by Group ===")
    print(group_stats)

    return group_stats


def analyze_audio_judgments(df):
    """Analyze audio source judgments."""

    if 'Q3_audio_judgment' not in df.columns:
        print("Q3_audio_judgment column not found")
        return None

    print("\n=== Audio Source Judgment Distribution ===")

    # Overall distribution
    judgment_counts = df['Q3_audio_judgment'].value_counts()
    print("\nOverall:")
    print(judgment_counts)

    # By category
    if 'Q3_judgment_category' in df.columns:
        category_counts = df['Q3_judgment_category'].value_counts()
        print("\nBy Category:")
        print(category_counts)

        # By group
        print("\nBy Group:")
        group_judgment = pd.crosstab(df['group_number'], df['Q3_judgment_category'], normalize='index') * 100
        print(group_judgment.round(1))

    return judgment_counts


def analyze_confidence(df):
    """Analyze confidence ratings."""

    print("\n=== Confidence Analysis ===")

    # Q2: Confidence in false memory judgment
    if 'Q2_confidence_num' in df.columns:
        print("\nQ2 (False Memory Confidence):")
        print(df['Q2_confidence_num'].describe())

    # Q4: Confidence in audio judgment
    if 'Q4_confidence_num' in df.columns:
        print("\nQ4 (Audio Judgment Confidence):")
        print(df['Q4_confidence_num'].describe())

        # By group
        conf_by_group = df.groupby('group_number')['Q4_confidence_num'].mean()
        print("\nQ4 Confidence by Group:")
        print(conf_by_group.round(2))


def analyze_judgment_factors(df):
    """Analyze which factors influenced judgments."""

    print("\n=== Judgment Factor Analysis ===")

    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]

    if not factor_cols:
        print("No Q5 factor columns found")
        return None

    # Mean importance of each factor
    factor_means = df[factor_cols].mean().sort_values(ascending=False)

    print("\nMean Factor Importance:")
    for col, val in factor_means.items():
        factor_name = col.replace('Q5_', '').replace('_num', '')
        print(f"  {factor_name}: {val:.2f}")

    return factor_means


def analyze_familiarity_effect(df):
    """Analyze effect of voice familiarity on judgments."""

    print("\n=== Familiarity Effect Analysis ===")

    if 'Q8_voice_familiarity_num' not in df.columns:
        print("Q8_voice_familiarity_num column not found")
        return None

    # Correlation with false memory
    if 'Q1_false_memory_binary' in df.columns:
        valid_data = df[['Q8_voice_familiarity_num', 'Q1_false_memory_binary']].dropna()
        if len(valid_data) > 5:
            r, p = stats.pointbiserialr(valid_data['Q1_false_memory_binary'],
                                        valid_data['Q8_voice_familiarity_num'])
            print(f"\nFamiliarity × False Memory: r={r:.3f}, p={p:.4f}")


def analyze_by_condition(df):
    """Analyze Treatment vs Neutral condition differences."""

    print("\n=== CONDITION-LEVEL ANALYSIS (Treatment vs Neutral) ===")

    # Add condition column
    df = df.copy()
    df['condition'] = df['group_number'].apply(
        lambda x: 'Treatment' if x in TREATMENT_GROUPS else 'Neutral'
    )

    # Filter valid data
    df_valid = df[df['Q1_false_memory_binary'].notna()].copy()

    print(f"\nValid observations: {len(df_valid)}")
    print(f"  Treatment: {len(df_valid[df_valid['condition'] == 'Treatment'])}")
    print(f"  Neutral: {len(df_valid[df_valid['condition'] == 'Neutral'])}")

    # False Memory by condition
    print("\n--- False Memory by Condition ---")
    fm_by_cond = df_valid.groupby('condition')['Q1_false_memory_binary'].agg(['sum', 'count', 'mean'])
    fm_by_cond['rate_pct'] = fm_by_cond['mean'] * 100
    print(fm_by_cond)

    # Chi-square test
    treatment = df_valid[df_valid['condition'] == 'Treatment']['Q1_false_memory_binary']
    neutral = df_valid[df_valid['condition'] == 'Neutral']['Q1_false_memory_binary']

    contingency = np.array([
        [treatment.sum(), len(treatment) - treatment.sum()],
        [neutral.sum(), len(neutral) - neutral.sum()]
    ])

    chi2_yates, p_yates, dof, _ = stats.chi2_contingency(contingency, correction=True)
    cramers_v = np.sqrt(chi2_yates / len(df_valid))

    print(f"\nChi-square (Yates): χ²(1) = {chi2_yates:.2f}, p = {p_yates:.4f}")
    print(f"Cramér's V = {cramers_v:.3f}")

    # Audio judgment by condition
    if 'Q3_judgment_category' in df_valid.columns:
        print("\n--- Audio Judgment by Condition ---")
        judgment_pct = pd.crosstab(
            df_valid['condition'],
            df_valid['Q3_judgment_category'],
            normalize='index'
        ) * 100
        print(judgment_pct.round(1))

    # Confidence by condition
    print("\n--- Confidence by Condition ---")
    if 'Q2_confidence_num' in df_valid.columns:
        q2_by_cond = df_valid.groupby('condition')['Q2_confidence_num'].agg(['mean', 'std'])
        print("Q2 (False Memory Confidence):")
        print(q2_by_cond.round(2))

    if 'Q4_confidence_num' in df_valid.columns:
        q4_by_cond = df_valid.groupby('condition')['Q4_confidence_num'].agg(['mean', 'std'])
        print("\nQ4 (Audio Judgment Confidence):")
        print(q4_by_cond.round(2))

    # Likeability by condition
    if 'Q6_likeability_num' in df_valid.columns:
        print("\n--- Likeability by Condition ---")
        like_by_cond = df_valid.groupby('condition')['Q6_likeability_num'].agg(['mean', 'std'])
        print(like_by_cond.round(2))

        treat_like = df_valid[df_valid['condition'] == 'Treatment']['Q6_likeability_num'].dropna()
        neut_like = df_valid[df_valid['condition'] == 'Neutral']['Q6_likeability_num'].dropna()
        t, p = stats.ttest_ind(treat_like, neut_like)
        print(f"t-test: t = {t:.2f}, p = {p:.4f}")

    return df_valid


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_false_memory_by_group(df, save_path=None):
    """Plot false memory rates by group."""

    fig, ax = plt.subplots(figsize=(12, 6))

    group_means = df.groupby('group_number')['Q1_false_memory_binary'].mean() * 100

    colors = plt.cm.tab20(np.linspace(0, 1, len(group_means)))
    bars = ax.bar(group_means.index.astype(str), group_means.values, color=colors, edgecolor='black')

    ax.set_xlabel('Group Number')
    ax.set_ylabel('False Memory Rate (%)')
    ax.set_title('False Memory Rate by Experimental Group')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, group_means.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_audio_judgment_distribution(df, save_path=None):
    """Plot audio judgment distribution."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Overall distribution
    judgment_counts = df['Q3_audio_judgment'].value_counts()

    # Shorten labels for display
    short_labels = [str(label)[:30] + '...' if len(str(label)) > 30 else str(label)
                    for label in judgment_counts.index]

    axes[0].barh(range(len(judgment_counts)), judgment_counts.values, color='steelblue')
    axes[0].set_yticks(range(len(judgment_counts)))
    axes[0].set_yticklabels(short_labels, fontsize=8)
    axes[0].set_xlabel('Count')
    axes[0].set_title('Audio Source Judgment Distribution')
    axes[0].invert_yaxis()

    # Right: AI vs Real category
    if 'Q3_judgment_category' in df.columns:
        cat_counts = df['Q3_judgment_category'].value_counts()
        colors = {'AI': '#E24A33', 'Real': '#348ABD', 'Other': '#988ED5'}
        bar_colors = [colors.get(cat, 'gray') for cat in cat_counts.index]

        axes[1].bar(cat_counts.index, cat_counts.values, color=bar_colors, edgecolor='black')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Judgment Category (AI vs Real)')

        for i, (cat, val) in enumerate(cat_counts.items()):
            axes[1].text(i, val + 1, str(val), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_confidence_distribution(df, save_path=None):
    """Plot confidence rating distributions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q2 Confidence
    if 'Q2_confidence_num' in df.columns:
        q2_data = df['Q2_confidence_num'].dropna()
        axes[0].hist(q2_data, bins=5, edgecolor='black', color='#2ecc71', alpha=0.7)
        axes[0].set_xlabel('Confidence Rating')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Q2: False Memory Confidence\n(M={q2_data.mean():.2f}, SD={q2_data.std():.2f})')

    # Q4 Confidence
    if 'Q4_confidence_num' in df.columns:
        q4_data = df['Q4_confidence_num'].dropna()
        axes[1].hist(q4_data, bins=5, edgecolor='black', color='#3498db', alpha=0.7)
        axes[1].set_xlabel('Confidence Rating')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Q4: Audio Judgment Confidence\n(M={q4_data.mean():.2f}, SD={q4_data.std():.2f})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


def plot_factor_importance(df, save_path=None):
    """Plot judgment factor importance."""

    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]

    if not factor_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    factor_means = df[factor_cols].mean().sort_values(ascending=True)
    factor_names = [col.replace('Q5_', '').replace('_num', '') for col in factor_means.index]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(factor_means)))

    bars = ax.barh(factor_names, factor_means.values, color=colors, edgecolor='black')
    ax.set_xlabel('Mean Importance Rating')
    ax.set_title('Factors Influencing Audio Authenticity Judgment')

    # Add value labels
    for bar, val in zip(bars, factor_means.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(df, output_path):
    """Generate comprehensive analysis report."""

    report = []
    report.append("=" * 80)
    report.append("BEHAVIORAL DATA ANALYSIS REPORT - REAL DATA")
    report.append("Visual Priming & Deepfake Audio Perception Study")
    report.append("=" * 80)
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Sample characteristics
    report.append("\n" + "-" * 80)
    report.append("1. SAMPLE CHARACTERISTICS")
    report.append("-" * 80)

    n_participants = df['participant_idx'].nunique()
    n_observations = len(df)
    n_groups = df['group_number'].nunique()

    report.append(f"\nTotal participants: {n_participants}")
    report.append(f"Total observations (clips evaluated): {n_observations}")
    report.append(f"Number of experimental groups: {n_groups}")
    report.append(f"Groups: {sorted(df['group_number'].dropna().unique())}")

    # Observations per group
    report.append("\nObservations per group:")
    obs_per_group = df.groupby('group_number').size()
    for grp, count in obs_per_group.items():
        report.append(f"  Group {grp}: {count}")

    # False Memory Analysis
    report.append("\n" + "-" * 80)
    report.append("2. FALSE MEMORY ANALYSIS")
    report.append("-" * 80)

    if 'Q1_false_memory_binary' in df.columns:
        overall_fm_rate = df['Q1_false_memory_binary'].mean() * 100
        report.append(f"\nOverall false memory rate: {overall_fm_rate:.1f}%")

        report.append("\nFalse memory rate by group:")
        fm_by_group = df.groupby('group_number')['Q1_false_memory_binary'].agg(['mean', 'count'])
        fm_by_group['rate'] = (fm_by_group['mean'] * 100).round(1)
        for grp, row in fm_by_group.iterrows():
            report.append(f"  Group {grp}: {row['rate']:.1f}% (n={int(row['count'])})")

    # Audio Judgment Analysis
    report.append("\n" + "-" * 80)
    report.append("3. AUDIO SOURCE JUDGMENT ANALYSIS")
    report.append("-" * 80)

    if 'Q3_audio_judgment' in df.columns:
        report.append("\nJudgment distribution:")
        judgment_counts = df['Q3_audio_judgment'].value_counts()
        for judgment, count in judgment_counts.items():
            pct = count / len(df) * 100
            report.append(f"  {judgment}: {count} ({pct:.1f}%)")

        if 'Q3_judgment_category' in df.columns:
            report.append("\nBy category (AI vs Real):")
            cat_counts = df['Q3_judgment_category'].value_counts()
            for cat, count in cat_counts.items():
                pct = count / len(df) * 100
                report.append(f"  {cat}: {count} ({pct:.1f}%)")

    # Confidence Analysis
    report.append("\n" + "-" * 80)
    report.append("4. CONFIDENCE ANALYSIS")
    report.append("-" * 80)

    if 'Q2_confidence_num' in df.columns:
        q2_stats = df['Q2_confidence_num'].describe()
        report.append(f"\nQ2 (False Memory Confidence):")
        report.append(f"  Mean: {q2_stats['mean']:.2f}")
        report.append(f"  SD: {q2_stats['std']:.2f}")
        report.append(f"  Range: {q2_stats['min']:.0f} - {q2_stats['max']:.0f}")

    if 'Q4_confidence_num' in df.columns:
        q4_stats = df['Q4_confidence_num'].describe()
        report.append(f"\nQ4 (Audio Judgment Confidence):")
        report.append(f"  Mean: {q4_stats['mean']:.2f}")
        report.append(f"  SD: {q4_stats['std']:.2f}")
        report.append(f"  Range: {q4_stats['min']:.0f} - {q4_stats['max']:.0f}")

    # Factor Analysis
    report.append("\n" + "-" * 80)
    report.append("5. JUDGMENT FACTOR IMPORTANCE")
    report.append("-" * 80)

    factor_cols = [col for col in df.columns if col.startswith('Q5_') and col.endswith('_num')]
    if factor_cols:
        report.append("\nMean importance rating (higher = more influential):")
        factor_means = df[factor_cols].mean().sort_values(ascending=False)
        for col, val in factor_means.items():
            factor_name = col.replace('Q5_', '').replace('_num', '')
            report.append(f"  {factor_name}: {val:.2f}")

    # Familiarity Analysis
    report.append("\n" + "-" * 80)
    report.append("6. FAMILIARITY ANALYSIS")
    report.append("-" * 80)

    if 'Q7_listening_freq_num' in df.columns:
        q7_stats = df['Q7_listening_freq_num'].describe()
        report.append(f"\nQ7 (Listening Frequency):")
        report.append(f"  Mean: {q7_stats['mean']:.2f}")
        report.append(f"  SD: {q7_stats['std']:.2f}")

    if 'Q8_voice_familiarity_num' in df.columns:
        q8_stats = df['Q8_voice_familiarity_num'].describe()
        report.append(f"\nQ8 (Voice Familiarity):")
        report.append(f"  Mean: {q8_stats['mean']:.2f}")
        report.append(f"  SD: {q8_stats['std']:.2f}")

    # Likeability
    report.append("\n" + "-" * 80)
    report.append("7. LIKEABILITY ANALYSIS")
    report.append("-" * 80)

    if 'Q6_likeability_num' in df.columns:
        q6_stats = df['Q6_likeability_num'].describe()
        report.append(f"\nQ6 (Overall Likeability):")
        report.append(f"  Mean: {q6_stats['mean']:.2f}")
        report.append(f"  SD: {q6_stats['std']:.2f}")

        like_by_group = df.groupby('group_number')['Q6_likeability_num'].mean()
        report.append("\nLikeability by group:")
        for grp, val in like_by_group.items():
            report.append(f"  Group {grp}: {val:.2f}")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved: {output_path}")

    return report_text


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main analysis pipeline."""

    print("=" * 60)
    print("Real Data Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n>>> Loading data...")
    df1, df2, participants, errors = load_all_data()

    # Reshape to long format
    print("\n>>> Reshaping data to long format...")
    df1_long = reshape_to_long_format(df1, "1-8")
    df2_long = reshape_to_long_format(df2, "9-16")

    # Combine
    df_combined = pd.concat([df1_long, df2_long], ignore_index=True)
    print(f"Combined dataset: {len(df_combined)} observations")

    # Parse responses
    print("\n>>> Parsing responses...")
    df_combined = parse_responses(df_combined)

    # Save processed data
    processed_path = OUTPUT_DIR / 'processed_real_data.csv'
    df_combined.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"Processed data saved: {processed_path}")

    # Run analyses
    print("\n>>> Running analyses...")
    analyze_false_memory_by_group(df_combined)
    analyze_audio_judgments(df_combined)
    analyze_confidence(df_combined)
    analyze_judgment_factors(df_combined)
    analyze_familiarity_effect(df_combined)
    analyze_by_condition(df_combined)  # Treatment vs Neutral comparison

    # Generate visualizations
    print("\n>>> Generating visualizations...")
    plot_false_memory_by_group(df_combined, FIGURES_DIR / 'real_false_memory_by_group.png')
    plot_audio_judgment_distribution(df_combined, FIGURES_DIR / 'real_audio_judgment_dist.png')
    plot_confidence_distribution(df_combined, FIGURES_DIR / 'real_confidence_dist.png')
    plot_factor_importance(df_combined, FIGURES_DIR / 'real_factor_importance.png')

    # Generate report
    print("\n>>> Generating report...")
    report = generate_report(df_combined, OUTPUT_DIR / 'real_data_analysis_report.txt')

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return df_combined


if __name__ == "__main__":
    df = main()
