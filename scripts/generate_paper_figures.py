#!/usr/bin/env python3
"""
Generate publication-ready figures for the paper
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "/Users/user/Desktop/audio_ai/deepfake_audio/paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Figure 1: False Memory Rates by Condition
# =============================================================================
def create_false_memory_figure():
    """Create false memory comparison figure matching paper data."""

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ['Treatment\n(AI Cover Priming)', 'Neutral\n(Control)']
    rates = [7.1, 17.8]  # From paper: Treatment 7.1%, Neutral 17.8%
    colors = ['#E74C3C', '#3498DB']

    bars = ax.bar(conditions, rates, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(f'{rate}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    # Add significance marker
    ax.plot([0, 1], [22, 22], 'k-', lw=1.5)
    ax.text(0.5, 23, '*', ha='center', fontsize=20)
    ax.text(0.5, 26, 'p = .030', ha='center', fontsize=11)

    ax.set_ylabel('False Memory Rate (%)', fontsize=14)
    ax.set_title('False Memory Rates by Experimental Condition', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_false_memory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_false_memory.png")

# =============================================================================
# Figure 2: Audio Source Judgment Distribution
# =============================================================================
def create_audio_judgment_figure():
    """Create audio source judgment comparison figure."""

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['AI-Generated', 'Other', 'Real']
    treatment = [47.3, 19.6, 33.0]  # From paper
    neutral = [41.6, 26.7, 31.7]    # From paper

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, treatment, width, label='Treatment', color='#E74C3C', edgecolor='black')
    bars2 = ax.bar(x + width/2, neutral, width, label='Neutral', color='#3498DB', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Percentage of Responses (%)', fontsize=14)
    ax.set_title('Audio Source Judgments by Condition', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_audio_judgment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_audio_judgment.png")

# =============================================================================
# Figure 3: EEG Frequency Indices
# =============================================================================
def create_eeg_frequency_figure():
    """Create EEG frequency indices comparison figure."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    indices = ['Relative Alpha\n(RA)', 'RSMR', 'Engagement\nIndex']
    treatment_means = [-0.135, 0.536, 0.048]
    treatment_sds = [0.634, 0.104, 0.038]
    neutral_means = [-0.168, 0.533, 0.041]
    neutral_sds = [0.606, 0.155, 0.018]
    p_values = [0.843, 0.918, 0.359]

    for i, (ax, idx, t_m, t_sd, n_m, n_sd, p) in enumerate(zip(
            axes, indices, treatment_means, treatment_sds, neutral_means, neutral_sds, p_values)):

        x = np.arange(2)
        means = [t_m, n_m]
        sds = [t_sd, n_sd]
        colors = ['#E74C3C', '#3498DB']

        bars = ax.bar(x, means, yerr=sds, capsize=5, color=colors, edgecolor='black', width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(['Treatment', 'Neutral'], fontsize=11)
        ax.set_title(idx, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add p-value
        ax.text(0.5, ax.get_ylim()[1] * 0.9, f'p = {p:.3f}\nns', ha='center', fontsize=10)

    axes[0].set_ylabel('Index Value', fontsize=12)

    fig.suptitle('Frequency-Based Neural Indices by Condition', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_eeg_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_eeg_frequency.png")

# =============================================================================
# Figure 4: Experimental Design Schematic
# =============================================================================
def create_design_figure():
    """Create experimental design schematic."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Experimental Design', ha='center', fontsize=18, fontweight='bold')

    # Treatment condition
    ax.add_patch(plt.Rectangle((0.5, 4), 2.5, 2.5, fill=True, facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
    ax.text(1.75, 6.7, 'Treatment Condition', ha='center', fontsize=12, fontweight='bold', color='#E74C3C')
    ax.text(1.75, 5.5, 'Priming Phase:\nAI Cover Thumbnail\n+ AI Cover Audio', ha='center', fontsize=10)

    # Arrow
    ax.annotate('', xy=(3.5, 5.25), xytext=(3, 5.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Test phase (Treatment)
    ax.add_patch(plt.Rectangle((3.7, 4), 2.5, 2.5, fill=True, facecolor='#F5EEF8', edgecolor='#8E44AD', linewidth=2))
    ax.text(4.95, 5.5, 'Test Phase:\nIdol Image\n+ Test Audio', ha='center', fontsize=10)

    # Arrow to survey
    ax.annotate('', xy=(6.7, 5.25), xytext=(6.2, 5.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Survey
    ax.add_patch(plt.Rectangle((7, 4), 2, 2.5, fill=True, facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=2))
    ax.text(8, 5.5, 'Survey:\nFalse Memory\nAudio Judgment\nConfidence', ha='center', fontsize=10)

    # Neutral condition
    ax.add_patch(plt.Rectangle((0.5, 0.8), 2.5, 2.5, fill=True, facecolor='#D6EAF8', edgecolor='#3498DB', linewidth=2))
    ax.text(1.75, 3.5, 'Neutral Condition', ha='center', fontsize=12, fontweight='bold', color='#3498DB')
    ax.text(1.75, 2.3, 'Priming Phase:\nClassical/Synthpop\nThumbnail + Audio', ha='center', fontsize=10)

    # Arrow
    ax.annotate('', xy=(3.5, 2.05), xytext=(3, 2.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Test phase (Neutral)
    ax.add_patch(plt.Rectangle((3.7, 0.8), 2.5, 2.5, fill=True, facecolor='#F5EEF8', edgecolor='#8E44AD', linewidth=2))
    ax.text(4.95, 2.3, 'Test Phase:\nIdol Image\n+ Test Audio', ha='center', fontsize=10)

    # Arrow to survey
    ax.annotate('', xy=(6.7, 2.05), xytext=(6.2, 2.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # EEG recording note
    ax.add_patch(plt.Rectangle((9.3, 2.5), 2.3, 3.5, fill=True, facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2))
    ax.text(10.45, 5.8, 'EEG Recording', ha='center', fontsize=11, fontweight='bold', color='#F39C12')
    ax.text(10.45, 4.5, 'Continuous\nDSI-24\n(24 channels)\n300 Hz', ha='center', fontsize=10)

    # Time labels
    ax.text(1.75, 3.8, '30 sec', ha='center', fontsize=9, style='italic')
    ax.text(4.95, 3.8, '30 sec', ha='center', fontsize=9, style='italic')
    ax.text(8, 3.8, '2-3 min', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_design.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_design.png")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("Generating publication figures...")
    create_false_memory_figure()
    create_audio_judgment_figure()
    create_eeg_frequency_figure()
    create_design_figure()
    print("\nAll figures saved to:", OUTPUT_DIR)
