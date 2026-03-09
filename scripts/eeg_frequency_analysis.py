#!/usr/bin/env python3
"""
EEG Frequency Band Analysis for Visual Priming Study
Analyzes Relative Alpha (RA), RSMR, and other frequency-based indices

Based on methodology from:
- Frontal Alpha Asymmetry (FAA) for affective processing
- Relative Sensorimotor Rhythm (RSMR) for attention
"""

import os
import numpy as np
import pandas as pd
import pyxdf
import mne
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "/Users/user/Desktop/audio_ai/gdrive_download"
OUTPUT_DIR = "/Users/user/Desktop/audio_ai/analysis_results"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# EEG Parameters
SFREQ = 300  # Hz
LOWPASS = 40  # Hz
HIGHPASS = 0.1  # Hz

# Frequency Bands (Hz)
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'smr': (12, 15),  # Sensorimotor rhythm
    'beta': (13, 30),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
}

# DSI-24 Channel Layout (10-20 approximate mapping)
# Based on DSI-24 documentation
CHANNEL_NAMES = [f'EEG{str(i).zfill(3)}' for i in range(1, 25)]

# Frontal channels for alpha asymmetry
# DSI-24: EEG001-006 are frontal region
LEFT_FRONTAL = ['EEG001', 'EEG002', 'EEG003']  # F3, F7, Fp1 equivalent
RIGHT_FRONTAL = ['EEG004', 'EEG005', 'EEG006']  # F4, F8, Fp2 equivalent

# Central/Parietal for RSMR
CENTRAL_CHANNELS = ['EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012']
PARIETAL_CHANNELS = ['EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018']

# Condition mapping based on subject number
def get_condition(subject_id):
    """Determine condition based on subject ID pattern."""
    # Format: sub-25_06_25_1_1_ses-S001_task-Default_run-001_eeg
    # Extract the subject number (e.g., "1_1" -> 1)
    import re

    # Try to find pattern like _N_N_ where N is the subject number
    match = re.search(r'sub-\d+_\d+_\d+_(\d+)_\d+', subject_id)
    if match:
        subj_num = int(match.group(1))
        return 'Treatment' if subj_num % 2 == 1 else 'Neutral'

    # Fallback: try the old method
    parts = subject_id.replace('sub-', '').split('_')
    for part in parts:
        try:
            subj_num = int(part)
            if 1 <= subj_num <= 100:  # Reasonable subject number range
                return 'Treatment' if subj_num % 2 == 1 else 'Neutral'
        except:
            continue

    return 'Unknown'

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_xdf_data(filepath):
    """Load XDF file and extract EEG data."""
    try:
        streams, header = pyxdf.load_xdf(filepath)

        eeg_stream = None
        marker_stream = None

        for stream in streams:
            stream_type = stream['info']['type'][0].lower()
            if 'eeg' in stream_type:
                eeg_stream = stream
            elif 'marker' in stream_type or 'event' in stream_type:
                marker_stream = stream

        if eeg_stream is None:
            return None, None, None

        eeg_data = np.array(eeg_stream['time_series']).T
        eeg_times = np.array(eeg_stream['time_stamps'])

        if eeg_data.shape[1] == 0:
            return None, None, None

        # Get markers
        markers = []
        if marker_stream is not None:
            marker_times = np.array(marker_stream['time_stamps'])
            marker_values = marker_stream['time_series']
            for t, v in zip(marker_times, marker_values):
                markers.append({'time': t, 'value': str(v[0]) if isinstance(v, list) else str(v)})

        return eeg_data, eeg_times, markers

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None

def preprocess_eeg(eeg_data, sfreq=SFREQ):
    """Basic preprocessing: filtering."""
    n_channels = eeg_data.shape[0]
    ch_names = CHANNEL_NAMES[:n_channels]
    ch_types = ['eeg'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data * 1e-6, info, verbose=False)  # Convert to V

    # Bandpass filter
    raw.filter(HIGHPASS, LOWPASS, verbose=False)

    # Notch filter for line noise
    raw.notch_filter(60, verbose=False)

    return raw

# ============================================================================
# FREQUENCY ANALYSIS
# ============================================================================

def compute_psd(data, sfreq, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = min(256, len(data))

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
    return freqs, psd

def get_band_power(freqs, psd, band):
    """Extract power in a specific frequency band."""
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx]) if np.any(idx) else 0

def compute_relative_alpha(raw, epoch_data=None):
    """
    Compute Relative Alpha (RA) - Frontal Alpha Asymmetry.
    RA = log(Right Alpha) - log(Left Alpha)
    Positive RA indicates greater left frontal activity (approach motivation)
    """
    if epoch_data is not None:
        data = epoch_data
    else:
        data = raw.get_data()

    ch_names = raw.ch_names

    # Get channel indices
    left_idx = [ch_names.index(ch) for ch in LEFT_FRONTAL if ch in ch_names]
    right_idx = [ch_names.index(ch) for ch in RIGHT_FRONTAL if ch in ch_names]

    if not left_idx or not right_idx:
        return np.nan

    # Compute alpha power for each side
    left_alpha = []
    right_alpha = []

    for idx in left_idx:
        if data.ndim == 2:
            freqs, psd = compute_psd(data[idx], SFREQ)
        else:  # 3D epoched data
            freqs, psd = compute_psd(data[:, idx, :].mean(axis=0), SFREQ)
        left_alpha.append(get_band_power(freqs, psd, FREQ_BANDS['alpha']))

    for idx in right_idx:
        if data.ndim == 2:
            freqs, psd = compute_psd(data[idx], SFREQ)
        else:
            freqs, psd = compute_psd(data[:, idx, :].mean(axis=0), SFREQ)
        right_alpha.append(get_band_power(freqs, psd, FREQ_BANDS['alpha']))

    left_mean = np.mean(left_alpha)
    right_mean = np.mean(right_alpha)

    if left_mean > 0 and right_mean > 0:
        ra = np.log(right_mean) - np.log(left_mean)
        return ra
    return np.nan

def compute_rsmr(raw, epoch_data=None):
    """
    Compute Relative Sensorimotor Rhythm (RSMR).
    RSMR = SMR Power / Alpha Power
    Higher RSMR indicates focused attention.
    """
    if epoch_data is not None:
        data = epoch_data
    else:
        data = raw.get_data()

    ch_names = raw.ch_names

    # Use central channels for SMR
    central_idx = [ch_names.index(ch) for ch in CENTRAL_CHANNELS if ch in ch_names]

    if not central_idx:
        return np.nan

    smr_power = []
    alpha_power = []

    for idx in central_idx:
        if data.ndim == 2:
            freqs, psd = compute_psd(data[idx], SFREQ)
        else:
            freqs, psd = compute_psd(data[:, idx, :].mean(axis=0), SFREQ)
        smr_power.append(get_band_power(freqs, psd, FREQ_BANDS['smr']))
        alpha_power.append(get_band_power(freqs, psd, FREQ_BANDS['alpha']))

    smr_mean = np.mean(smr_power)
    alpha_mean = np.mean(alpha_power)

    if alpha_mean > 0:
        return smr_mean / alpha_mean
    return np.nan

def compute_engagement_index(raw, epoch_data=None):
    """
    Compute Engagement Index.
    EI = Beta / (Alpha + Theta)
    Higher values indicate greater cognitive engagement.
    """
    if epoch_data is not None:
        data = epoch_data
    else:
        data = raw.get_data()

    ch_names = raw.ch_names
    all_idx = list(range(len(ch_names)))

    theta_power = []
    alpha_power = []
    beta_power = []

    for idx in all_idx:
        if data.ndim == 2:
            freqs, psd = compute_psd(data[idx], SFREQ)
        else:
            freqs, psd = compute_psd(data[:, idx, :].mean(axis=0), SFREQ)
        theta_power.append(get_band_power(freqs, psd, FREQ_BANDS['theta']))
        alpha_power.append(get_band_power(freqs, psd, FREQ_BANDS['alpha']))
        beta_power.append(get_band_power(freqs, psd, FREQ_BANDS['beta']))

    beta_mean = np.mean(beta_power)
    alpha_mean = np.mean(alpha_power)
    theta_mean = np.mean(theta_power)

    denominator = alpha_mean + theta_mean
    if denominator > 0:
        return beta_mean / denominator
    return np.nan

def compute_all_band_powers(raw, epoch_data=None):
    """Compute power in all frequency bands."""
    if epoch_data is not None:
        data = epoch_data
    else:
        data = raw.get_data()

    band_powers = {}

    for band_name, band_range in FREQ_BANDS.items():
        powers = []
        for idx in range(data.shape[0] if data.ndim == 2 else data.shape[1]):
            if data.ndim == 2:
                freqs, psd = compute_psd(data[idx], SFREQ)
            else:
                freqs, psd = compute_psd(data[:, idx, :].mean(axis=0), SFREQ)
            powers.append(get_band_power(freqs, psd, band_range))
        band_powers[band_name] = np.mean(powers)

    # Compute relative powers (normalized)
    total_power = sum(band_powers.values())
    if total_power > 0:
        for band_name in list(band_powers.keys()):
            band_powers[f'{band_name}_relative'] = band_powers[band_name] / total_power

    return band_powers

# ============================================================================
# EPOCH EXTRACTION
# ============================================================================

def extract_task_epochs(raw, markers, eeg_times, epoch_duration=30.0):
    """
    Extract epochs corresponding to task periods.
    Looking for markers that indicate stimulus onset.
    """
    data = raw.get_data()
    epochs = []

    # Find relevant markers (stimulus onset markers)
    stim_markers = []
    for m in markers:
        # Look for markers indicating audio/visual stimulus
        marker_val = m['value'].lower()
        if any(kw in marker_val for kw in ['audio', 'video', 'stim', 'start', 'trial']):
            stim_markers.append(m)

    if not stim_markers:
        # If no specific markers, use all markers
        stim_markers = markers

    for marker in stim_markers:
        marker_time = marker['time']

        # Find corresponding sample index
        time_diffs = np.abs(eeg_times - marker_time)
        start_idx = np.argmin(time_diffs)

        # Extract epoch
        n_samples = int(epoch_duration * SFREQ)
        end_idx = start_idx + n_samples

        if end_idx <= data.shape[1]:
            epoch_data = data[:, start_idx:end_idx]
            epochs.append({
                'data': epoch_data,
                'marker': marker['value'],
                'start_time': marker_time
            })

    return epochs

def extract_continuous_segments(raw, segment_duration=30.0):
    """
    Extract continuous segments for analysis when markers are not reliable.
    """
    data = raw.get_data()
    n_samples = int(segment_duration * SFREQ)
    total_samples = data.shape[1]

    segments = []
    start_idx = 0

    while start_idx + n_samples <= total_samples:
        segment_data = data[:, start_idx:start_idx + n_samples]
        segments.append({
            'data': segment_data,
            'start_sample': start_idx
        })
        start_idx += n_samples  # Non-overlapping

    return segments

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_subject(filepath, subject_id):
    """Analyze a single subject's EEG data."""
    print(f"  Processing {subject_id}...")

    # Load data
    eeg_data, eeg_times, markers = load_xdf_data(filepath)

    if eeg_data is None:
        print(f"    Failed to load data")
        return None

    # Preprocess
    raw = preprocess_eeg(eeg_data)

    # Get condition
    condition = get_condition(subject_id)

    # Compute indices on full recording
    ra = compute_relative_alpha(raw)
    rsmr = compute_rsmr(raw)
    ei = compute_engagement_index(raw)
    band_powers = compute_all_band_powers(raw)

    result = {
        'subject_id': subject_id,
        'condition': condition,
        'relative_alpha': ra,
        'rsmr': rsmr,
        'engagement_index': ei,
        'duration_s': eeg_data.shape[1] / SFREQ,
        'n_markers': len(markers),
    }
    result.update(band_powers)

    # Also extract and analyze epochs if markers available
    if markers:
        epochs = extract_task_epochs(raw, markers, eeg_times)
        if epochs:
            epoch_ras = []
            epoch_rsmrs = []
            epoch_eis = []

            for epoch in epochs:
                epoch_ra = compute_relative_alpha(raw, epoch['data'])
                epoch_rsmr = compute_rsmr(raw, epoch['data'])
                epoch_ei = compute_engagement_index(raw, epoch['data'])

                if not np.isnan(epoch_ra):
                    epoch_ras.append(epoch_ra)
                if not np.isnan(epoch_rsmr):
                    epoch_rsmrs.append(epoch_rsmr)
                if not np.isnan(epoch_ei):
                    epoch_eis.append(epoch_ei)

            if epoch_ras:
                result['epoch_ra_mean'] = np.mean(epoch_ras)
                result['epoch_ra_std'] = np.std(epoch_ras)
            if epoch_rsmrs:
                result['epoch_rsmr_mean'] = np.mean(epoch_rsmrs)
                result['epoch_rsmr_std'] = np.std(epoch_rsmrs)
            if epoch_eis:
                result['epoch_ei_mean'] = np.mean(epoch_eis)
                result['epoch_ei_std'] = np.std(epoch_eis)

            result['n_epochs'] = len(epochs)

    return result

def run_analysis():
    """Run the full frequency analysis on all subjects."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("EEG FREQUENCY BAND ANALYSIS")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 70)

    # Find all XDF files
    xdf_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.xdf'):
                xdf_files.append(os.path.join(root, f))

    print(f"\nFound {len(xdf_files)} XDF files")

    # Process each file
    results = []
    for filepath in sorted(xdf_files):
        filename = os.path.basename(filepath)
        subject_id = filename.replace('.xdf', '')

        result = analyze_subject(filepath, subject_id)
        if result:
            results.append(result)

    if not results:
        print("No valid data found!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'eeg_frequency_results.csv'), index=False)

    # Separate by condition
    treatment = df[df['condition'] == 'Treatment']
    neutral = df[df['condition'] == 'Neutral']

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nSample sizes:")
    print(f"  Treatment: {len(treatment)}")
    print(f"  Neutral: {len(neutral)}")

    # Statistical comparisons
    print(f"\n{'-' * 70}")
    print("STATISTICAL COMPARISONS")
    print(f"{'-' * 70}")

    stats_results = []

    # Key indices to compare
    indices = [
        ('relative_alpha', 'Relative Alpha (RA)'),
        ('rsmr', 'RSMR'),
        ('engagement_index', 'Engagement Index'),
        ('alpha', 'Alpha Power'),
        ('alpha_relative', 'Relative Alpha Power'),
        ('theta', 'Theta Power'),
        ('theta_relative', 'Relative Theta Power'),
        ('beta', 'Beta Power'),
        ('beta_relative', 'Relative Beta Power'),
    ]

    for col, label in indices:
        if col in df.columns:
            t_data = treatment[col].dropna()
            n_data = neutral[col].dropna()

            if len(t_data) >= 3 and len(n_data) >= 3:
                t_stat, p_val = stats.ttest_ind(t_data, n_data)

                # Cohen's d
                pooled_std = np.sqrt(((len(t_data)-1)*t_data.std()**2 +
                                     (len(n_data)-1)*n_data.std()**2) /
                                    (len(t_data) + len(n_data) - 2))
                cohens_d = (t_data.mean() - n_data.mean()) / pooled_std if pooled_std > 0 else 0

                sig = ""
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"

                print(f"\n{label}:")
                print(f"  Treatment: M = {t_data.mean():.4f}, SD = {t_data.std():.4f}")
                print(f"  Neutral:   M = {n_data.mean():.4f}, SD = {n_data.std():.4f}")
                print(f"  t({len(t_data)+len(n_data)-2}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
                print(f"  Cohen's d = {cohens_d:.3f}")

                stats_results.append({
                    'Index': label,
                    'Treatment_M': t_data.mean(),
                    'Treatment_SD': t_data.std(),
                    'Neutral_M': n_data.mean(),
                    'Neutral_SD': n_data.std(),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': p_val < 0.05
                })

    # Save statistics
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'eeg_frequency_statistics.csv'), index=False)

    # Generate visualizations
    create_visualizations(df, treatment, neutral, stats_results)

    # Generate report
    generate_report(df, treatment, neutral, stats_results)

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

def create_visualizations(df, treatment, neutral, stats_results):
    """Create visualization figures."""

    # Figure 1: Bar plot of key indices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    indices = ['relative_alpha', 'rsmr', 'engagement_index']
    labels = ['Relative Alpha (RA)', 'RSMR', 'Engagement Index']

    for ax, col, label in zip(axes, indices, labels):
        if col in df.columns:
            t_data = treatment[col].dropna()
            n_data = neutral[col].dropna()

            means = [t_data.mean(), n_data.mean()]
            stds = [t_data.std(), n_data.std()]

            bars = ax.bar(['Treatment', 'Neutral'], means, yerr=stds,
                         capsize=5, color=['#e74c3c', '#3498db'], alpha=0.7)
            ax.set_ylabel(label)
            ax.set_title(label)

            # Add significance marker if applicable
            stat = next((s for s in stats_results if label in s['Index']), None)
            if stat and stat['p_value'] < 0.05:
                max_y = max(means) + max(stds)
                ax.plot([0, 1], [max_y * 1.1, max_y * 1.1], 'k-', lw=1)
                ax.text(0.5, max_y * 1.15, '*', ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'frequency_indices_comparison.png'), dpi=150)
    plt.close()

    # Figure 2: Frequency band power comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    bands = ['delta', 'theta', 'alpha', 'smr', 'beta']
    x = np.arange(len(bands))
    width = 0.35

    t_means = [treatment[b].dropna().mean() for b in bands]
    t_stds = [treatment[b].dropna().std() for b in bands]
    n_means = [neutral[b].dropna().mean() for b in bands]
    n_stds = [neutral[b].dropna().std() for b in bands]

    ax.bar(x - width/2, t_means, width, yerr=t_stds, label='Treatment',
           color='#e74c3c', alpha=0.7, capsize=3)
    ax.bar(x + width/2, n_means, width, yerr=n_stds, label='Neutral',
           color='#3498db', alpha=0.7, capsize=3)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Power (uV^2/Hz)')
    ax.set_title('Frequency Band Power by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'frequency_band_power.png'), dpi=150)
    plt.close()

    # Figure 3: Relative band power
    fig, ax = plt.subplots(figsize=(10, 6))

    rel_bands = [f'{b}_relative' for b in bands]

    t_means = [treatment[b].dropna().mean() if b in treatment.columns else 0 for b in rel_bands]
    n_means = [neutral[b].dropna().mean() if b in neutral.columns else 0 for b in rel_bands]

    ax.bar(x - width/2, t_means, width, label='Treatment', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, n_means, width, label='Neutral', color='#3498db', alpha=0.7)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Relative Power')
    ax.set_title('Relative Frequency Band Power by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'relative_band_power.png'), dpi=150)
    plt.close()

    print(f"\nFigures saved to: {FIGURES_DIR}")

def generate_report(df, treatment, neutral, stats_results):
    """Generate a text report."""

    report_path = os.path.join(OUTPUT_DIR, 'eeg_frequency_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EEG FREQUENCY BAND ANALYSIS REPORT\n")
        f.write("Visual Priming & Deepfake Audio Perception Study\n")
        f.write("=" * 70 + "\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. SAMPLE CHARACTERISTICS\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"Total participants with valid EEG: {len(df)}\n")
        f.write(f"  Treatment condition: {len(treatment)}\n")
        f.write(f"  Neutral condition: {len(neutral)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("2. ANALYSIS PARAMETERS\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"Sampling rate: {SFREQ} Hz\n")
        f.write(f"Bandpass filter: {HIGHPASS}-{LOWPASS} Hz\n")
        f.write(f"Notch filter: 60 Hz\n\n")
        f.write("Frequency bands:\n")
        for band, (low, high) in FREQ_BANDS.items():
            f.write(f"  {band.capitalize()}: {low}-{high} Hz\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("3. KEY INDICES\n")
        f.write("-" * 70 + "\n\n")
        f.write("Relative Alpha (RA): log(Right Alpha) - log(Left Alpha)\n")
        f.write("  - Positive RA: greater left frontal activity (approach motivation)\n")
        f.write("  - Negative RA: greater right frontal activity (withdrawal)\n\n")
        f.write("RSMR: SMR Power / Alpha Power\n")
        f.write("  - Higher values indicate focused attention\n\n")
        f.write("Engagement Index: Beta / (Alpha + Theta)\n")
        f.write("  - Higher values indicate greater cognitive engagement\n\n")

        f.write("-" * 70 + "\n")
        f.write("4. RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for stat in stats_results:
            sig = ""
            if stat['p_value'] < 0.001:
                sig = "***"
            elif stat['p_value'] < 0.01:
                sig = "**"
            elif stat['p_value'] < 0.05:
                sig = "*"

            f.write(f"{stat['Index']}:\n")
            f.write(f"  Treatment: M = {stat['Treatment_M']:.4f}, SD = {stat['Treatment_SD']:.4f}\n")
            f.write(f"  Neutral:   M = {stat['Neutral_M']:.4f}, SD = {stat['Neutral_SD']:.4f}\n")
            f.write(f"  t = {stat['t_statistic']:.3f}, p = {stat['p_value']:.4f} {sig}\n")
            f.write(f"  Cohen's d = {stat['cohens_d']:.3f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("5. INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        sig_results = [s for s in stats_results if s['p_value'] < 0.05]
        if sig_results:
            f.write("Significant differences found:\n")
            for s in sig_results:
                direction = "higher" if s['Treatment_M'] > s['Neutral_M'] else "lower"
                f.write(f"  - {s['Index']}: Treatment showed {direction} values than Neutral\n")
        else:
            f.write("No significant differences found between conditions.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Statistical significance: * p < .05, ** p < .01, *** p < .001\n")
        f.write("=" * 70 + "\n")

    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    run_analysis()
