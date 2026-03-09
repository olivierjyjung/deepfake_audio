#!/usr/bin/env python3
"""
EEG ERP Analysis Script for Visual Priming & Deepfake Audio Perception Study
=============================================================================

This script performs full ERP analysis comparing Treatment vs Neutral conditions.

Components analyzed:
- MMN (Mismatch Negativity): 100-250ms
- P300: 250-500ms
- LPP (Late Positive Potential): 400-700ms
- Theta power (4-8 Hz)
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pyxdf
import mne
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/Users/user/Desktop/audio_ai/gdrive_download")
OUTPUT_DIR = Path("/Users/user/Desktop/audio_ai/analysis_results")
FIGURES_DIR = OUTPUT_DIR / "figures" / "eeg"
PARTICIPANT_FILE = Path("/Users/user/Desktop/audio_ai/participant_group for Audio Task.csv")

# EEG parameters
SFREQ = 300  # Actual sampling frequency from data
LOWCUT = 0.1
HIGHCUT = 40
NOTCH_FREQ = 60

# Epoch parameters
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
BASELINE = (-0.2, 0)

# ERP windows (in seconds)
ERP_WINDOWS = {
    'MMN': (0.100, 0.250),
    'P300': (0.250, 0.500),
    'LPP': (0.400, 0.700),
}

# Channel regions for ERP analysis (assuming standard layout approximation)
# DSI-24 typical layout - using channel indices
FRONTAL_CHANNELS = ['EEG001', 'EEG002', 'EEG003', 'EEG004', 'EEG005', 'EEG006']  # Front region
CENTRAL_CHANNELS = ['EEG007', 'EEG008', 'EEG009', 'EEG010', 'EEG011', 'EEG012']  # Central
PARIETAL_CHANNELS = ['EEG013', 'EEG014', 'EEG015', 'EEG016', 'EEG017', 'EEG018']  # Parietal

# Component-specific channel selections
COMPONENT_CHANNELS = {
    'MMN': FRONTAL_CHANNELS,      # MMN is typically frontal
    'P300': PARIETAL_CHANNELS,    # P300 is typically parietal
    'LPP': CENTRAL_CHANNELS + PARIETAL_CHANNELS,  # LPP is centro-parietal
}

# Markers of interest
AUDIO_MARKERS = ['audio_1_stim_on', 'audio_2_stim_on', 'audio_3_stim_on', 'audio_4_stim_on']

# =============================================================================
# Helper Functions
# =============================================================================

def load_participant_groups() -> Dict[str, str]:
    """Load participant to condition mapping."""
    df = pd.read_csv(PARTICIPANT_FILE)
    mapping = {}
    for _, row in df.iterrows():
        pid = str(row['Participant ID']).strip()
        group = row['Condition']
        if pd.notna(group) and str(group).strip():
            group_num = int(float(group))
            condition = 'Treatment' if group_num <= 8 else 'Neutral'
            mapping[pid] = condition
    return mapping


def find_xdf_files() -> List[Tuple[Path, str]]:
    """Find all XDF files and their participant IDs."""
    xdf_files = []
    for subject_dir in sorted(DATA_DIR.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            eeg_dir = subject_dir / 'ses-S001' / 'eeg'
            if eeg_dir.exists():
                for xdf_file in eeg_dir.glob('*.xdf'):
                    # Extract participant ID from folder name
                    pid = subject_dir.name.replace('sub-', '')
                    xdf_files.append((xdf_file, pid))
    return xdf_files


def load_xdf_data(xdf_path: Path) -> Tuple[Optional[np.ndarray], float, List[Tuple[float, str]]]:
    """Load EEG data and markers from XDF file."""
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))

        eeg_data = None
        sfreq = SFREQ
        markers = []

        for stream in streams:
            stream_type = stream['info']['type'][0].lower()

            if 'eeg' in stream_type:
                eeg_data = np.array(stream['time_series']).T
                try:
                    sfreq = float(stream['info']['nominal_srate'][0])
                except:
                    pass
                eeg_times = stream['time_stamps']

            elif 'marker' in stream_type:
                for ts, marker in zip(stream['time_stamps'], stream['time_series']):
                    marker_val = marker[0] if isinstance(marker, list) else marker
                    markers.append((ts, marker_val))

        return eeg_data, sfreq, markers, eeg_times

    except Exception as e:
        print(f"Error loading {xdf_path}: {e}")
        return None, SFREQ, [], None


def create_mne_raw(data: np.ndarray, sfreq: float) -> mne.io.RawArray:
    """Create MNE Raw object."""
    n_channels = data.shape[0]
    ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    data_volts = data * 1e-6  # Convert to Volts

    return mne.io.RawArray(data_volts, info, verbose=False)


def preprocess_raw(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Apply preprocessing pipeline."""
    raw = raw.copy()
    raw.filter(l_freq=LOWCUT, h_freq=HIGHCUT, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    # Don't apply average reference - keep original reference for meaningful amplitudes
    return raw


def markers_to_events(markers: List[Tuple[float, str]], eeg_times: np.ndarray,
                      sfreq: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert markers to MNE events array."""
    # Use consistent event_id mapping
    event_id = {
        'audio_1_stim_on': 1,
        'audio_2_stim_on': 2,
        'audio_3_stim_on': 3,
        'audio_4_stim_on': 4,
    }
    events_list = []

    for ts, marker in markers:
        if marker in AUDIO_MARKERS:
            # Find closest sample
            sample_idx = np.argmin(np.abs(eeg_times - ts))
            events_list.append([sample_idx, 0, event_id[marker]])

    if events_list:
        return np.array(events_list), event_id
    return np.array([]).reshape(0, 3), event_id


def extract_erp_amplitude(evoked: mne.Evoked, time_window: Tuple[float, float],
                          channels: List[str] = None) -> float:
    """Extract mean amplitude in time window for specific channels."""
    tmin, tmax = time_window
    times = evoked.times
    idx_start = np.argmin(np.abs(times - tmin))
    idx_end = np.argmin(np.abs(times - tmax))

    if channels:
        # Pick only specified channels that exist
        available_channels = [ch for ch in channels if ch in evoked.ch_names]
        if available_channels:
            evoked_subset = evoked.copy().pick_channels(available_channels)
            data = evoked_subset.data[:, idx_start:idx_end]
        else:
            data = evoked.data[:, idx_start:idx_end]
    else:
        data = evoked.data[:, idx_start:idx_end]

    return np.mean(data) * 1e6  # Convert to microvolts


# =============================================================================
# Main Analysis
# =============================================================================

def process_all_subjects():
    """Process all subjects and extract ERP measures."""

    print("=" * 60)
    print("EEG ERP Analysis")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load participant groups
    participant_groups = load_participant_groups()
    print(f"\nLoaded {len(participant_groups)} participant-group mappings")

    # Find XDF files
    xdf_files = find_xdf_files()
    print(f"Found {len(xdf_files)} XDF files")

    # Storage for results
    all_epochs_treatment = []
    all_epochs_neutral = []
    results = []

    for xdf_path, pid in xdf_files:
        # Get condition
        condition = participant_groups.get(pid)
        if condition is None:
            print(f"  Skipping {pid}: No condition mapping")
            continue

        print(f"\nProcessing: {pid} ({condition})")

        # Load data
        eeg_data, sfreq, markers, eeg_times = load_xdf_data(xdf_path)

        if eeg_data is None or eeg_data.shape[1] == 0:
            print(f"  Skipping: No data")
            continue

        if eeg_times is None:
            print(f"  Skipping: No timestamps")
            continue

        # Check for audio markers
        audio_markers = [m for _, m in markers if m in AUDIO_MARKERS]
        if len(audio_markers) < 2:
            print(f"  Skipping: Only {len(audio_markers)} audio markers")
            continue

        print(f"  Data shape: {eeg_data.shape}, Markers: {len(audio_markers)}")

        # Create and preprocess raw
        raw = create_mne_raw(eeg_data, sfreq)
        raw = preprocess_raw(raw)

        # Create events
        events, event_id = markers_to_events(markers, eeg_times, sfreq)

        if len(events) == 0:
            print(f"  Skipping: No valid events")
            continue

        # Create epochs
        try:
            epochs = mne.Epochs(raw, events, event_id,
                               tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                               baseline=BASELINE, preload=True,
                               verbose=False)

            if len(epochs) == 0:
                print(f"  Skipping: No valid epochs")
                continue

            print(f"  Created {len(epochs)} epochs")

            # Store epochs by condition
            if condition == 'Treatment':
                all_epochs_treatment.append(epochs)
            else:
                all_epochs_neutral.append(epochs)

            # Compute subject-level ERP
            evoked = epochs.average()

            # Extract component amplitudes
            subject_result = {
                'participant_id': pid,
                'condition': condition,
                'n_epochs': len(epochs),
            }

            for comp_name, time_window in ERP_WINDOWS.items():
                channels = COMPONENT_CHANNELS.get(comp_name)
                amp = extract_erp_amplitude(evoked, time_window, channels)
                subject_result[f'{comp_name}_amplitude'] = amp

            results.append(subject_result)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ==========================================================================
    # Grand Average ERPs
    # ==========================================================================

    print("\n" + "=" * 60)
    print("Computing Grand Average ERPs")
    print("=" * 60)

    # Concatenate epochs
    if all_epochs_treatment and all_epochs_neutral:
        epochs_treatment = mne.concatenate_epochs(all_epochs_treatment)
        epochs_neutral = mne.concatenate_epochs(all_epochs_neutral)

        print(f"Treatment epochs: {len(epochs_treatment)}")
        print(f"Neutral epochs: {len(epochs_neutral)}")

        # Grand averages
        evoked_treatment = epochs_treatment.average()
        evoked_neutral = epochs_neutral.average()

        # =======================================================================
        # Statistical Analysis
        # =======================================================================

        print("\n" + "=" * 60)
        print("Statistical Analysis")
        print("=" * 60)

        results_df = pd.DataFrame(results)

        stats_results = {}

        for comp_name in ERP_WINDOWS.keys():
            col = f'{comp_name}_amplitude'

            treatment_vals = results_df[results_df['condition'] == 'Treatment'][col].dropna()
            neutral_vals = results_df[results_df['condition'] == 'Neutral'][col].dropna()

            if len(treatment_vals) > 1 and len(neutral_vals) > 1:
                t_stat, p_val = stats.ttest_ind(treatment_vals, neutral_vals)

                # Cohen's d
                pooled_std = np.sqrt(((len(treatment_vals)-1)*treatment_vals.var() +
                                      (len(neutral_vals)-1)*neutral_vals.var()) /
                                     (len(treatment_vals) + len(neutral_vals) - 2))
                cohens_d = (treatment_vals.mean() - neutral_vals.mean()) / pooled_std if pooled_std > 0 else 0

                stats_results[comp_name] = {
                    'treatment_mean': treatment_vals.mean(),
                    'treatment_sd': treatment_vals.std(),
                    'neutral_mean': neutral_vals.mean(),
                    'neutral_sd': neutral_vals.std(),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'n_treatment': len(treatment_vals),
                    'n_neutral': len(neutral_vals),
                }

                sig = "*" if p_val < 0.05 else ""
                print(f"\n{comp_name}:")
                print(f"  Treatment: M = {treatment_vals.mean():.3f}, SD = {treatment_vals.std():.3f} (n={len(treatment_vals)})")
                print(f"  Neutral:   M = {neutral_vals.mean():.3f}, SD = {neutral_vals.std():.3f} (n={len(neutral_vals)})")
                print(f"  t({len(treatment_vals)+len(neutral_vals)-2}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
                print(f"  Cohen's d = {cohens_d:.3f}")

        # =======================================================================
        # Generate Figures
        # =======================================================================

        print("\n" + "=" * 60)
        print("Generating Figures")
        print("=" * 60)

        # ERP Comparison Plot - using central-parietal channels
        fig, ax = plt.subplots(figsize=(12, 6))

        # Select representative channels for plotting
        plot_channels = CENTRAL_CHANNELS + PARIETAL_CHANNELS
        available_treatment = [ch for ch in plot_channels if ch in evoked_treatment.ch_names]
        available_neutral = [ch for ch in plot_channels if ch in evoked_neutral.ch_names]

        evoked_treatment_subset = evoked_treatment.copy().pick_channels(available_treatment)
        evoked_neutral_subset = evoked_neutral.copy().pick_channels(available_neutral)

        times = evoked_treatment_subset.times * 1000  # Convert to ms
        data_treatment = evoked_treatment_subset.data.mean(axis=0) * 1e6
        data_neutral = evoked_neutral_subset.data.mean(axis=0) * 1e6

        ax.plot(times, data_treatment, label='Treatment (AI Cover Priming)',
                color='#E24A33', linewidth=2)
        ax.plot(times, data_neutral, label='Neutral (Classical/Synthpop)',
                color='#348ABD', linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

        # Shade ERP windows
        colors = {'MMN': 'green', 'P300': 'orange', 'LPP': 'purple'}
        for name, (t1, t2) in ERP_WINDOWS.items():
            ax.axvspan(t1*1000, t2*1000, alpha=0.15, color=colors[name], label=f'{name} window')

        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title('Grand Average ERP: Treatment vs Neutral Condition', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(-200, 800)
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'erp_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'erp_comparison.png'}")
        plt.close()

        # Component Bar Plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for idx, (comp_name, stats_data) in enumerate(stats_results.items()):
            ax = axes[idx]

            means = [stats_data['treatment_mean'], stats_data['neutral_mean']]
            sds = [stats_data['treatment_sd'], stats_data['neutral_sd']]

            bars = ax.bar(['Treatment', 'Neutral'], means, yerr=sds,
                         color=['#E24A33', '#348ABD'], capsize=5, alpha=0.8)

            ax.set_ylabel('Amplitude (μV)')
            ax.set_title(f'{comp_name}\nt={stats_data["t_statistic"]:.2f}, p={stats_data["p_value"]:.3f}')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            if stats_data['p_value'] < 0.05:
                ax.annotate('*', xy=(0.5, max(means) + max(sds) * 1.1),
                           fontsize=20, ha='center')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'erp_components_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'erp_components_comparison.png'}")
        plt.close()

        # =======================================================================
        # Save Results
        # =======================================================================

        # Save individual results
        results_df.to_csv(OUTPUT_DIR / 'eeg_erp_results.csv', index=False)
        print(f"\nSaved: {OUTPUT_DIR / 'eeg_erp_results.csv'}")

        # Save statistics summary
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(OUTPUT_DIR / 'eeg_erp_statistics.csv')
        print(f"Saved: {OUTPUT_DIR / 'eeg_erp_statistics.csv'}")

        # Generate report
        report_path = OUTPUT_DIR / 'eeg_erp_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EEG ERP ANALYSIS REPORT\n")
            f.write("Visual Priming & Deepfake Audio Perception Study\n")
            f.write("=" * 70 + "\n\n")

            f.write("-" * 70 + "\n")
            f.write("1. SAMPLE CHARACTERISTICS\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Total participants with valid EEG: {len(results_df)}\n")
            f.write(f"  Treatment condition: {len(results_df[results_df['condition']=='Treatment'])}\n")
            f.write(f"  Neutral condition: {len(results_df[results_df['condition']=='Neutral'])}\n")
            f.write(f"\nTotal epochs analyzed:\n")
            f.write(f"  Treatment: {len(epochs_treatment)}\n")
            f.write(f"  Neutral: {len(epochs_neutral)}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("2. EEG RECORDING PARAMETERS\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"Channels: 24 (EEG)\n")
            f.write(f"Sampling rate: {SFREQ} Hz\n")
            f.write(f"Bandpass filter: {LOWCUT}-{HIGHCUT} Hz\n")
            f.write(f"Notch filter: {NOTCH_FREQ} Hz\n")
            f.write(f"Reference: Average\n")
            f.write(f"Epoch window: {EPOCH_TMIN*1000:.0f} to {EPOCH_TMAX*1000:.0f} ms\n")
            f.write(f"Baseline correction: {BASELINE[0]*1000:.0f} to {BASELINE[1]*1000:.0f} ms\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("3. ERP COMPONENT ANALYSIS\n")
            f.write("-" * 70 + "\n\n")

            for comp_name, stats_data in stats_results.items():
                f.write(f"{comp_name} ({ERP_WINDOWS[comp_name][0]*1000:.0f}-{ERP_WINDOWS[comp_name][1]*1000:.0f} ms):\n")
                f.write(f"  Treatment: M = {stats_data['treatment_mean']:.3f} μV, SD = {stats_data['treatment_sd']:.3f}\n")
                f.write(f"  Neutral:   M = {stats_data['neutral_mean']:.3f} μV, SD = {stats_data['neutral_sd']:.3f}\n")
                f.write(f"  t({stats_data['n_treatment']+stats_data['n_neutral']-2}) = {stats_data['t_statistic']:.3f}, ")
                f.write(f"p = {stats_data['p_value']:.4f}")
                if stats_data['p_value'] < 0.05:
                    f.write(" *")
                if stats_data['p_value'] < 0.01:
                    f.write("*")
                if stats_data['p_value'] < 0.001:
                    f.write("*")
                f.write(f"\n  Cohen's d = {stats_data['cohens_d']:.3f}\n\n")

            f.write("-" * 70 + "\n")
            f.write("4. INTERPRETATION\n")
            f.write("-" * 70 + "\n\n")

            sig_components = [c for c, s in stats_results.items() if s['p_value'] < 0.05]
            if sig_components:
                f.write(f"Significant differences found in: {', '.join(sig_components)}\n\n")
            else:
                f.write("No significant differences found between conditions.\n\n")

            f.write("=" * 70 + "\n")
            f.write("Statistical significance: * p < .05, ** p < .01, *** p < .001\n")
            f.write("=" * 70 + "\n")

        print(f"Saved: {report_path}")

        return results_df, stats_results

    else:
        print("ERROR: Not enough data for analysis")
        return None, None


if __name__ == "__main__":
    results_df, stats_results = process_all_subjects()
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
