#!/usr/bin/env python3
"""
EEG Data Analysis Script for Visual Priming & Deepfake Audio Perception Study
==============================================================================

This script processes XDF format EEG data collected during the experiment.

Dependencies:
    pip install pyxdf mne numpy pandas scipy matplotlib seaborn

Usage:
    python eeg_analysis.py

Author: Generated for audio_ai experiment
Date: 2026-03-09
"""

import os
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# EEG processing libraries
try:
    import pyxdf
    import mne
    from scipy import signal, stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyxdf mne numpy pandas scipy matplotlib seaborn")
    HAS_DEPENDENCIES = False


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/Users/user/Desktop/audio_ai/gdrive_download")
OUTPUT_DIR = Path("/Users/user/Desktop/audio_ai/analysis_results")
FIGURES_DIR = OUTPUT_DIR / "figures"

# EEG parameters
SFREQ = 500  # Sampling frequency (Hz) - adjust based on your actual data
LOWCUT = 0.1  # High-pass filter cutoff (Hz)
HIGHCUT = 40  # Low-pass filter cutoff (Hz)
NOTCH_FREQ = 60  # Notch filter frequency (Hz) - use 50 for Korea/Europe

# Event-related potential (ERP) parameters
EPOCH_TMIN = -0.2  # Pre-stimulus baseline (seconds)
EPOCH_TMAX = 0.8   # Post-stimulus window (seconds)
BASELINE = (-0.2, 0)  # Baseline correction window

# ERP components of interest
ERP_WINDOWS = {
    'MMN': (0.100, 0.250),      # Mismatch Negativity
    'P300': (0.250, 0.500),     # P300 component
    'LPP': (0.400, 0.700),      # Late Positive Potential
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def find_xdf_files(data_dir: Path) -> List[Path]:
    """Find all XDF files in the data directory."""
    xdf_files = []
    for subject_dir in sorted(data_dir.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            # Standard BIDS path
            eeg_dir = subject_dir / 'ses-S001' / 'eeg'
            if eeg_dir.exists():
                xdf_files.extend(eeg_dir.glob('*.xdf'))
    return sorted(xdf_files)


def load_xdf_data(xdf_path: Path) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[list]]:
    """
    Load EEG data from XDF file.

    Returns:
        data: EEG data array (channels x samples)
        info: Stream metadata
        markers: Event markers if present
    """
    try:
        streams, header = pyxdf.load_xdf(str(xdf_path))

        eeg_data = None
        eeg_info = None
        markers = []

        for stream in streams:
            stream_type = stream['info']['type'][0].lower()

            if 'eeg' in stream_type or stream['info']['type'][0] == 'EEG':
                eeg_data = np.array(stream['time_series']).T  # channels x samples
                eeg_info = stream['info']

            elif 'marker' in stream_type or 'event' in stream_type:
                markers = list(zip(
                    stream['time_stamps'],
                    [m[0] if isinstance(m, list) else m for m in stream['time_series']]
                ))

        return eeg_data, eeg_info, markers

    except Exception as e:
        print(f"Error loading {xdf_path}: {e}")
        return None, None, None


def extract_subject_id(xdf_path: Path) -> str:
    """Extract subject ID from file path."""
    # Path format: .../sub-25_06_25_1_1/ses-S001/eeg/...
    parts = xdf_path.parts
    for part in parts:
        if part.startswith('sub-'):
            return part
    return xdf_path.stem


# =============================================================================
# Preprocessing Functions
# =============================================================================

def create_mne_raw(data: np.ndarray, sfreq: float, ch_names: Optional[List[str]] = None) -> mne.io.RawArray:
    """
    Create MNE Raw object from numpy array.

    Args:
        data: EEG data (channels x samples)
        sfreq: Sampling frequency
        ch_names: Channel names (optional)

    Returns:
        MNE Raw object
    """
    n_channels = data.shape[0]

    if ch_names is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    # Create info object
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )

    # Scale data to Volts if needed (MNE expects Volts)
    # Adjust scaling factor based on your equipment
    data_volts = data * 1e-6  # Assuming data is in microvolts

    raw = mne.io.RawArray(data_volts, info)

    return raw


def preprocess_eeg(raw: mne.io.RawArray,
                   lowcut: float = LOWCUT,
                   highcut: float = HIGHCUT,
                   notch_freq: float = NOTCH_FREQ) -> mne.io.RawArray:
    """
    Apply standard EEG preprocessing pipeline.

    Steps:
    1. Bandpass filter
    2. Notch filter (power line noise)
    3. Re-reference to average
    """
    # Copy to avoid modifying original
    raw = raw.copy()

    # Bandpass filter
    raw.filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin')

    # Notch filter for power line noise
    raw.notch_filter(freqs=notch_freq)

    # Re-reference to average
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()

    return raw


def detect_bad_channels(raw: mne.io.RawArray, threshold: float = 3.0) -> List[str]:
    """
    Detect bad channels based on variance.

    Args:
        raw: MNE Raw object
        threshold: Z-score threshold for marking bad channels

    Returns:
        List of bad channel names
    """
    data = raw.get_data()
    variances = np.var(data, axis=1)

    z_scores = stats.zscore(variances)
    bad_idx = np.where(np.abs(z_scores) > threshold)[0]

    bad_channels = [raw.ch_names[i] for i in bad_idx]

    return bad_channels


# =============================================================================
# ERP Analysis Functions
# =============================================================================

def create_epochs(raw: mne.io.RawArray,
                  events: np.ndarray,
                  event_id: Dict[str, int],
                  tmin: float = EPOCH_TMIN,
                  tmax: float = EPOCH_TMAX,
                  baseline: Tuple[float, float] = BASELINE) -> mne.Epochs:
    """
    Create epochs from continuous data.

    Args:
        raw: Preprocessed MNE Raw object
        events: Events array (n_events x 3)
        event_id: Dictionary mapping event names to IDs
        tmin, tmax: Epoch time bounds
        baseline: Baseline correction window

    Returns:
        MNE Epochs object
    """
    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None  # Add rejection thresholds if needed
    )

    return epochs


def compute_erp(epochs: mne.Epochs, condition: str = None) -> mne.Evoked:
    """
    Compute event-related potential (grand average).

    Args:
        epochs: MNE Epochs object
        condition: Condition name to average (None = all)

    Returns:
        MNE Evoked object
    """
    if condition:
        return epochs[condition].average()
    return epochs.average()


def extract_erp_amplitude(evoked: mne.Evoked,
                          time_window: Tuple[float, float],
                          channels: List[str] = None) -> float:
    """
    Extract mean amplitude in a time window.

    Args:
        evoked: MNE Evoked object
        time_window: (start, end) in seconds
        channels: List of channels to average (None = all)

    Returns:
        Mean amplitude in the window
    """
    tmin, tmax = time_window

    if channels:
        evoked = evoked.copy().pick_channels(channels)

    # Get time indices
    times = evoked.times
    idx_start = np.argmin(np.abs(times - tmin))
    idx_end = np.argmin(np.abs(times - tmax))

    # Extract data and compute mean
    data = evoked.data[:, idx_start:idx_end]
    mean_amplitude = np.mean(data) * 1e6  # Convert back to microvolts

    return mean_amplitude


# =============================================================================
# Time-Frequency Analysis
# =============================================================================

def compute_time_frequency(epochs: mne.Epochs,
                           freqs: np.ndarray = None,
                           n_cycles: int = 7) -> mne.time_frequency.AverageTFR:
    """
    Compute time-frequency representation using Morlet wavelets.

    Args:
        epochs: MNE Epochs object
        freqs: Frequencies of interest
        n_cycles: Number of wavelet cycles

    Returns:
        MNE AverageTFR object
    """
    if freqs is None:
        freqs = np.arange(4, 40, 1)  # 4-40 Hz

    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True
    )

    return power


def extract_theta_power(power: mne.time_frequency.AverageTFR,
                        time_window: Tuple[float, float] = (0.2, 0.5),
                        freq_band: Tuple[float, float] = (4, 8)) -> float:
    """
    Extract theta band power in a time window.

    Args:
        power: MNE AverageTFR object
        time_window: (start, end) in seconds
        freq_band: (low, high) frequency bounds

    Returns:
        Mean theta power
    """
    # Get indices
    times = power.times
    freqs = power.freqs

    time_idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
    freq_idx = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]

    # Extract and average
    data = power.data[:, freq_idx, :][:, :, time_idx]
    mean_power = np.mean(data)

    return mean_power


# =============================================================================
# Statistical Analysis
# =============================================================================

def compare_conditions(group_a_data: np.ndarray,
                       group_b_data: np.ndarray) -> Dict[str, float]:
    """
    Compare two experimental conditions using t-test and effect size.

    Args:
        group_a_data: Data from Group A (visual priming)
        group_b_data: Data from Group B (control)

    Returns:
        Dictionary with t-statistic, p-value, and Cohen's d
    """
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data)

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(group_a_data) - 1) * np.var(group_a_data) +
         (len(group_b_data) - 1) * np.var(group_b_data)) /
        (len(group_a_data) + len(group_b_data) - 2)
    )
    cohens_d = (np.mean(group_a_data) - np.mean(group_b_data)) / pooled_std

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_erp_comparison(evoked_a: mne.Evoked,
                        evoked_b: mne.Evoked,
                        title: str = "ERP Comparison",
                        save_path: Path = None):
    """
    Plot ERP comparison between two conditions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = evoked_a.times * 1000  # Convert to ms

    # Average across channels
    data_a = evoked_a.data.mean(axis=0) * 1e6  # Convert to µV
    data_b = evoked_b.data.mean(axis=0) * 1e6

    ax.plot(times, data_a, label='Group A (Visual Priming)', color='#E24A33', linewidth=2)
    ax.plot(times, data_b, label='Group B (Control)', color='#348ABD', linewidth=2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, label='Stimulus onset')

    # Mark ERP windows
    for name, (t1, t2) in ERP_WINDOWS.items():
        ax.axvspan(t1 * 1000, t2 * 1000, alpha=0.2, label=f'{name} window')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()  # EEG convention: negative up

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()


def plot_topography(evoked: mne.Evoked,
                    time_window: Tuple[float, float],
                    title: str = "Topography",
                    save_path: Path = None):
    """
    Plot scalp topography for a time window.
    """
    fig = evoked.plot_topomap(
        times=np.mean(time_window),
        average=time_window[1] - time_window[0],
        title=title,
        show=False
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_subject(xdf_path: Path,
                    output_dir: Path = OUTPUT_DIR) -> Optional[Dict]:
    """
    Process a single subject's EEG data.

    Args:
        xdf_path: Path to XDF file
        output_dir: Directory for saving results

    Returns:
        Dictionary with subject results
    """
    subject_id = extract_subject_id(xdf_path)
    print(f"\nProcessing: {subject_id}")

    # Load data
    data, info, markers = load_xdf_data(xdf_path)

    if data is None:
        print(f"  Skipping {subject_id}: Could not load data")
        return None

    print(f"  Data shape: {data.shape}")
    print(f"  Markers found: {len(markers) if markers else 0}")

    # Skip if no data
    if data.shape[1] == 0:
        print(f"  Skipping {subject_id}: Empty data file")
        return None

    # Get sampling rate from metadata
    try:
        sfreq = float(info['nominal_srate'][0])
    except (KeyError, TypeError):
        sfreq = SFREQ
        print(f"  Using default sampling rate: {sfreq} Hz")

    # Create MNE Raw object
    raw = create_mne_raw(data, sfreq)

    # Detect bad channels
    bad_channels = detect_bad_channels(raw)
    if bad_channels:
        print(f"  Bad channels detected: {bad_channels}")
        raw.info['bads'] = bad_channels

    # Preprocess
    raw_preprocessed = preprocess_eeg(raw)

    # Store results
    results = {
        'subject_id': subject_id,
        'n_channels': data.shape[0],
        'n_samples': data.shape[1],
        'sfreq': sfreq,
        'duration_s': data.shape[1] / sfreq,
        'n_markers': len(markers) if markers else 0,
        'bad_channels': bad_channels,
    }

    return results


def run_batch_processing():
    """
    Run batch processing on all subjects.
    """
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all XDF files
    xdf_files = find_xdf_files(DATA_DIR)
    print(f"Found {len(xdf_files)} XDF files")

    # Process each subject
    all_results = []

    for xdf_path in xdf_files:
        result = process_subject(xdf_path)
        if result:
            all_results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    # Save summary
    summary_path = OUTPUT_DIR / 'processing_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total subjects processed: {len(df)}")
    print(f"Average duration: {df['duration_s'].mean():.1f} seconds")
    print(f"Subjects with bad channels: {(df['bad_channels'].apply(len) > 0).sum()}")

    return df


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if not HAS_DEPENDENCIES:
        print("\nPlease install required dependencies and try again.")
        exit(1)

    print("=" * 60)
    print("EEG Analysis Pipeline")
    print("Visual Priming & Deepfake Audio Perception Study")
    print("=" * 60)

    # Check data directory
    if not DATA_DIR.exists():
        print(f"\nError: Data directory not found: {DATA_DIR}")
        exit(1)

    # Run batch processing
    results_df = run_batch_processing()

    print("\nAnalysis complete!")
