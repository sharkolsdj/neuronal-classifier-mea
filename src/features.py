"""
Electrophysiological feature extraction from patch-clamp / MEA recordings.

Features are computed from raw voltage traces and spike trains, replicating
the manual classification workflow used in the Baldelli lab (UniGe) and the
Benfenati lab (IIT, Genova) for excitatory/inhibitory neuron identification.

References:
    - Gouwens et al. (2019). Nature Neuroscience.
    - Prestigio C*, Ferrante D* et al. (2022). eLife. doi:10.7554/eLife.69058
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import variation
from typing import Optional


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def detect_spikes(
    voltage: np.ndarray,
    threshold: float = -20.0,
    sampling_rate: float = 20000.0,
    refractory_ms: float = 2.0,
) -> np.ndarray:
    """
    Detect action potential peaks in a voltage trace.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage trace in mV, shape (n_samples,).
    threshold : float
        Minimum peak voltage in mV for spike detection. Default -20 mV.
    sampling_rate : float
        Sampling rate in Hz. Default 20 kHz (standard patch-clamp).
    refractory_ms : float
        Absolute refractory period in ms. Spikes closer than this are merged.

    Returns
    -------
    spike_indices : np.ndarray
        Sample indices of detected spike peaks.
    """
    refractory_samples = int(refractory_ms * sampling_rate / 1000)
    peaks, _ = find_peaks(voltage, height=threshold, distance=refractory_samples)
    return peaks


# ---------------------------------------------------------------------------
# ISI-based features
# ---------------------------------------------------------------------------

def compute_isi_features(
    spike_indices: np.ndarray,
    sampling_rate: float = 20000.0,
) -> dict[str, float]:
    """
    Compute inter-spike interval (ISI) statistics from spike train.

    ISI statistics are among the most reliable discriminators between
    excitatory pyramidal cells (irregular, low-frequency) and inhibitory
    interneurons (fast-spiking, regular). See Gouwens et al. (2019).

    Parameters
    ----------
    spike_indices : np.ndarray
        Sample indices of spikes, shape (n_spikes,).
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    features : dict
        Dictionary of ISI-derived features.
    """
    if len(spike_indices) < 2:
        return {
            "mean_isi_ms": np.nan,
            "std_isi_ms": np.nan,
            "cv_isi": np.nan,
            "mean_firing_rate_hz": 0.0,
            "adaptation_index": np.nan,
            "isi_ratio": np.nan,
        }

    isis_ms = np.diff(spike_indices) / sampling_rate * 1000.0  # convert to ms

    # Adaptation index: ratio of last ISI to first ISI (>1 = adapting = likely excitatory)
    adaptation_index = isis_ms[-1] / isis_ms[0] if isis_ms[0] > 0 else np.nan

    # ISI ratio: max/min ISI — high values indicate burst-pause patterns
    isi_ratio = isis_ms.max() / isis_ms.min() if isis_ms.min() > 0 else np.nan

    duration_s = (spike_indices[-1] - spike_indices[0]) / sampling_rate
    mean_fr = (len(spike_indices) - 1) / duration_s if duration_s > 0 else 0.0

    return {
        "mean_isi_ms": float(np.mean(isis_ms)),
        "std_isi_ms": float(np.std(isis_ms)),
        "cv_isi": float(variation(isis_ms)) if len(isis_ms) > 1 else np.nan,
        "mean_firing_rate_hz": float(mean_fr),
        "adaptation_index": float(adaptation_index),
        "isi_ratio": float(isi_ratio),
    }


# ---------------------------------------------------------------------------
# Spike shape features
# ---------------------------------------------------------------------------

def compute_spike_shape_features(
    voltage: np.ndarray,
    spike_indices: np.ndarray,
    sampling_rate: float = 20000.0,
    window_ms: float = 3.0,
) -> dict[str, float]:
    """
    Compute action potential waveform shape features.

    Spike shape — particularly AP half-width and AHP depth — is a key
    discriminator between fast-spiking interneurons (narrow AP, deep AHP)
    and regular-spiking excitatory neurons (broad AP, shallow AHP).

    Parameters
    ----------
    voltage : np.ndarray
        Full voltage trace in mV.
    spike_indices : np.ndarray
        Peak indices of detected spikes.
    sampling_rate : float
        Sampling rate in Hz.
    window_ms : float
        Window around each spike peak for waveform extraction, in ms.

    Returns
    -------
    features : dict
        Mean spike shape features across all detected spikes.
    """
    if len(spike_indices) == 0:
        return {
            "ap_peak_mv": np.nan,
            "ap_threshold_mv": np.nan,
            "ap_half_width_ms": np.nan,
            "ahp_depth_mv": np.nan,
            "ap_rise_time_ms": np.nan,
        }

    window_samples = int(window_ms * sampling_rate / 1000)
    peaks, thresholds, half_widths, ahp_depths, rise_times = [], [], [], [], []

    for idx in spike_indices:
        start = max(0, idx - window_samples // 2)
        end = min(len(voltage), idx + window_samples // 2)
        waveform = voltage[start:end]

        if len(waveform) < 4:
            continue

        peak_mv = float(waveform.max())
        peaks.append(peak_mv)

        # Threshold: voltage at 10% of peak-to-baseline rise (simplified)
        baseline = float(waveform[:5].mean())
        threshold_mv = baseline + 0.1 * (peak_mv - baseline)
        thresholds.append(threshold_mv)

        # Half-width: duration at half peak amplitude
        half_amp = baseline + 0.5 * (peak_mv - baseline)
        above_half = waveform >= half_amp
        if above_half.any():
            half_width_samples = int(above_half.sum())
            half_widths.append(half_width_samples / sampling_rate * 1000.0)

        # AHP depth: minimum voltage after spike peak relative to baseline
        peak_local = int(waveform.argmax())
        post_spike = waveform[peak_local:]
        if len(post_spike) > 0:
            ahp_depths.append(float(post_spike.min()) - baseline)

        # Rise time: time from threshold crossing to peak (simplified)
        above_thresh = np.where(waveform >= threshold_mv)[0]
        if len(above_thresh) > 0:
            rise_time_samples = int(waveform.argmax()) - int(above_thresh[0])
            rise_times.append(rise_time_samples / sampling_rate * 1000.0)

    return {
        "ap_peak_mv": float(np.nanmean(peaks)) if peaks else np.nan,
        "ap_threshold_mv": float(np.nanmean(thresholds)) if thresholds else np.nan,
        "ap_half_width_ms": float(np.nanmean(half_widths)) if half_widths else np.nan,
        "ahp_depth_mv": float(np.nanmean(ahp_depths)) if ahp_depths else np.nan,
        "ap_rise_time_ms": float(np.nanmean(rise_times)) if rise_times else np.nan,
    }


# ---------------------------------------------------------------------------
# Subthreshold features
# ---------------------------------------------------------------------------

def compute_subthreshold_features(
    voltage: np.ndarray,
    current_injection: np.ndarray,
    sampling_rate: float = 20000.0,
) -> dict[str, float]:
    """
    Compute passive membrane properties from subthreshold current injection.

    Input resistance (Rin) is a key parameter distinguishing large pyramidal
    cells (low Rin, high capacitance) from small interneurons (high Rin).

    Parameters
    ----------
    voltage : np.ndarray
        Voltage trace in mV during hyperpolarising step, shape (n_samples,).
    current_injection : float or np.ndarray
        Injected current in pA.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    features : dict
        Passive membrane features.
    """
    baseline = float(np.mean(voltage[:int(0.05 * sampling_rate)]))
    steady_state = float(np.mean(voltage[-int(0.05 * sampling_rate):]))
    delta_v = steady_state - baseline

    # Current step amplitude
    if hasattr(current_injection, '__len__'):
        delta_i = float(np.mean(current_injection)) * 1e-12  # pA to A
    else:
        delta_i = float(current_injection) * 1e-12

    # Input resistance in MOhm
    rin_mohm = (delta_v * 1e-3) / delta_i * 1e-6 if delta_i != 0 else np.nan

    # Sag ratio (Ih current indicator — HCN channel activity)
    peak_hyperpol = float(voltage.min())
    sag_ratio = (steady_state - peak_hyperpol) / (baseline - peak_hyperpol) if (baseline - peak_hyperpol) != 0 else np.nan

    return {
        "input_resistance_mohm": float(rin_mohm) if not np.isnan(rin_mohm) else np.nan,
        "sag_ratio": float(sag_ratio) if not np.isnan(sag_ratio) else np.nan,
        "resting_potential_mv": float(baseline),
    }


# ---------------------------------------------------------------------------
# Full feature extraction pipeline
# ---------------------------------------------------------------------------

def extract_all_features(
    voltage: np.ndarray,
    sampling_rate: float = 20000.0,
    spike_threshold_mv: float = -20.0,
    current_injection: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Extract the full feature set from a single neuron voltage trace.

    This is the main entry point. It runs spike detection, then computes
    ISI features, spike shape features, and (if current injection is
    provided) subthreshold passive membrane features.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage trace in mV, shape (n_samples,).
    sampling_rate : float
        Sampling rate in Hz.
    spike_threshold_mv : float
        Spike detection threshold in mV.
    current_injection : np.ndarray, optional
        Current trace in pA. If provided, passive features are computed.

    Returns
    -------
    features : dict
        All features as a flat dictionary, ready for DataFrame row.
    """
    spikes = detect_spikes(voltage, threshold=spike_threshold_mv, sampling_rate=sampling_rate)

    isi_feats = compute_isi_features(spikes, sampling_rate)
    shape_feats = compute_spike_shape_features(voltage, spikes, sampling_rate)

    features = {**isi_feats, **shape_feats, "n_spikes": len(spikes)}

    if current_injection is not None:
        sub_feats = compute_subthreshold_features(voltage, current_injection, sampling_rate)
        features.update(sub_feats)

    return features


def build_feature_matrix(
    recordings: list[dict],
    sampling_rate: float = 20000.0,
) -> pd.DataFrame:
    """
    Build a feature matrix from a list of recording dictionaries.

    Parameters
    ----------
    recordings : list of dict
        Each dict must have 'voltage' (np.ndarray) and optionally
        'current' (np.ndarray) and 'label' (str) keys.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    df : pd.DataFrame
        Feature matrix, one row per cell.
    """
    rows = []
    for rec in recordings:
        feats = extract_all_features(
            voltage=rec["voltage"],
            sampling_rate=sampling_rate,
            current_injection=rec.get("current"),
        )
        if "label" in rec:
            feats["label"] = rec["label"]
        if "cell_id" in rec:
            feats["cell_id"] = rec["cell_id"]
        rows.append(feats)

    return pd.DataFrame(rows)
