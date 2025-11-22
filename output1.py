"""
EEG Feature Extraction for Response Time Prediction

This module provides three complementary feature extraction functions for EEG data
that capture different aspects of neural activity relevant to cognitive performance
and response time prediction using Generalized Linear Models.

Functions:
    extract_spectral_features: Frequency-domain oscillatory patterns
    extract_temporal_statistical_features: Time-domain statistical and morphological features
    extract_connectivity_features: Inter-channel relationships and spatial patterns
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, welch
import warnings

warnings.filterwarnings('ignore')


def extract_spectral_features(data, sampling_freq=250):
    """
    Extract frequency-domain features capturing oscillatory patterns in EEG data.
    
    Spectral features reflect the organization of neural oscillations across standard
    frequency bands, which are known correlates of cognitive state, arousal, and
    motor preparation relevant to response time.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_timepoints)
        EEG data array
    sampling_freq : int, default=250
        Sampling frequency in Hz
    
    Returns
    -------
    features : ndarray, shape (n_features,)
        Concatenated spectral features normalized for GLM input
        Order: absolute powers, relative powers, spectral edges, peak metrics
    """
    
    # Validate input
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    if data.shape[1] < sampling_freq // 2:  # Minimum 200ms window
        raise ValueError(f"Window too short: {data.shape[1]} timepoints")
    
    n_channels = data.shape[0]
    features = []
    
    # Define frequency bands (Hz)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Handle NaN values
    data_clean = np.copy(data)
    for ch in range(n_channels):
        if np.isnan(data_clean[ch]).any():
            data_clean[ch] = np.nanmean(data_clean[ch]) * np.ones_like(data_clean[ch])
    
    # Compute PSD using Welch's method for each channel
    freqs = None
    psd_per_channel = []
    
    for ch in range(n_channels):
        # Check for flat signal
        if np.var(data_clean[ch]) < 1e-10:
            # Generate dummy PSD to maintain structure
            freqs = np.linspace(0, sampling_freq / 2, 129)
            psd = np.ones_like(freqs) * 1e-10
        else:
            freqs, psd = welch(data_clean[ch], fs=sampling_freq, 
                              window='hamming', nperseg=min(256, data.shape[1]))
        
        psd_per_channel.append(psd)
    
    psd_per_channel = np.array(psd_per_channel)
    
    # Extract absolute and relative band powers
    absolute_powers = []
    relative_powers = []
    peak_freqs_alpha = []
    peak_freqs_beta = []
    peak_powers_alpha = []
    peak_powers_beta = []
    spectral_edges = []
    
    for ch in range(n_channels):
        psd = psd_per_channel[ch]
        total_power = np.trapz(psd, freqs)
        
        if total_power < 1e-10:
            total_power = 1e-10
        
        # Absolute and relative band powers
        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            band_power = np.trapz(psd[mask], freqs[mask]) if mask.sum() > 0 else 1e-10
            absolute_powers.append(np.log10(band_power + 1e-12))  # Log scale
            relative_powers.append(band_power / total_power)
        
        # Peak frequency and power in alpha and beta bands
        for band_name, (f_low, f_high) in [('alpha', bands['alpha']), 
                                           ('beta', bands['beta'])]:
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() > 0:
                peak_idx = np.argmax(psd[mask])
                peak_freq = freqs[mask][peak_idx]
                peak_power = psd[mask][peak_idx]
                if band_name == 'alpha':
                    peak_freqs_alpha.append(peak_freq)
                    peak_powers_alpha.append(np.log10(peak_power + 1e-12))
                else:
                    peak_freqs_beta.append(peak_freq)
                    peak_powers_beta.append(np.log10(peak_power + 1e-12))
            else:
                if band_name == 'alpha':
                    peak_freqs_alpha.append(10.5)  # Center frequency
                    peak_powers_alpha.append(np.log10(1e-12))
                else:
                    peak_freqs_beta.append(21.5)
                    peak_powers_beta.append(np.log10(1e-12))
        
        # Spectral edge frequency (95% power cutoff)
        cumsum = np.cumsum(psd)
        cumsum = cumsum / cumsum[-1]
        edge_idx = np.where(cumsum >= 0.95)[0][0] if (cumsum >= 0.95).any() else len(freqs) - 1
        spectral_edges.append(freqs[edge_idx])
    
    # Concatenate features
    features.extend(absolute_powers)
    features.extend(relative_powers)
    features.extend(spectral_edges)
    features.extend(peak_freqs_alpha)
    features.extend(peak_powers_alpha)
    features.extend(peak_freqs_beta)
    features.extend(peak_powers_beta)
    
    features = np.array(features, dtype=np.float64)
    
    # Normalize features
    features = _normalize_features(features)
    
    # Ensure finite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def extract_temporal_statistical_features(data, sampling_freq=250):
    """
    Extract time-domain statistical and morphological features from EEG data.
    
    Time-domain features capture the amplitude dynamics and statistical properties
    of the neural signal, including non-linear complexity measures that reflect
    cognitive engagement and motor preparation states.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_timepoints)
        EEG data array
    sampling_freq : int, default=250
        Sampling frequency in Hz
    
    Returns
    -------
    features : ndarray, shape (n_features,)
        Concatenated temporal-statistical features normalized for GLM input
    """
    
    # Validate input
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    n_channels, n_timepoints = data.shape
    if n_timepoints < 10:
        raise ValueError(f"Window too short: {n_timepoints} timepoints (minimum 10)")
    
    features = []
    
    # Handle NaN values
    data_clean = np.copy(data)
    for ch in range(n_channels):
        if np.isnan(data_clean[ch]).any():
            data_clean[ch] = np.nanmean(data_clean[ch]) * np.ones_like(data_clean[ch])
    
    for ch in range(n_channels):
        signal_data = data_clean[ch]
        
        # Basic statistics
        features.append(np.mean(signal_data))
        features.append(np.std(signal_data))
        features.append(stats.skew(signal_data))
        features.append(stats.kurtosis(signal_data))
        
        # Hjorth parameters
        first_deriv = np.diff(signal_data)
        second_deriv = np.diff(first_deriv)
        
        activity = np.var(signal_data)
        mobility = np.sqrt(np.var(first_deriv) / activity) if activity > 1e-10 else 0
        complexity = (np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility 
                     if mobility > 1e-10 else 0)
        
        features.append(np.log10(activity + 1e-12))
        features.append(mobility)
        features.append(complexity)
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
        zcr = zero_crossings / (n_timepoints - 1)
        features.append(zcr)
        
        # Peak-to-peak amplitude
        ptp = np.ptp(signal_data)
        features.append(ptp)
        
        # Line length (sum of absolute consecutive differences)
        line_length = np.sum(np.abs(np.diff(signal_data)))
        features.append(np.log10(line_length + 1e-12))
        
        # Sample entropy (m=2, r=0.2*std)
        r = 0.2 * np.std(signal_data)
        if r < 1e-10:
            r = 1e-10
        sample_ent = _sample_entropy(signal_data, m=2, r=r)
        features.append(sample_ent)
        
        # Hurst exponent
        hurst_exp = _hurst_exponent(signal_data)
        features.append(hurst_exp)
        
        # Statistics of first derivative (temporal dynamics)
        features.append(np.mean(first_deriv))
        features.append(np.std(first_deriv))
        features.append(np.max(np.abs(first_deriv)))
    
    features = np.array(features, dtype=np.float64)
    
    # Normalize features
    features = _normalize_features(features)
    
    # Ensure finite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def extract_connectivity_features(data, sampling_freq=250):
    """
    Extract inter-channel relationship and spatial pattern features from EEG data.
    
    Connectivity features reflect functional integration and communication between
    neural populations, which are known to relate to cognitive performance and
    response preparation. Spatial patterns capture the distributed organization
    of neural activity across the scalp.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_timepoints)
        EEG data array
    sampling_freq : int, default=250
        Sampling frequency in Hz
    
    Returns
    -------
    features : ndarray, shape (n_features,)
        Concatenated connectivity features normalized for GLM input
    """
    
    # Validate input
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    n_channels, n_timepoints = data.shape
    if n_timepoints < 50:
        raise ValueError(f"Window too short for connectivity: {n_timepoints} timepoints")
    
    features = []
    
    # Handle NaN values
    data_clean = np.copy(data)
    for ch in range(n_channels):
        if np.isnan(data_clean[ch]).any():
            data_clean[ch] = np.nanmean(data_clean[ch]) * np.ones_like(data_clean[ch])
    
    # === Phase-based connectivity ===
    # Compute analytic signal via Hilbert transform for all channels
    analytic_signal = np.zeros_like(data_clean, dtype=complex)
    phase_data = np.zeros_like(data_clean)
    
    for ch in range(n_channels):
        analytic_signal[ch] = hilbert(data_clean[ch])
        phase_data[ch] = np.angle(analytic_signal[ch])
    
    # Phase-locking value (PLV) for alpha and beta bands
    bands_conn = {'alpha': (8, 13), 'beta': (13, 30)}
    plv_matrices = {}
    
    for band_name, (f_low, f_high) in bands_conn.items():
        # Filter in band using butterworth
        sos = signal.butter(4, [f_low, f_high], btype='band', fs=sampling_freq, output='sos')
        filtered = np.zeros_like(data_clean)
        
        for ch in range(n_channels):
            filtered[ch] = signal.sosfilt(sos, data_clean[ch])
        
        # Compute phase
        band_analytic = np.zeros_like(filtered, dtype=complex)
        band_phase = np.zeros_like(filtered)
        for ch in range(n_channels):
            band_analytic[ch] = hilbert(filtered[ch])
            band_phase[ch] = np.angle(band_analytic[ch])
        
        # PLV computation
        plv_mat = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = band_phase[i] - band_phase[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_mat[i, j] = plv
                plv_mat[j, i] = plv
        
        plv_matrices[band_name] = plv_mat
        
        # Extract upper triangle
        features.extend(plv_mat[np.triu_indices(n_channels, k=1)])
    
    # Mean phase coherence across channels
    mean_phase_coherence = np.mean([plv_matrices[band].mean() for band in bands_conn.keys()])
    features.append(mean_phase_coherence)
    
    # === Amplitude-based connectivity ===
    # Pearson correlation between all channel pairs
    corr_matrix = np.corrcoef(data_clean)
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations for feature extraction
    features.extend(corr_matrix[np.triu_indices(n_channels, k=1)])
    
    # Amplitude envelope correlation (using Hilbert transform)
    envelope = np.abs(analytic_signal)
    envelope_corr = np.corrcoef(envelope)
    np.fill_diagonal(envelope_corr, 0)
    features.extend(envelope_corr[np.triu_indices(n_channels, k=1)])
    
    # === Spatial features ===
    # Global field power (spatial standard deviation across channels at each timepoint)
    gfp = np.std(data_clean, axis=0)
    features.append(np.mean(gfp))
    features.append(np.std(gfp))
    features.append(np.max(gfp))
    
    # Spatial complexity via eigenvalue entropy of covariance matrix
    cov_matrix = np.cov(data_clean)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.abs(eigenvalues[eigenvalues > 1e-10])  # Remove numerical artifacts
    
    if len(eigenvalues) > 0:
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        spatial_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        features.append(spatial_entropy)
    else:
        features.append(0.0)
    
    # Common spatial pattern-inspired features (variance ratio)
    if n_channels > 1:
        # Ratio of max to min variance
        channel_vars = np.var(data_clean, axis=1)
        var_ratio = np.max(channel_vars) / (np.min(channel_vars) + 1e-10)
        features.append(np.log10(var_ratio + 1e-10))
    else:
        features.append(0.0)
    
    # === Graph metrics from correlation matrix ===
    # Threshold correlation matrix (absolute value > 0.3)
    threshold = 0.3
    adj_matrix = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj_matrix, 0)
    
    # Clustering coefficient
    clustering_coeff = _clustering_coefficient(adj_matrix)
    features.append(clustering_coeff)
    
    # Global efficiency
    global_eff = _global_efficiency(corr_matrix, threshold)
    features.append(global_eff)
    
    # Modularity (using spectral approach)
    modularity = _modularity(adj_matrix)
    features.append(modularity)
    
    features = np.array(features, dtype=np.float64)
    
    # Normalize features
    features = _normalize_features(features)
    
    # Ensure finite values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


# ============================================================================
# Helper functions
# ============================================================================

def _normalize_features(features):
    """
    Z-score normalize features for GLM input.
    
    Parameters
    ----------
    features : ndarray
        Feature vector
    
    Returns
    -------
    normalized : ndarray
        Z-score normalized features
    """
    mean_val = np.mean(features)
    std_val = np.std(features)
    
    if std_val < 1e-10:
        return np.zeros_like(features)
    
    return (features - mean_val) / (std_val + 1e-10)


def _sample_entropy(signal_data, m=2, r=0.2):
    """
    Compute sample entropy of a signal.
    
    Sample entropy measures the regularity and predictability of a time series.
    Lower values indicate more regular, predictable signals (e.g., during drowsiness),
    while higher values suggest more complex, irregular activity (engaged cognition).
    
    Parameters
    ----------
    signal_data : ndarray, shape (n_timepoints,)
        Input signal
    m : int, default=2
        Embedding dimension
    r : float, default=0.2
        Tolerance (typically 0.2 * std of signal)
    
    Returns
    -------
    sample_entropy : float
        Sample entropy value
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = [[signal_data[j] for j in range(i, i + m - 1 + 1)] for i in range(len(signal_data) - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (len(x) - 1.0) for x_i in x]
        return (len(x) - 1.0) ** (-1) * sum(np.log(C))
    
    try:
        return abs(_phi(m + 1) - _phi(m))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _hurst_exponent(signal_data, max_lag=None):
    """
    Compute Hurst exponent using rescaled range analysis.
    
    The Hurst exponent characterizes the long-range dependence and self-similarity
    of neural signals. Values > 0.5 indicate persistent behavior (trends),
    while values < 0.5 suggest mean-reverting behavior.
    
    Parameters
    ----------
    signal_data : ndarray
        Input signal
    max_lag : int, optional
        Maximum lag for analysis
    
    Returns
    -------
    hurst : float
        Hurst exponent (clamped to [0, 2])
    """
    n = len(signal_data)
    
    if n < 10:
        return 0.5
    
    if max_lag is None:
        max_lag = min(n // 3, 128)
    
    tau = []
    lags = []
    
    for k in range(10, max_lag):
        if k >= n:
            break
        
        # Mean-adjusted series
        mean_adj = signal_data[:k] - np.mean(signal_data[:k])
        
        # Cumulative sum
        y = np.cumsum(mean_adj)
        
        # R/S calculation
        r = np.max(y) - np.min(y)
        s = np.std(signal_data[:k], ddof=1)
        
        if s > 0:
            tau.append(r / s)
            lags.append(k)
    
    # Fit log-log curve
    if len(tau) < 2:
        return 0.5
    
    tau = np.array(tau)
    lags = np.array(lags)
    
    log_tau = np.log(tau)
    log_lags = np.log(lags)
    
    try:
        hurst = np.polyfit(log_lags, log_tau, 1)[0]
        return np.clip(hurst, 0, 2)
    except (ValueError, np.linalg.LinAlgError):
        return 0.5


def _clustering_coefficient(adj_matrix):
    """
    Compute average clustering coefficient of a graph.
    
    Clustering coefficient measures the degree to which channels tend to
    cluster together, reflecting local network density and integration.
    
    Parameters
    ----------
    adj_matrix : ndarray, shape (n_nodes, n_nodes)
        Binary adjacency matrix
    
    Returns
    -------
    clustering_coeff : float
        Average clustering coefficient [0, 1]
    """
    n = adj_matrix.shape[0]
    if n < 3:
        return 0.0
    
    clustering = []
    
    for i in range(n):
        neighbors = np.where(adj_matrix[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            clustering.append(0.0)
            continue
        
        # Count edges between neighbors
        edges = np.sum(adj_matrix[np.ix_(neighbors, neighbors)]) / 2
        possible_edges = k * (k - 1) / 2
        
        clustering.append(edges / possible_edges)
    
    return np.mean(clustering) if clustering else 0.0


def _global_efficiency(corr_matrix, threshold=0.3):
    """
    Compute global efficiency of a weighted network.
    
    Global efficiency reflects how efficiently information is exchanged
    across the entire network, indicating global integration.
    
    Parameters
    ----------
    corr_matrix : ndarray, shape (n_nodes, n_nodes)
        Correlation matrix (symmetric)
    threshold : float, default=0.3
        Threshold for converting to binary adjacency
    
    Returns
    -------
    global_efficiency : float
        Global efficiency measure
    """
    n = corr_matrix.shape[0]
    
    if n < 2:
        return 0.0
    
    # Create weighted adjacency from correlation
    adj = np.abs(corr_matrix) * (np.abs(corr_matrix) > threshold)
    np.fill_diagonal(adj, 0)
    
    # Compute shortest path lengths via distance
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = 1.0 / (adj + 1e-10)
        distance[adj == 0] = np.inf
    
    # Compute global efficiency
    efficiencies = []
    for i in range(n):
        row = distance[i].copy()
        row[np.isinf(row)] = 0
        row[i] = 0
        if np.sum(row) > 0:
            efficiencies.append(np.sum(1.0 / (row + 1e-10)))
    
    if not efficiencies:
        return 0.0
    
    return np.mean(efficiencies) / (n - 1)


def _modularity(adj_matrix):
    """
    Compute modularity of a network using spectral approach.
    
    Modularity measures the extent to which a network can be partitioned into
    modules with dense internal connections and sparse external connections.
    
    Parameters
    ----------
    adj_matrix : ndarray, shape (n_nodes, n_nodes)
        Binary adjacency matrix
    
    Returns
    -------
    modularity : float
        Modularity measure (normalized to [0, 1])
    """
    n = adj_matrix.shape[0]
    
    if n < 3:
        return 0.0
    
    # Ensure symmetric
    adj_sym = (adj_matrix + adj_matrix.T) / 2
    
    # Degree
    degree = np.sum(adj_sym, axis=1)
    m = np.sum(adj_sym) / 2
    
    if m == 0:
        return 0.0
    
    # Modularity matrix
    mod_matrix = adj_sym - np.outer(degree, degree) / (2 * m)
    
    # Eigenvalue decomposition
    try:
        eigenvalues, _ = np.linalg.eigh(mod_matrix)
        max_eigenvalue = np.max(eigenvalues)
        modularity = max(0, max_eigenvalue / (2 * m))
        return np.clip(modularity, 0, 1)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


if __name__ == '__main__':
    """
    Example usage and testing of feature extraction functions.
    """
    
    # Generate synthetic EEG data for testing
    np.random.seed(42)
    n_channels = 8
    duration_sec = 5
    sampling_freq = 250
    n_timepoints = duration_sec * sampling_freq
    
    # Create synthetic multi-component EEG signal
    t = np.arange(n_timepoints) / sampling_freq
    eeg_data = np.zeros((n_channels, n_timepoints))
    
    for ch in range(n_channels):
        # Alpha oscillation (10 Hz)
        alpha = 20 * np.sin(2 * np.pi * 10 * t)
        # Beta oscillation (20 Hz)
        beta = 10 * np.sin(2 * np.pi * 20 * t)
        # Noise
        noise = 5 * np.random.randn(n_timepoints)
        
        eeg_data[ch] = alpha + beta + noise
    
    print("=" * 70)
    print("EEG FEATURE EXTRACTION - EXAMPLE USAGE")
    print("=" * 70)
    print(f"\nInput data shape: {eeg_data.shape} ({n_channels} channels, {n_timepoints} timepoints)")
    print(f"Sampling frequency: {sampling_freq} Hz\n")
    
    # Extract spectral features
    print("Extracting spectral features...")
    spectral_features = extract_spectral_features(eeg_data, sampling_freq=sampling_freq)
    print(f"  ✓ Spectral features shape: {spectral_features.shape}")
    print(f"    Mean: {np.mean(spectral_features):.4f}, Std: {np.std(spectral_features):.4f}")
    
    # Extract temporal-statistical features
    print("\nExtracting temporal-statistical features...")
    temporal_features = extract_temporal_statistical_features(eeg_data, sampling_freq=sampling_freq)
    print(f"  ✓ Temporal features shape: {temporal_features.shape}")
    print(f"    Mean: {np.mean(temporal_features):.4f}, Std: {np.std(temporal_features):.4f}")
    
    # Extract connectivity features
    print("\nExtracting connectivity features...")
    connectivity_features = extract_connectivity_features(eeg_data, sampling_freq=sampling_freq)
    print(f"  ✓ Connectivity features shape: {connectivity_features.shape}")
    print(f"    Mean: {np.mean(connectivity_features):.4f}, Std: {np.std(connectivity_features):.4f}")
    
    # Combine all features
    all_features = np.concatenate([spectral_features, temporal_features, connectivity_features])
    print(f"\n{'=' * 70}")
    print(f"Combined feature vector shape: {all_features.shape}")
    print(f"Total features for GLM: {len(all_features)}")
    print(f"{'=' * 70}")
    
    # Verify all features are finite
    assert np.all(np.isfinite(all_features)), "Non-finite values detected in feature vector"
    print("\n✓ All features are finite and ready for GLM input")