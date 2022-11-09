import numpy as np


def simulate_pac_signal(time, freq_phase, freq_amp, max_amp_lo=2.,
                        max_amp_hi=.5, frac_pac=.1, snr_lo=4.,
                        snr_hi=4., mask_pac_times=None):
    """Simulate a signal with phase-amplitude coupling according to [1].

    Parameters
    ----------
    time : array, shape (n_times,)
        The times for the signal (which implicitly defines the sampling
        frequency).
    freq_phase : float
        The frequency of the low-frequency phase that modulates amplitude.
    freq_amp : float
        The frequency of the high-frequency amplitude that is modulated by
        phase.
    max_amp_lo : float
        The maximum amplitude for the low-frequency phase signal.
    max_amp_hi : float
        The maximum amplitude for the high-frequency amplitude signal.
    frac_pac : float, between (0., 1.)
        The fraction of the high-frequency amplitude that is modulated by
        low-frequency phase.
    snr_lo : float | None
        The ratio of signal to noise in the low-frequency signal.
        Defaults to 4.
    snr_hi : float | None
        The ratio of signal to noise in the high-frequency signal.
        Defaults to 4.
    mask_pac_times : array, dtype bool, shape (n_times,) | None
        Whether to mask specific times to induce PAC. Values where
        `mask_pac_times` is False will have `frac_pac` set to 0.
        If None, all times have PAC.

    Returns
    -------
    signal : array, shape (n_times,)
        The simulated PAC signal w/ both low and high frequency components.
    phase_signal : array, shape (n_times,)
        The low-frequency phase.
    amp_signal : array, shape (n_times,)
        The high-frequency amplitude.

    References
    ----------
    .. [1] Tort, et al. "Measuring Phase-Amplitude Coupling Between Neuronal
           Oscillations of Different Frequencies." Journal of Neurophysiology,
           vol. 104, issue 2, 2010.
    """
    if frac_pac < 0. or frac_pac > 1.:
        raise ValueError('frac_pac must be between 0. and 1.')
    if not all([isinstance(i, (float, int)) for i in [freq_amp, freq_phase]]):
        raise ValueError('freq_amp and freq_phase must be a single float')
    if mask_pac_times is None:
        mask_pac_times = np.ones_like(time).astype(bool)

    # Simulate noise
    noise = np.random.randn(2, time.shape[0])
    noise = noise * np.array([max_amp_lo / snr_lo,
                              max_amp_hi / snr_hi])[:, np.newaxis]

    # Low-freq phase
    phase_signal = max_amp_lo * np.sin(2 * np.pi * freq_phase * time)
    phase_signal = phase_signal + noise[0]

    # High-freq amp
    frac_non_pac = 1. - frac_pac
    frac_non_pac = np.where(mask_pac_times, frac_non_pac, 1)
    amp_signal = (1 - frac_non_pac) * np.sin(2 * np.pi * freq_phase * time)
    amp_signal += 1 + frac_non_pac
    amp_signal = max_amp_hi * amp_signal / 2.
    amp_signal = amp_signal * np.sin(2 * np.pi * freq_amp * time)
    amp_signal = amp_signal + noise[1]

    # Combine them
    signal = amp_signal + phase_signal

    return signal, phase_signal, amp_signal
