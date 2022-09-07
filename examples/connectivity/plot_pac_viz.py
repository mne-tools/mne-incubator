"""
==============================================================
Visualize phase-amplitude coupling measures between signals
==============================================================
Computes the normalized amplitude traces for a cross frequency coupled
signal across a given range of frequencies and displays it along with
the event related average response.
References
----------
[1] Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
    Berger MS, Barbaro NM, Knight RT. "High gamma power is phase-locked to
    theta oscillations in human neocortex." Science. 2006.
[2] Tort ABL, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude
    coupling between neuronal oscillations of different frequencies. Journal of
    Neurophysiology. 2010.
"""
# Author: Chris Holdgraf <choldgraf@berkeley.edu>
#         Praveen Sripad <praveen.sripad@rwth-aachen.de>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.datasets import sample
from mne_incubator.connectivity import (phase_amplitude_coupling,
                                        plot_phase_locked_amplitude,
                                        plot_phase_binned_amplitude)
import matplotlib.pyplot as plt

print(__doc__)

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
events = mne.read_events(event_fname)
ev_ixs = events[:, 0].astype(int)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG gradiometers
raw = raw.pick_types(meg='grad', eeg=False, stim=False, eog=True,
                     exclude='bads')

# Define a pair of indices
ixs = [(4, 10)]
ix_ph, ix_amp = ixs[0]

# First we can simply calculate a PAC statistic for these signals
f_range_phase = (6, 8)
f_range_amp = (40, 60)

# Create some artifical PAC data to show the effect
# Calculate the low-frequency phase
raw_phase = raw.copy()
raw_phase = raw_phase.filter(*f_range_phase)
raw_phase.apply_hilbert([ix_ph])
angles = np.angle(raw_phase._data[1])
msk_angles = angles > (.5 * np.pi)

# Take the high-frequency component of the signal, and modulate it w/ the phase
raw_band = raw.copy()
raw_band = raw_band.filter(*f_range_amp)
raw_band._data[ix_amp][~msk_angles] = 0
raw_band._data[ix_amp][msk_angles] *= 10.

# Now add the high-freq signal back into the raw data
raw_artificial = raw.copy()
raw_artificial._data[ix_amp] += raw_band._data[ix_amp]

for i_data in [raw, raw_artificial]:
    pac, freqs_pac = phase_amplitude_coupling(
        i_data, f_range_phase, f_range_amp, ixs, pac_func='glm',
        events=ev_ixs, tmin=0, tmax=.5)
    pac = pac.mean()  # Average across events

    # We can also visualize these relationships
    # Create epochs for left-visual condition
    event_id, tmin, tmax = 3, -1, 4
    epochs = mne.Epochs(i_data, events, event_id, tmin, tmax,
                        baseline=(None, 0.),
                        reject=dict(grad=4000e-13, eog=150e-6), preload=True)
    ph_range = np.linspace(*f_range_phase, num=6)
    amp_range = np.linspace(*f_range_amp, num=20)

    # Show the amp for a range of frequencies, phase-locked to a low-freq
    ax = plot_phase_locked_amplitude(epochs, ph_range, amp_range, ixs[0][0],
                                     ixs[0][1], normalize=True)
    ax.set_title('Phase Locked Amplitude, PAC = {0}'.format(pac))

    # Show the avg amp of the high freqs for bins of phase in the low freq
    ax = plot_phase_binned_amplitude(epochs, ph_range, amp_range,
                                     ixs[0][0], ixs[0][1], normalize=True,
                                     n_bins=20)
    ax.set_title('Phase Binned Amplitude, PAC = {0}'.format(pac))

plt.tight_layout()
plt.show(block=True)
