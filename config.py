from numba.np.arrayobj import np_array
from scipy import signal
import numpy as np

def get_23_resampling_taps(N = 23):
    # 1. Calculate Kaiser beta for 40dB attenuation (approx 3.3953)
    beta = signal.kaiser_beta(40)
    # 2. Design 23 taps for factor-of-2 (cutoff at 0.5 Nyquist)
    # 3. Multiply by 2 to compensate for the 6dB loss from zero-stuffing
    h = signal.firwin(N, cutoff=0.5, window=('kaiser', beta)) * 2
    return h

# Usage
h_resample = get_23_resampling_taps(31)

mmw = 1

if mmw:
    Nc = 8
    scs = 120000
    fc1 = 417 * scs
    Nf = np.arange(-(Nc - 1), Nc, 2)*fc1
    # 5G NR FR2 (mmWave) - Standardized for 120 kHz SCS

    bw_options = [50, 100, 200, 400]
    # Max RBs for 120kHz SCS
    rb_array    = [32, 66, 132, 264]
    # FFT sizes for these bandwidths
    fft_array   = [512, 1024, 2048, 4096]
    # CP lengths scale inversely with SCS (shorter time duration)
    # For 120kHz, Normal CP is 36 samples for a 512 FFT
    cp_normal_array = [36, 72, 144, 288]
    cp_first_array  = [44, 80, 160, 320] # Slightly longer first symbol
    idx = 1  # 10 MHz
else:
    # Standard LTE - 15 kHz SCS
    scs = 15000
    bw_options = [1.4, 3, 5, 10, 15, 20]
    rb_array    = [6, 15, 25, 50, 75, 100]
    fft_array   = [128, 256, 512, 1024, 1536, 2048]
    cp_normal_array = [9, 18, 36, 72, 108, 144]
    cp_first_array  = [10, 20, 40, 80, 120, 160]
    idx = 5 # 10 MHz
# The Fix: Use the 'scs' variable so it works for both modes
fs_array = [n * scs for n in fft_array]


bw = bw_options[idx]
fs = fs_array[idx] # Current sampling frequency
n_rb = rb_array[idx]
n_fft = fft_array[idx]
cp_first = cp_first_array[idx]
cp_normal = cp_normal_array[idx]
num_sc = n_rb * 12
resample_factor = 16
fs_d2a2d = resample_factor * fs
tx_fir1 = get_23_resampling_taps()
tx_fir2 = np.array([3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3])/256
tx_fir3 = np.array([-1, 0,  9, 16, 9,  0, -1 ])/16
np.random.seed(42)  # Use any constant integer
