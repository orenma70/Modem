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

bw_options = [1.4, 3, 5, 10, 15, 20]
rb_array    = [6, 15, 25, 50, 75, 100]
fft_array   = [128, 256, 512, 1024, 1536, 2048]
cp_first_array  = [10, 20, 40, 80, 120, 160]
cp_normal_array = [9, 18, 36, 72, 108, 144]
fs_array = [n * 15000 for n in fft_array]

idx = 5 # 10 MHz
bw = bw_options[idx]
fs = fs_array[idx] # Current sampling frequency
n_rb = rb_array[idx]
n_fft = fft_array[idx]
cp0 = cp_first_array[idx]
cp_other = cp_normal_array[idx]
num_sc = n_rb * 12
resample_factor = 16
fs_d2a2d = resample_factor * fs
tx_fir1 = get_23_resampling_taps()
tx_fir2 = np.array([3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3])/256
tx_fir3 = np.array([-1, 0,  9, 16, 9,  0, -1 ])/16
np.random.seed(42)  # Use any constant integer
