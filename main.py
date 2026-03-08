
import numpy as np
import matplotlib.pyplot as plt
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
h_resample = get_23_resampling_taps()

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
snr_db = 40

resample_factor = 2
tx_fir1 = get_23_resampling_taps()

def lte_tx_symbol(symbol_idx):
    cp_len = cp0 if (symbol_idx % 7 == 0) else cp_other
    bits = np.random.randint(0, 2, num_sc * 2)
    syms = ((1 - 2*bits[0::2]) + 1j*(1 - 2*bits[1::2])) / np.sqrt(2)
    buffer = np.zeros(n_fft, dtype=complex)
    start = (n_fft - num_sc) // 2
    buffer[start : start + num_sc] = syms
    time_sig = np.fft.ifft(np.fft.ifftshift(buffer))
    tx_out = np.concatenate([time_sig[-cp_len:], time_sig])
    if resample_factor >1:

        tx_upsampled = np.zeros(len(tx_out) * 2, dtype=complex)
        tx_upsampled[::2] = tx_out

        # Filter to smooth out the zeros (Anti-Imaging)
        tx_out = np.convolve(tx_upsampled, tx_fir1, mode='same')

    return tx_out, bits

def add_awgn(signal, snr):
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr/10))
    noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal)) +
    1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal)))
    return signal + noise

def lte_rx_symbol(rx_in, symbol_idx):
    if resample_factor > 1:
        rx_in = np.convolve(rx_in, tx_fir1 / 2, mode='same')
        rx_in = rx_in[::2]


    cp_len = cp0 if (symbol_idx % 7 == 0) else cp_other
    no_cp = rx_in[cp_len:]
    freq = np.fft.fftshift(np.fft.fft(no_cp))
    start = (n_fft - num_sc) // 2
    extracted = freq[start : start + num_sc]
    rx_bits = np.zeros(num_sc * 2, dtype=int)
    rx_bits[0::2] = (extracted.real < 0).astype(int)
    rx_bits[1::2] = (extracted.imag < 0).astype(int)
    return rx_bits, extracted, freq

total_errors = 0
all_rx_samples = []
all_rx_sig_samples = []

last_freq_data = None # Start as empty

print(f"LTE {bw}MHz | FFT {n_fft} | SNR {snr_db}dB")

for s_idx in range(7):
    tx_sig, tx_bits = lte_tx_symbol(s_idx)
    rx_sig = add_awgn(tx_sig, snr_db)
    rx_bits, rx_samples, last_freq_data = lte_rx_symbol(rx_sig, s_idx)
    all_rx_samples.extend(rx_samples)
    all_rx_sig_samples.extend(rx_sig)


print("-" * 30)
print(f"Final BER: {total_errors / (7 * num_sc * 2)}")

plt.figure(figsize=(12, 5))

#Plot 1: Constellation
plt.subplot(2, 2, 1)
plt.scatter(np.real(all_rx_samples), np.imag(all_rx_samples), s=1, color='blue', alpha=0.5)
plt.title(f"QPSK Constellation @ {snr_db}dB")
plt.grid(True); plt.axhline(0, color='black'); plt.axvline(0, color='black')

plt.subplot(2, 2, 3)
psd = 10 * np.log10(np.abs(rx_sig)**2 + 1e-12)
freq_axis = np.linspace(-fs/2, fs/2, n_fft)
plt.plot(freq_axis / 1e6, psd) # Scale to MHz
plt.title(f"Centered PSD ({bw} MHz BW)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.grid(True)



#Plot 2: Centered PSD
plt.subplot(2, 2, 4)
psd = 10 * np.log10(np.abs(last_freq_data)**2 + 1e-12)
freq_axis = np.linspace(-fs/2, fs/2, n_fft)
plt.plot(freq_axis / 1e6, psd) # Scale to MHz
plt.title(f"Centered PSD ({bw} MHz BW)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()