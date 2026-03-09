
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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

fs_d2a2d = resample_factor * fs

tx_fir1 = get_23_resampling_taps()
np.random.seed(42)  # Use any constant integer

def lte_tx_symbol(symbol_idx):
    cp_len = cp0 if (symbol_idx % 7 == 0) else cp_other
    bits = np.random.randint(0, 2, num_sc * 2)
    syms = ((1 - 2*bits[0::2]) + 1j*(1 - 2*bits[1::2])) / np.sqrt(2)
    buffer = np.zeros(n_fft, dtype=complex)
    start = (n_fft - num_sc) // 2
    buffer[start : start + num_sc] = syms
    time_sig = np.fft.ifft(np.fft.ifftshift(buffer))
    tx_out = np.concatenate([time_sig[-cp_len:], time_sig])


    return tx_out, bits

def tx_dfe(tx_in):

    if resample_factor >1:

        tx_upsampled = np.zeros(len(tx_in) * 2, dtype=complex)
        tx_upsampled[::2] = tx_in

        # Filter to smooth out the zeros (Anti-Imaging)
        target_len = len(tx_upsampled)
        tx_full = np.convolve(tx_upsampled, tx_fir1, mode='full')
        tx_out = tx_full[11: 11 + target_len]

        return tx_out


def rx_dfe(rx_in):
    if resample_factor > 1:
        rx_full = np.convolve(rx_in, tx_fir1 / 2, mode='full')
        rx_aligned = rx_full[11: 11 + len(rx_in)]
        return  rx_aligned[::2]


def add_awgn(signal, snr):
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr/10))
    noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal)) +
    1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal)))
    return signal + noise


def lte_rx_symbol(rx_in, symbol_idx):
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

current_pos = 0


all_tx_sig = []
all_tx_bits = []

for s_idx in range(7):
    # This should return the UPSAMPLED (interpolated) signal
    tx_sig, tx_bits = lte_tx_symbol(s_idx)
    all_tx_sig.extend(tx_sig)
    all_tx_bits.append(tx_bits)


all_tx_sig = tx_dfe(all_tx_sig)
# Now apply the channel/noise to the WHOLE stream
rx_sig_total = add_awgn(np.array(all_tx_sig), snr_db)
rx_sig = rx_dfe(rx_sig_total)

if resample_factor == 1:
    delay_offset = 0
else:
    delay_offset = (len(tx_fir1) - 1) // (2)

for s_idx in range(7):
    cp_len = cp0 if (s_idx % 7 == 0) else cp_other
    symbol_total_length = cp_len + n_fft

    # Extract the specific segment for this symbol from the total stream
    # Apply the delay_offset to keep the FFT window aligned
    start_idx = current_pos #+ delay_offset
    end_idx = start_idx + symbol_total_length
    symbol_segment = rx_sig[start_idx: end_idx]

    # Process this segment
    rx_bits, extracted, freq = lte_rx_symbol(symbol_segment, s_idx)
    total_errors += np.sum(all_tx_bits[s_idx] != rx_bits)
    all_rx_samples.extend(extracted)
    current_pos += symbol_total_length







# Verify the spectrum of the continuous stream
# Use return_onesided=False for complex IQ data
f, pxx = signal.welch(rx_sig_total, fs=fs_d2a2d, nperseg=1024, return_onesided=False)

f = np.fft.fftshift(f)
pxx = np.fft.fftshift(pxx)

# Normalize to the average power in the center (passband)
# instead of just the absolute max to get a cleaner 0dB line
passband_indices = np.where(np.abs(f) < (bw * 1e6 / 2))
normalization_factor = np.mean(pxx[passband_indices])
pxx_normalized = pxx / normalization_factor

plt.plot(f / 1e6, 10 * np.log10(pxx_normalized))
plt.title("Normalized Spectrum (0 dB Passband)")
plt.ylim([-60, 5])
plt.grid(True)
plt.show()
print("-" * 30)
print(f"Final BER: {total_errors / (7 * num_sc * 2)}")
print("-" * 30)
'''



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
'''

