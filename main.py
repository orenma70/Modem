
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_welch, lte_rx_symbol, lte_tx_symbol, rx_dfe, tx_dfe, add_awgn
import config



snr_db = 6


#decimation_fir = [-43 -86 36 106 -145 -141 625 1179]/2048;




total_errors = 0
all_rx_samples = []
all_rx_sig_samples = []

last_freq_data = None # Start as empty

print(f"LTE {config.bw}MHz | FFT {config.n_fft} | SNR {snr_db}dB")

current_pos = 0

all_tx_sig = [[] for _ in range(config.Nc)]
all_tx_bits = [[] for _ in range(config.Nc)]

for f_idx in range(config.Nc):
    for s_idx in range(7):
        # Generate the signal (e.g., length 2192 * resample_factor)
        tx_sig, tx_bits = lte_tx_symbol(s_idx)

        # Store in the row corresponding to the current carrier
        all_tx_sig[f_idx].extend(tx_sig)
        all_tx_bits[f_idx].append(tx_bits)

all_tx_sig = tx_dfe(np.array(all_tx_sig))
# Now apply the channel/noise to the WHOLE stream
rx_sig_total = add_awgn(all_tx_sig, snr_db)
rx_sig = rx_dfe(rx_sig_total)


delay_offset = 0

# Initialize error counter and sample storage per channel
total_errors = np.zeros(config.Nc)
all_rx_samples = [[] for _ in range(config.Nc)]

# Outer Loop: Process each of the 8 channels
for f_idx in range(config.Nc):
    current_pos = 0
    # Pick the 1D signal for the current carrier
    rx_carrier_sig = rx_sig[f_idx]

    # Inner Loop: Process the 7 symbols for this specific carrier
    for s_idx in range(7):
        # CP length logic (80 for s_idx=0, 72 otherwise for 120kHz)
        cp_len = config.cp_first if s_idx == 0 else config.cp_normal
        symbol_total_length = cp_len + config.n_fft

        # Windowing: Extract the segment from the stream
        start_idx = current_pos + delay_offset
        end_idx = start_idx + symbol_total_length

        # Guard against index out of bounds
        if end_idx > len(rx_carrier_sig):
            break

        symbol_segment = rx_carrier_sig[start_idx:end_idx]

        # Standard LTE/5G RX processing
        rx_bits, extracted, freq = lte_rx_symbol(symbol_segment, s_idx)

        # Update error count for THIS specific carrier
        # Comparing against the 2D bit array: [carrier][symbol]
        total_errors[f_idx] += np.sum(all_tx_bits[f_idx][s_idx] != rx_bits)

        # Save extracted QAM samples for constellation plots
        all_rx_samples[f_idx].extend(extracted)

        current_pos += symbol_total_length

# Reporting
for i in range(config.Nc):
    print(f"Channel {i} Errors: {total_errors[i]}")


print("-" * 30)
print(f"Final BER: {total_errors / (7 * config.num_sc * 2)}")
print("-" * 30)





#plot_welch(rx_sig, fs = fs, color='blue', label='PSD', fig_flag='hold')

'''
# --- Calculate Filter Response (Black) ---
# We map the digital frequency (0 to 1) to the 2Fs frequency scale
w, h_resp = signal.freqz(tx_fir1, worN=1024, whole=True)
f_filter = w / (2 * np.pi) * fs_d2a2d
f_filter = np.where(f_filter >= fs_d2a2d/2, f_filter - fs_d2a2d, f_filter) # Wrap frequencies
# Re-sort for plotting
sort_idx = np.argsort(f_filter)
f_filter = f_filter[sort_idx]
mag_filter = 20 * np.log10(np.abs(h_resp[sort_idx]) + 1e-12)

# --- Plotting with "Hold" (Overlaid) ---


# 2. PSD of 1Fs (Blue)
plt.plot(f_1fs / 1e6, pxx_1fs - np.max(pxx_1fs), color='blue', label='PSD @ 1Fs (Original)', linewidth=2)
'''

