
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_welch, lte_rx_symbol, lte_tx_symbol, rx_dfe, tx_dfe, add_awgn
import config



snr_db = -12


#decimation_fir = [-43 -86 36 106 -145 -141 625 1179]/2048;




total_errors = 0
all_rx_samples = []
all_rx_sig_samples = []

last_freq_data = None # Start as empty

print(f"LTE {config.bw}MHz | FFT {config.n_fft} | SNR {snr_db}dB")

current_pos = 0


all_tx_sig = []
all_tx_bits = []

for s_idx in range(7):
    # This should return the UPSAMPLED (interpolated) signal
    tx_sig, tx_bits = lte_tx_symbol(s_idx)
    all_tx_sig.extend(tx_sig)
    all_tx_bits.append(tx_bits)


all_tx_sig = tx_dfe(np.array(all_tx_sig))
# Now apply the channel/noise to the WHOLE stream
rx_sig_total = add_awgn(np.array(all_tx_sig), snr_db)
rx_sig = rx_dfe(rx_sig_total)


delay_offset = 0


for s_idx in range(7):
    cp_len = config.cp0 if (s_idx % 7 == 0) else config.cp_other
    symbol_total_length = cp_len + config.n_fft

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

