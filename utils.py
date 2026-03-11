import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import config


resample_factor = config.resample_factor
cp0 = config.cp0
n_fft = config.n_fft
tx_fir3 = config.tx_fir3
num_sc = config.num_sc
tx_fir2 = config.tx_fir2
tx_fir1 = config.tx_fir1
cp_other = config.cp_other
fs = config.fs
fs_d2a2d = config.fs_d2a2d

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

def rx_dfe(rx_in):
    rx_out = rx_in
    if config.resample_factor >= 16:
        rx_out = decimation2(rx_in, tx_fir3)
    else:
        rx_out = rx_in
    if resample_factor >= 8:
        rx_out = decimation2(rx_out, tx_fir3)
    if resample_factor >= 4:
        rx_out = decimation2(rx_out, tx_fir2)
    if resample_factor >= 2:
        rx_out = decimation2(rx_out, tx_fir1)

    return rx_out

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
    plot_welch(tx_in, fs=fs, color='blue', label='PSD', fig_flag='new')
    tx_out = tx_in
    if resample_factor >= 2:
        tx_out =interpolation2(tx_in,tx_fir1)
    if resample_factor >= 4:
        tx_out = interpolation2(tx_out, tx_fir2)
    if resample_factor >= 8:
        tx_out = interpolation2(tx_out, tx_fir3)
    if resample_factor >= 16:
        tx_out = interpolation2(tx_out, tx_fir3)

    return tx_out



def add_awgn(signal, snr):
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10**(snr/10))
    noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal)) +
    1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal)))
    return signal + noise

def interpolation2(tx_in,fir_taps):
    tx_upsampled = np.zeros(len(tx_in) * 2, dtype=complex)
    tx_upsampled[::2] = tx_in
    # plot_welch(tx_upsampled, fs=fs_d2a2d, color='blue', label='PSD', fig_flag='new')

    # Filter to smooth out the zeros (Anti-Imaging)
    target_len = len(tx_upsampled)
    tx_full = np.convolve(tx_upsampled, fir_taps, mode='full')
    fir_delay = (len(fir_taps) -1) // 2
    tx_out = tx_full[fir_delay: fir_delay + target_len]
    return tx_out

def decimation2(rx_in, fir_taps):
    rx_out =[]
    rx_out = np.convolve(rx_in, fir_taps / 2, mode='full')
    fir_delay = (len(fir_taps) - 1) // 2
    rx_out = rx_out[fir_delay: fir_delay + len(rx_in)]
    return  rx_out[::2]


def plot_welch(t_sig, fs, color='red', label='PSD', fig_flag='hold'):
    # 1. Create a new figure if requested
    if fig_flag == 'new':
        plt.figure(figsize=(12, 6))

    # 2. Calculate PSD
    f, pxx = signal.welch(t_sig, fs=fs, nperseg=1024*2, window='blackman', return_onesided=False)

    # 3. Shift and Convert to dB
    f = np.fft.fftshift(f)
    pxx_db = 10 * np.log10(np.fft.fftshift(pxx) + 1e-12)



    # 4. Plot (Normalized to 0dB peak for comparison)
    plt.plot(f / 1e6, pxx_db - np.max(pxx_db), color=color, label=label, alpha=0.7)

    # Standard formatting that applies to both new and hold
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim([-80, 10])  # Adjust to see the stopband clearly
    plt.xlim([-fs / (2 * 1e6), fs / (2 * 1e6)])
    plt.show()