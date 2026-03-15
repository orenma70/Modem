

import numpy as np
import config
from ftc_fir import farrow_resample
from utils2 import plot_welch

resample_factor = config.resample_factor
cp0 = config.cp_first
n_fft = config.n_fft
tx_fir3 = config.tx_fir3
num_sc = config.num_sc
tx_fir2 = config.tx_fir2
tx_fir1 = config.tx_fir1
cp_other = config.cp_normal
fs = config.fs
fs_d2a2d = config.fs_d2a2d
f_nco = config.f_nco

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


