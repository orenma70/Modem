import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

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