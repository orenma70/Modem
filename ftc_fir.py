import numpy as np
import config
from utils2 import plot_welch



def farrow_resample(ftc_in, fs_in, fs_out):
    # 1. Setup dimensions and ratio
    L, N_poly = config.farrow_taps.shape  # L=7, N_poly=5
    ratio = fs_in / fs_out
    pad_len = (L - 1) // 2
    half_fir = (L - 1) // 2
    # 2. Generate the floating-point clock grid for the OUTPUT
    # We go from 0 up to the end of the input signal
    N_in = len(ftc_in)
    m_float = np.arange(0, N_in - 1, ratio)

    # 3. Find n_int (nearest neighbor) and mu (fractional offset)
    n_int = np.floor(m_float + 0.5).astype(int)
    mu_array = m_float - n_int  # Range: [-0.5, 0.5]

    # 4. Padding the input (Use 'reflect' for [4 3 2 1 2 3 4...] style)
    in_padded = np.pad(ftc_in, (pad_len, pad_len), mode='reflect')

    # 5. Prepare the Polynomial matrix (Powers of mu)
    # Shape: (len(m_float), N_poly)
    mu_powers = np.column_stack([mu_array ** i for i in range(N_poly)])

    # 6. Generate the filter for every single output sample
    # (Samples, 5) dot (5, 7) -> (Samples, 7)
    all_filters = np.dot(mu_powers, config.farrow_taps.T) / 256.0

    # 7. Extract windows from the padded signal
    # Shift n_int by pad_len because the signal was padded at the start
    n_shifted = n_int + pad_len

    # Efficient window extraction
    # We take L samples centered around each n_shifted
    windows = np.array([in_padded[idx - half_fir: idx + half_fir + 1 ][::-1] for idx in n_shifted])

    # 8. Final Dot Product
    # (Samples, 7) * (Samples, 7) summed across rows -> (Samples,)
    y = np.sum(windows * all_filters, axis=1)

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import signal



    # Assuming the functions are in the same file or imported

    fs_in = 100e6
    fs_out = 150e6
    f_sine = 10e6
    N = 100

    # Generate Input
    t_in = np.arange(N) / fs_in
    sin_in = np.exp(1j * 2 * np.pi * f_sine * t_in)

    #plot_welch(sin_in, fs_in, color='red', label='PSD', fig_flag='new')

    # Resample Forward
    ftc_out = farrow_resample(sin_in, fs_in, fs_out)

    # --- Time Domain Plotting ---
    plt.figure(figsize=(12, 6))

    # 1. Create time axes for each signal
    t_in_plot = np.arange(len(sin_in)) / fs_in
    t_out_plot = np.arange(len(ftc_out)) / fs_out

    # 2. Plot a small window (e.g., first 100 samples) to see the waves clearly
    # Using .real since the signal is complex (np.exp(1j...))
    num_samples_to_show = 200

    plt.plot(t_in_plot[:num_samples_to_show] * 1e9, sin_in[:num_samples_to_show].real,
             'r-o', label='Input (1966.08 MHz)', markersize=4)
    plt.plot(t_out_plot[:int(num_samples_to_show * fs_out / fs_in)]* 1e9,
             ftc_out[:int(num_samples_to_show * fs_out / fs_in)].real,
             'g-x', label='Farrow Output (1996.8 MHz)', markersize=4)

    plt.title(f'Time Domain Comparison: {f_sine / 1e6} MHz Sine Wave')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


    plot_welch(ftc_out, fs_in, color='green', label='PSD', fig_flag='hold')
    # Resample Backward
    sin_recon = farrow_resample(ftc_out, fs_out, fs_in)

    plot_welch(sin_recon, fs_in, color='blue', label='PSD', fig_flag='hold')


