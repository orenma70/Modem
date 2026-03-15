import numpy as np
import config
from ftc_fir import farrow_resample
from utils import interpolation2

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

def tx_dfe(tx_in):
    # tx_in is a 2D array: shape (8, samples)
    nc = tx_in.shape[0]
    Nco_fs = fs * resample_factor
    all_interpolated = []

    for i in range(nc):
        # Process each carrier individually using your existing logic
        row = tx_in[i]

        if resample_factor >= 2:
            row = interpolation2(row, tx_fir1)
        if resample_factor >= 4:
            row = interpolation2(row, tx_fir2)
        if resample_factor >= 8:
            row = interpolation2(row, tx_fir3)
        if resample_factor >= 16:
            row = interpolation2(row, tx_fir3)

        n = np.arange(len(row))
        nco = np.exp(1j * 2 * np.pi * config.Nf[i]/Nco_fs * n)
        row = row * nco

        all_interpolated.append(row)

    combined_signal = np.sum(all_interpolated, axis=0)

    combined_signal = combined_signal / nc

    ftc_out = farrow_resample(combined_signal, f_nco, fs_d2a2d)

    return np.array(ftc_out)
