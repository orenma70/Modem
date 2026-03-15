import numpy as np
import config
from ftc_fir import farrow_resample
from utils import decimation2

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

def rx_dfe(rx_in):
    all_rx_carriers = []
    nc = config.Nc
    Nco_fs = fs * resample_factor
    ftc_out = farrow_resample(rx_in, fs_d2a2d, f_nco)

    #plot_welch(rx_out, fs=fs_d2a2d, color='blue', label='PSD', fig_flag='new')
    for i in range(nc):
        rx_out = ftc_out

        n = np.arange(len(rx_out))
        nco = np.exp(-1j * 2 * np.pi * config.Nf[i] / Nco_fs * n)
        rx_out = rx_out * nco
        #plot_welch(rx_out, fs=Nco_fs, label=f'Carrier {i} at DC', fig_flag='new')
        if config.resample_factor >= 16:
            rx_out = decimation2(rx_out, tx_fir3)
        if resample_factor >= 8:
            rx_out = decimation2(rx_out, tx_fir3)
        if resample_factor >= 4:
            rx_out = decimation2(rx_out, tx_fir2)
        if resample_factor >= 2:
            rx_out = decimation2(rx_out, tx_fir1)

        #plot_welch(rx_out, fs=fs, color='blue', label='PSD', fig_flag='new')

        all_rx_carriers.append(rx_out)



    return np.array(all_rx_carriers)
