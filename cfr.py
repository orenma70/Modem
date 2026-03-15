import numpy as np


def calculate_cfr(x):
    """Calculates Crest Factor (PAPR) in dB."""
    peak_amp = np.max(np.abs(x))
    rms_amp = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms_amp == 0: return 0
    return 20 * np.log10(peak_amp / rms_amp)


def cfr(signal_in, cfr_max_db):
    # 1. Calculate input RMS and current CFR
    rms_linear = np.sqrt(np.mean(np.abs(signal_in) ** 2))
    cfr_in_db = calculate_cfr(signal_in)

    # 2. Convert dB threshold to linear amplitude limit
    # Threshold = RMS_linear * 10^(cfr_max_db / 20)
    limit_linear = rms_linear * (10 ** (cfr_max_db / 20))

    # 3. Hard Clipping
    magnitudes = np.abs(signal_in)
    # Create a scaling factor for points exceeding the limit
    scale = np.ones_like(magnitudes)
    over_limit = magnitudes > limit_linear
    scale[over_limit] = limit_linear / magnitudes[over_limit]

    signal_out = signal_in * scale

    # 4. Calculate output CFR
    cfr_out_db = calculate_cfr(signal_out)

    return signal_out, cfr_in_db, cfr_out_db