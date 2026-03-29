import numpy as np

cfr_method = 'CORDIC' # HC = Hard clipping  CORDIC

def calculate_cfr(x):
    """Calculates Crest Factor (PAPR) in dB."""
    peak_amp = np.max(np.abs(x))
    rms_amp = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms_amp == 0: return 0
    return 20 * np.log10(peak_amp / rms_amp)


def cfr_cordic(signal_in, cfr_max_db, num_iter=9):
    # 1. Setup Parameters
    rms_linear = np.sqrt(np.mean(np.abs(signal_in) ** 2))
    limit_linear = rms_linear * (10 ** (cfr_max_db / 20))

    # CORDIC Gain (G) and Inverse Gain (1/G)
    # 1/1.6467 is approx 0.60725
    cordic_gain = 1.646760258
    inv_cordic_gain = 1.0 / cordic_gain

    atan_lut = np.arctan(1.0 / (2.0 ** np.arange(num_iter)))

    magnitudes = np.abs(signal_in)
    over_limit = magnitudes > limit_linear

    signal_out = np.copy(signal_in)

    # Process only samples that exceed the threshold
    for idx in np.where(over_limit)[0]:
        sample = signal_in[idx]
        x_orig = sample.real
        if x_orig < 0:
            neg_flag = -1
        else:
            neg_flag = 1

        x = x_orig * neg_flag
        y = sample.imag
        z = 0.0
        x_ = limit_linear * inv_cordic_gain
        y_ = 0.0
        # --- PHASE 1: VECTORING MODE ---
        # Get the phase (z)
        for i in range(num_iter):
            d = -1.0 if y < 0 else 1.0
            x_new = x + y * d * (2.0 ** -i)
            y_new = y - x * d * (2.0 ** -i)
            x_new_ = x_ - y_ * d * (2.0 ** -i)
            y_new_ = y_ + x_ * d * (2.0 ** -i)

            #z_new = z + d * atan_lut[i]
            x, y = x_new, y_new
            x_, y_ = x_new_, y_new_



        # No scaling needed here! x and y are now at the correct magnitude.
        signal_out[idx] = x_ * neg_flag + 1j * y_

    return signal_out


def cfr_hc(signal_in, cfr_max_db):
    # 1. Calculate the Vector Limit (The circular radius)
    rms_linear = np.sqrt(np.mean(np.abs(signal_in) ** 2))
    limit_vector = rms_linear * (10 ** (cfr_max_db / 20))

    # 2. Adjust Limit for Independent I and Q (The square box)
    # We use / sqrt(2) so the corners of the box don't exceed the vector peak
    limit_comp = limit_vector / np.sqrt(2)

    # 3. Independent Clipping
    real_clipped = np.clip(signal_in.real, -limit_comp, limit_comp)
    imag_clipped = np.clip(signal_in.imag, -limit_comp, limit_comp)

    signal_out = real_clipped + 1j * imag_clipped

    return signal_out


def cfr(signal_in, cfr_max_db):
    # 1. Calculate input RMS and current CFR
    cfr_in_db = calculate_cfr(signal_in)

    if cfr_method == 'HC':
        signal_out=cfr_hc(signal_in, cfr_max_db)
    elif cfr_method == 'CORDIC':
        signal_out=cfr_cordic(signal_in, cfr_max_db)
    # 4. Calculate output CFR
    else:
        signal_out = signal_in


    cfr_out_db = calculate_cfr(signal_out)

    print(f"CFR In-Out:  {cfr_in_db:.2f} dB   {cfr_out_db:.2f} dB")

    return signal_out, cfr_in_db, cfr_out_db