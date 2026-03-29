import numpy as np
from scipy.signal import lfilter


def pa_amp(signal, model_type='rapp', **kwargs):
    """
    Unified Power Amplifier Impairment Function.

    :param signal: Complex baseband input signal (numpy array)
    :param model_type: 'rapp', 'poly', or 'volterra'
    :param kwargs: Model-specific parameters
    :return: Distorted complex signal
    """
    gain1  = 16
    signal = signal * gain1
    # Ensure input is a numpy array
    x = np.array(signal, dtype=complex)

    if model_type.lower() == 'rapp':
        # SSPA soft-clipping (AM/AM)
        p = kwargs.get('p', 2.0)
        v_sat = kwargs.get('v_sat', 1.0)
        gain = 10 ** (kwargs.get('gain_db', 0) / 20)

        amplitude = np.abs(x)
        phase = np.angle(x)
        out_amp = (gain * amplitude) / (1 + (gain * amplitude / v_sat) ** (2 * p)) ** (1 / (2 * p))
        out = out_amp * np.exp(1j * phase)

    elif model_type.lower() == 'poly':
        # Memoryless Polynomial (AM/AM + AM/PM)
        # Default coefficients represent slight compression
        c1 = kwargs.get('c1', 1.0 + 0j)
        c3 = kwargs.get('c3', -0.05 - 0.01j)
        c5 = kwargs.get('c5', -0.002 - 0.0005j)

        mag_sq = np.abs(x) ** 2
        out = c1 * x + c3 * mag_sq * x + c5 * (mag_sq ** 2) * x

    elif model_type.lower() == 'volterra':
        # Memory Polynomial Model (Simplified Volterra)
        # coeffs: list of arrays [[a10, a11...], [a30, a31...], [a50, a51...]]
        # Default: 1st and 3rd order with memory depth 2
        memory_depth = kwargs.get('memory_depth', 2)

        # Default coeffs if none provided (Linear gain 1.0, 3rd order compression)
        a1 = kwargs.get('a1', np.array([1.0, -0.1]))  # Linear part with memory
        a3 = kwargs.get('a3', np.array([-0.05, -0.02]))  # 3rd order with memory

        # Apply filters for each non-linear branch
        branch1 = lfilter(a1, [1], x)
        branch3 = lfilter(a3, [1], x * np.abs(x) ** 2)

        out = branch1 + branch3

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return out/gain1
# --- Quick Test ---
# lte_signal = np.random.randn(1000) + 1j*np.random.randn(1000)
# out_rapp = pa_amp(lte_signal, 'rapp', p=3, v_sat=1.2)
# out_volt = pa_amp(lte_signal, 'volterra', a1=[0.9, 0.1], a3=[-0.04, -0.01])