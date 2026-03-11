

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
Fs=1996.8
BW=1400
rel_bw=BW/Fs
w_p=10 # 10 bits
farrow_len=7
farrow_order=5

M=32


sb_freq_factor=0.95

pass_freq=rel_bw/(2*M)
stop_freq=sb_freq_factor*(1/M-pass_freq)


numtaps = M * farrow_len + 1
bands = [0, 2*pass_freq, 2*stop_freq, 1.0]
desired = [1, 1, 0, 0]
weights = [1, 200]

lpf = signal.firls(numtaps, bands, desired, weight=weights)

t = np.linspace(-farrow_len/2, farrow_len/2, numtaps)

plt.figure(figsize=(10, 4))
plt.stem(t, lpf)
plt.title("FIR Least-Squares Filter (Impulse Response)")
plt.xlabel("Time Offset")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.show()

# --- 3. Polynomial Fitting ---
mu = np.linspace(-0.5, 0.5, M + 1)
p = np.zeros((farrow_len, farrow_order))

plt.figure(figsize=(10, 6))

for i in range(farrow_len):
    # Slice the lpf: in Python, the slice is [start : end_exclusive]
    lpf_i = lpf[i * M: (i + 1) * M + 1]

    # Plotting segments (aligned to match MATLAB offsets)
    t_offset = -(farrow_len + 1) / 2 + (i + 1) + mu
    plt.plot(t_offset, lpf_i, 'r.')

    # polyfit returns highest power first: [a_n, ..., a_1, a_0]
    p[i, :] = np.polyfit(mu, lpf_i, farrow_order - 1)

# --- 4. Conversions and Gain Scaling ---
# fliplr flips the matrix horizontally, moving constant term (a_0) to the first column
P = np.fliplr(p)

# Gain calculations
Pg2unity = np.floor(1.0 / np.max(np.abs(P)))
fp_gain = Pg2unity * 2 ** (w_p - 1)

# Final integer coefficients
# Note: 1.5657 is your custom normalization factor
Pint = np.round(fp_gain * P / 1.5657).astype(int)

# --- Output ---
print("Pint Matrix (Fixed-Point):")
print(Pint)


'''
function out=lagrange ftc(in,m,tx, info)
hw_phase len=8;
acc_init=0;
ftc_n=1;
out_size=ceil(length(in) /m);
acc = acc_init+(0:(out_size-1)) * m; % timing phase accumalator values in Fin UI
sh = floor(acc/ftc_n);
acc = acc-sh*ftc_n;
mu=acc-1/2;
cbhr_all=[in;zeros(hw_phase len,1)];
cbhr=cbr_all(kron(sh',ones(1,hw_ phase len))+kron(1l:hw_phase len,ones(out_size,1)));
cbr=reshape (cbr, out_size,hw_ phase len); % avoid a failure when length (idx)=1
P=lagrange coef (hw _phase len);
P=round (P*2*%12)/2°12;
eee ONE
method='farrow';
switch method
case 'polyval'
x=zeros (hw_phase_ len,out_size);
for r=l:hw_phase len
x(r,:)=polyval (P(r,:),mu);
end
out=sum(cbr.*x',2);
case 'farrow'

z=cbr*P;

x=zeros(out_size,1);

for r=l:hw_phase len-1

x=(xt+z(:,r)).*mu';

end

out=x+z(:,hw_phase len);
end

'''






