from scipy import signal
from scipy.fft import rfft, irfft, rfftfreq
from statsmodels.tsa.stattools import acf, pacf

val_a, limits_a = acf(output_ar, nlags=30, alpha=0.05)
val_p, limits_p = pacf(output_ar, nlags=30, alpha=0.05)

temp_array_a = np.abs(val_a)
temp_limit_a = np.array([0.5 * (upper - lower) for lower, upper in limits_a])
temp_diff_array_a = np.array(
    [float(temp_array_a[idx]) if temp_array_a[idx] >= temp_limit_a[idx] else 0.0 for idx in range(len(temp_array_a))])

temp_array_p = np.abs(val_p)
temp_limit_p = np.array([0.5 * (upper - lower) for lower, upper in limits_p])
temp_diff_array_p = np.array(
    [float(temp_array_p[idx]) if temp_array_p[idx] >= temp_limit_p[idx] else 0.0 for idx in range(len(temp_array_p))])

spectrum_a = rfft(temp_diff_array_a)
amplitudes_a = np.abs(spectrum_a) / len(temp_diff_array_a)
phases_a = np.angle(spectrum_a)
frequencies_a = rfftfreq(len(temp_diff_array_a))

spectrum_p = rfft(temp_diff_array_p)
amplitudes_p = np.abs(spectrum_p) / len(temp_diff_array_p)
phases_p = np.angle(spectrum_p)
frequencies_p = rfftfreq(len(temp_diff_array_p))

spectrum_a_i = (amplitudes_a * np.cos(phases_a) + 1j * amplitudes_a * np.sin(phases_a)) * len(temp_diff_array_a)
original_a = irfft(spectrum_a_i, n=len(temp_diff_array_a))

# print(spectrum_a)
# print(amplitudes_a)
# print(phases_a)

# Graphic results

import matplotlib.pyplot as plt

# Input data

# Solution
_, ax = plt.subplots(figsize=(10, 4))

ax.stem(frequencies_a, amplitudes_a)

ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
# ax.minorticks_on()
# ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
ax.set_title(ds_seasonal_name, pad=10, loc='center', color='black')
ax.set_xlabel("Дата", labelpad=10, loc='center', color='black')
ax.set_ylabel("Значення", labelpad=10, loc='center', color='black')

plt.show()

# Graphic results

import matplotlib.pyplot as plt

# Input data

# Solution
_, ax = plt.subplots(figsize=(10, 4))

ax.stem(frequencies_a, phases_a)

ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
# ax.minorticks_on()
# ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
ax.set_title(ds_seasonal_name, pad=10, loc='center', color='black')
ax.set_xlabel("Дата", labelpad=10, loc='center', color='black')
ax.set_ylabel("Значення", labelpad=10, loc='center', color='black')

plt.show()