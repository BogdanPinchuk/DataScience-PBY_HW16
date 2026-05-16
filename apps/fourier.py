import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt


def get_signal_spectrum(signal_data: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get signal spectrum
    :param signal_data: signal array
    :return: (spectrum, amplitude, phase, frequency)
    """
    n_signal = len(signal_data)
    spectrum = rfft(signal_data)
    amplitudes = np.asarray(np.abs(spectrum) / n_signal)
    phases = np.asarray(np.angle(spectrum))
    frequencies = rfftfreq(n_signal)

    return spectrum, amplitudes, phases, frequencies


def plot_signal_spectrum_by_components(amplitudes: np.ndarray,
                                       phases: np.ndarray,
                                       frequencies: np.ndarray,
                                       signal_data: np.ndarray,
                                       figsize: tuple[float, float],
                                       restored_signal_data: np.ndarray = None) -> None:
    """
    Plot signal spectrum
    :param amplitudes: Amplitudes of the spectrum
    :param phases: Phases of the spectrum
    :param frequencies: Frequencies of the spectrum
    :param signal_data: Signal
    :param figsize: figure size
    :param restored_signal_data: Recovered signal
    """
    _, axes = plt.subplots(3, 1, figsize=figsize)

    # Amplitude
    ax = axes[0]
    ax.stem(frequencies, amplitudes)
    ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
    # ax.minorticks_on()
    # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
    ax.set_ylabel("Amplitude", labelpad=10, loc='center', color='black')
    # Frequency
    ax.set_xlabel("Frequency", labelpad=10, loc='center', color='black')

    # Phase
    ax = axes[1]
    ax.stem(frequencies, phases)
    ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
    # ax.minorticks_on()
    # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
    ax.set_ylabel("Phase", labelpad=10, loc='center', color='black')
    # Frequency
    ax.set_xlabel("Frequency", labelpad=10, loc='center', color='black')

    # Signal
    ax = axes[2]

    if restored_signal_data is None:
        ax.plot(signal_data, label="original signal")
    else:
        ax.plot(signal_data, linewidth=5.0, alpha=0.5, label="original signal")
        ax.plot(restored_signal_data, color="red", linewidth=1.0, label="original signal")

    ax.grid(axis='both', visible=True, which='major', ls='--', linewidth=1.0, color='tab:gray')
    # ax.minorticks_on()
    # ax.grid(axis='both', visible=True, which='minor', ls=':', linewidth=0.5, color='tab:green')
    ax.set_ylabel("Signal", labelpad=10, loc='center', color='black')
    ax.set_xlabel("Sample", labelpad=10, loc='center', color='black')

    plt.suptitle(f"Spectrum & Signal")

    plt.tight_layout()
    plt.show()


def plot_signal_spectrum(signal_data: np.ndarray, figsize: tuple[float, float]) -> None:
    """
    Plot signal spectrum
    :param signal_data: signal array
    :param figsize: figure size
    """
    _, amplitudes, phases, frequencies = get_signal_spectrum(signal_data)

    plot_signal_spectrum_by_components(amplitudes, phases, frequencies, signal_data, figsize)


def modify_signal_spectrum(signal_data: np.ndarray,
                           hr_lower_lim: float = 1.0,
                           hr_upper_lim: float = 0.0,
                           vr_lower_lim: float = 1.0,
                           vr_upper_lim: float = 0.0,
                           figsize: tuple[float, float] = None) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Modify the signal spectrum
    :param signal_data: original signal data
    :param hr_lower_lim: (0,1), resets the values to zero from left (hr_lower_lim) to right (end or hr_upper_lim)
    :param hr_upper_lim: (0,1), resets the values to zero from left (start or hr_lower_lim) to right (hr_upper_lim)
    :param vr_lower_lim: (0,1), resets the abs(values) to zero from down (vr_lower_lim) to up (end or vr_upper_lim)
    :param vr_upper_lim: (0,1), resets the abs(values) to zero from down (start or vr_lower_lim) to up (vr_upper_lim)
    :param figsize: figure size, if value is "None", the plot won't be shown
    :return: (spectrum, amplitude, phase, frequency, recovered signal data)
    """
    config_array = {hr_lower_lim, hr_upper_lim, vr_lower_lim, vr_upper_lim}
    for item in config_array:
        if item < 0.0 or 1.0 < item:
            raise ValueError("Invalid input parameter, it should be between 0.0 and 1.0!")

    n_signal = len(signal_data)
    spectrum, amplitudes, phases, frequencies = get_signal_spectrum(signal_data)

    n_spectrum = len(spectrum)
    idx_hr_lower = int(n_spectrum * hr_lower_lim)
    idx_hr_upper = int(n_spectrum * hr_upper_lim)

    max_amplitude = np.max(np.abs(amplitudes))
    level_vr_lower = max_amplitude * vr_lower_lim
    level_vr_upper = max_amplitude * vr_upper_lim

    changed_amplitudes = []
    changed_phases = []
    for idx in range(n_spectrum):
        amplitude = amplitudes[idx]
        phase = phases[idx]

        # horizontal cutting
        if idx_hr_lower < idx_hr_upper:
            if idx_hr_lower <= idx <= idx_hr_upper:
                amplitude = 0.0
                phase = 0.0
        else:
            if idx < idx_hr_upper or idx_hr_lower < idx:
                amplitude = 0.0
                phase = 0.0

        # vertical cutting
        abs_amplitude = np.abs(amplitude)
        if level_vr_lower < level_vr_upper:
            if level_vr_lower <= abs_amplitude <= level_vr_upper:
                amplitude = 0.0
                phase = 0.0
        else:
            if abs_amplitude < level_vr_upper or level_vr_lower < abs_amplitude:
                amplitude = 0.0
                phase = 0.0

        changed_amplitudes.append(amplitude)
        changed_phases.append(phase)

    changed_amplitudes = np.asarray(changed_amplitudes)
    changed_phases = np.asarray(changed_phases)

    # alternative
    # restored_spectrum = (changed_amplitudes * np.cos(changed_phases) + 1j * changed_amplitudes * np.sin(changed_phases)) * n_signal
    restored_spectrum = changed_amplitudes * np.exp(1j * changed_phases) * n_signal
    restored_signal = irfft(restored_spectrum, n=n_signal)

    if figsize is not None:
        plot_signal_spectrum_by_components(changed_amplitudes, changed_phases, frequencies,
                                           signal_data, figsize, restored_signal)

    return restored_spectrum, changed_amplitudes, changed_phases, frequencies, restored_signal
