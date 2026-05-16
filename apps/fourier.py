import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt


def get_signal_spectrum(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get signal spectrum
    :param array:
    :return: (spectrum, amplitude, phase, frequency)
    """
    n_array = len(array)
    spectrum = rfft(array)
    amplitudes = np.asarray(np.abs(spectrum) / n_array)
    phases = np.asarray(np.angle(spectrum))
    frequencies = rfftfreq(n_array)

    return spectrum, amplitudes, phases, frequencies


def plot_signal_spectrum_by_components(amplitudes: np.ndarray,
                                       phases: np.ndarray,
                                       frequencies: np.ndarray,
                                       signal_data: np.ndarray,
                                       figsize: tuple[float, float]) -> None:
    """
    Plot signal spectrum
    :param amplitudes: Amplitudes of the spectrum
    :param phases: Phases of the spectrum
    :param frequencies: Frequencies of the spectrum
    :param signal_data: Signal
    :param figsize: figure size
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
    ax.plot(signal_data)
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
