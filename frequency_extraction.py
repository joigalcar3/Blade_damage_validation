import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


def frequency_extraction(figure_number, signal, dt, switch_plot_fft=False, n_points=None):
    if n_points is None:
        n_points = signal.shape[0]

    # Output in the form of cycles/s. Transformed to rad/s
    yf = fft(signal)[0:n_points // 2]
    xf = fftfreq(n_points, dt)[:n_points // 2] * 2 * np.pi

    if switch_plot_fft:
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(xf, 2.0/n_points * np.abs(yf), color="r", marker="o")
        plt.ylabel("Amplitude [-]")
        plt.xlabel("Frequency [rad/s]")
        plt.grid()

    largest_frequency = xf[np.where(abs(yf) == max(abs(yf[1:])))[0].item()]

    return figure_number, largest_frequency
