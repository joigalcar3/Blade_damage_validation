#!/usr/bin/env python3
"""
Tests the accuracy of the Lomb-Scargle periodogram for signal reconstruction. In this case, the correct discovery of
a sinusoid amplitude which has been polluted with Gaussian noise. This file also allows the creation of a plot that
shows the reconstructed amplitude with respect to the standard deviation of the Gaussian noise polluting the original
clean sinusoid.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


import numpy as np
import pandas as pd
import os
from astropy.timeseries import LombScargle
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from Signal_id.sinusoid_scenarios import *
mpl.use('TkAgg')
np.random.seed(1)

# Matplotlib settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)


# Functions
def obtain_reconstructed_amplitude(figure_number, sinusoid_f, sampling_f, amplitude, bias, phase, std, time_std,
                                   max_time, switch_time_inaccuracy, switch_plotting):
    """
    Reconstructs a sinusoid and provides the amplitude of the reconstructed signal
    :param figure_number: number of the next figure
    :param sinusoid_f: frequency of the sine signal
    :param sampling_f: sampling frequency
    :param amplitude: amplitude of the sinusoid
    :param bias: vertical displacement of the sinusoid
    :param phase: phase of the sinusoid
    :param std: standard deviation of the Gaussian noise polluting the sinusoid
    :param time_std: standard deviation of the Gaussian noise polluting the sampling time
    :param max_time: maximum time that the sinusoid will be running for
    :param switch_time_inaccuracy: whether the time inaccuracy should be introduced, meaning polluting the sampling
    time with Gaussian noise
    :param switch_plotting: whether plots should be generated
    :return: number of the next figure and the magnitude of the Lomb-Scargle reconstructed amplitude
    """
    def sample_noisy_sinusoid(x, A, fq, B, phi, noise_std):
        """
        Provides a time step of a sinusoid signal with white noise
        :param x: time step
        :param A: amplitude
        :param fq: sine frequency
        :param B: signal vertical displacement (bias)
        :param phi: phase of the signal
        :param noise_std: standard deviation for the Gaussian noise sampling
        :return: single time step of the noisy sinusoid
        """
        omega = 2 * np.pi * fq
        noise = np.random.normal(0, noise_std)
        return B + A*np.sin(omega*x + np.deg2rad(phi)) + noise  # 2*A*np.sin(omega*x/2 + np.deg2rad(phase))

    # Check whether time error should be applied
    if not switch_time_inaccuracy:
        time_std = 0

    # Generating the noisy signal
    t = np.arange(0, max_time, 1/sampling_f) + np.random.normal(0, time_std)
    y = [sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, std) for i in t]

    frequency, power = LombScargle(t, y).autopower(minimum_frequency=sinusoid_f/2, maximum_frequency=sinusoid_f+10)

    # Fit sinusoid to the sample points
    t_fit = np.linspace(0, 0.5, 10000)
    ls = LombScargle(t, y)
    y_fit = ls.model(t_fit, sinusoid_f)
    reconstructed_amplitude = (max(y_fit)-min(y_fit))/2
    print(f"Actual sinusoid amplitude: {amplitude}. Reconstructed amplitude: {reconstructed_amplitude}. Std = {std}")

    if switch_plotting:
        # Generate plot with original sinusoid and samples. Used for the generation of Figure F.1 in thesis
        f = plt.figure(figure_number)
        ax = plt.gca()
        figure_number += 1
        sinusoid_time = np.arange(0, 0.5, 0.00001)
        plt.plot(sinusoid_time, [sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, 0)
                                 for i in sinusoid_time], color="#d62728", alpha=1, label="Clean signal")
        plt.plot(sinusoid_time, [sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, std)
                                 for i in sinusoid_time], color="#1f77b4", alpha=0.5, label="Noisy signal")
        plt.plot(sinusoid_time, [sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, 0)
                                 for i in sinusoid_time], color="#d62728", alpha=0.5)
        plt.plot(np.array(t)[np.array(t) <= 0.5], np.array(y)[np.array(t) <= 0.5], 'o', color="#2ca02c", markersize=10,
                 label="Noisy signal samples")
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [N]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
        f.subplots_adjust(left=0.085, top=0.94, right=0.98, bottom=0.13)
        plt.grid(True)
        leg = plt.legend(markerscale=2)

        # Change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)

        # Generate power plot
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(frequency, power)
        plt.grid(True)

        # Plot the fitted sinusoid. Used for the generation of Figure F.2 in thesis
        f = plt.figure(figure_number)
        ax = plt.gca()
        figure_number += 1
        plt.plot(np.array(sinusoid_time)[sinusoid_time <= 0.5],
                 np.array([sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, 0) for i in sinusoid_time])
                 [sinusoid_time <= 0.5], color="#d62728", label="Clean signal")
        plt.plot(t_fit, y_fit, label="Fitted sinusoid",  color="#8c564b")
        plt.plot(np.array(t)[np.array(t) <= 0.5], np.array(y)[np.array(t) <= 0.5], 'o', color="#2ca02c",
                 label="Noisy signal samples", markersize=10)
        plt.axhline(np.mean(y_fit), color="k", linestyle="--", label="Fitted mean")
        plt.axhline(np.mean([sample_noisy_sinusoid(i, amplitude, sinusoid_f, bias, phase, 0) for i in sinusoid_time]),
                    color="k", label="True mean")
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [N]")
        plt.grid(True)
        ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1] * 1.2))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
        f.subplots_adjust(left=0.085, top=0.94, right=0.98, bottom=0.13)
        plt.grid(True)
        leg = plt.legend(markerscale=2, ncol=2)

        # Change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)

    return figure_number, reconstructed_amplitude


def analyse_noise(figure_number, scenario_data, start, end, step):
    """
    Function to plot what is the mean and standard deviation of the reconstructed oscillations' amplitude for different
    values of white noise in the signal
    :param figure_number: number of the next figure
    :param scenario_data: the ground truth information of the signal to reconstruct (not the same as the signal data)
    :param start: the start of the white noise standard deviation range
    :param end: the end of the white noise standard deviation range
    :param step: the step of the white noise standard deviation range
    :return: number of next figure and mean reconstructed amplitude for the different degrees of white noise std
    """
    # Create list of white noise standard deviations
    std_lst = np.arange(start, end, step)
    reconstructed_amplitude_mean_lst = np.zeros(len(std_lst))
    reconstructed_amplitude_std_lst = np.zeros(len(std_lst))
    amplitude_lst = np.zeros(100)
    counter = 0

    # Iterating over all the standard deviations
    for std in std_lst:
        scenario_data[5] = std

        # Run a 100 random scenarios in order to retrieve a mean and standard deviation
        for i in range(100):
            figure_number, reconstructed_amplitude = obtain_reconstructed_amplitude(figure_number, *scenario_data)
            amplitude_lst[i] = reconstructed_amplitude
        amplitude_mu, amplitude_std = norm.fit(amplitude_lst)
        reconstructed_amplitude_mean_lst[counter] = amplitude_mu
        reconstructed_amplitude_std_lst[counter] = amplitude_std
        counter += 1

    # Plot the distributions with an error bar figure. Used for the generation of Figure F.3 in thesis
    f = plt.figure(figure_number)
    figure_number += 1
    plt.errorbar(std_lst, reconstructed_amplitude_mean_lst, yerr=reconstructed_amplitude_std_lst * 1.96,
                 color="#d62728", capsize=4, marker="o", markersize=10, alpha=0.5)
    plt.axhline(scenario_data[2], color="k", linestyle="--")
    plt.xlabel("Sinusoid white noise standard deviation [-]")
    plt.ylabel("Reconstructed amplitude [-]")
    plt.grid(True)
    f.subplots_adjust(left=0.1, top=0.94, right=0.98, bottom=0.13)
    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return figure_number, reconstructed_amplitude_mean_lst


def discover_t_std(preprocessed_data_directory, file="b0_a0_w0_r900.csv"):
    """
    Function to discover what is the sampling frequency and the deviation from that frequency in the form of a standard
    deviation
    :param preprocessed_data_directory: directory where the pre-processed data is located. The variable "file" should
    be in this directory
    :param file: name of the file to be used for the computation of the aforementioned parameters.
    :return: the average sampling frequency and the standard deviation of the sampling frequency
    """
    raw_data = pd.read_csv(os.path.join(preprocessed_data_directory, file))

    # Obtain the timestamps at which the thrust is measured
    time_stamps = raw_data['Time (s)'][raw_data['Thrust (N)'].notna()]

    # Make the first time step as the 0s timestep
    time_steps = [time_stamps.iloc[i] - time_stamps.iloc[i - 1] for i in range(1, len(time_stamps))]

    # Retrieve the mean and the standard deviation
    time_mu, time_std = norm.fit(time_steps)
    print(f"sampling_f={1/time_mu} and time_std={time_std}")
    return 1/time_mu, time_std


if __name__ == "__main__":
    # User input
    chosen_scenario = 3
    s_plotting = False
    fig_number = 1
    std_start = 0
    std_step = 0.0001
    std_end = 0.01+std_step
    folder = "Data_pre-processing\\Data\\2_Pre-processed_data_files"

    # Obtain scenario data
    scen_data = select_scenario(chosen_scenario)
    scen_data.append(s_plotting)
    fig_number, reconstructed_A = obtain_reconstructed_amplitude(fig_number, *scen_data)

    # Plot effect of noise
    samp_f, t_std = discover_t_std(folder)
    scen_data[1] = samp_f
    scen_data[6] = t_std
    analyse_noise(fig_number, scen_data, std_start, std_end, std_step)
