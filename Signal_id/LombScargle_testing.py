#!/usr/bin/env python3
"""
Tests the accuracy of the Lomb-Scargle periodogram for signal reconstruction. In this case, the correct discovery of
a sinusoid amplitude which has been polluted with Gaussian noise. This file also allows the creation of a plot that
shows the reconstructed amplitude with respect to the standard deviation of the Gaussian noise polluting the original
clean sinusoid.
"""

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


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
# mpl.use('Agg')
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)


# Functions
def obtain_reconstructed_amplitude(figure_number, sinusoid_f, sampling_f, amplitude, bias, phase, std, time_std, max_time,
                                   switch_time_innaccuracy, switch_plotting):
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
    :param switch_time_innaccuracy: whether the time innacuracy should be introduced, meaning polluting the sampling
    time with Gaussian noise
    :param switch_plotting: whether plots should be generated
    :return:
    """
    def sample_noisy_sinusoid(x, A, f, bias, phase, std):
        """
        Provides a time step of a sinusoid signal with white noise
        :param x: time step
        :param A: amplitude
        :param f: sine frequency
        :param bias: signal vertical displacement
        :param phase: phase of the signal
        :param std: standard deviation for the Gaussian noise sampling
        :return:
        """
        omega = 2 * np.pi * f
        noise = np.random.normal(0, std)
        return bias + A*np.sin(omega*x + np.deg2rad(phase)) + noise #+ 2*A*np.sin(omega*x/2 + np.deg2rad(phase))

    # Check whether time error should be applied
    if not switch_time_innaccuracy:
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
        # Generate plot with original sinusoid and samples
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
        plt.plot(np.array(t)[np.array(t) <= 0.5], np.array(y)[np.array(t) <= 0.5], 'o', color="#2ca02c", markersize=10, label="Noisy signal samples")
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [N]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
        f.subplots_adjust(left=0.085, top=0.94, right=0.98, bottom=0.13)
        plt.grid(True)
        leg = plt.legend(markerscale=2)
        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)

        # Generate power plot
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(frequency, power)
        plt.grid(True)

        # Plot the fitted sinusoid
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
        for line in leg.get_lines():
            line.set_linewidth(4.0)

    return figure_number, reconstructed_amplitude


def analyse_noise(figure_number, scenario_data, start, end, step):
    """
    Function to plot what is the mean and standard deviation of the reconstructed oscillations' amplitude for different
    values of white noise in the signal
    :param figure_number: number of the next figure
    :param scenario_data: the ground truth information of the signal to reconstruct
    :param start: the start of the white noise standard deviation range
    :param end: the end of the white noise standard deviation range
    :param step: the step of the white noise standard deviation range
    :return:
    """
    std_lst = np.arange(start, end, step)
    reconstructed_amplitude_mean_lst = np.zeros(len(std_lst))
    reconstructed_amplitude_std_lst = np.zeros(len(std_lst))
    amplitude_lst = np.zeros(100)
    counter = 0
    for std in std_lst:
        scenario_data[5] = std
        for i in range(100):
            figure_number, reconstructed_amplitude = obtain_reconstructed_amplitude(figure_number, *scenario_data)
            amplitude_lst[i] = reconstructed_amplitude
        amplitude_mu, amplitude_std = norm.fit(amplitude_lst)
        reconstructed_amplitude_mean_lst[counter] = amplitude_mu
        reconstructed_amplitude_std_lst[counter] = amplitude_std
        counter += 1

    f = plt.figure(figure_number)
    figure_number += 1
    plt.errorbar(std_lst, reconstructed_amplitude_mean_lst, yerr=reconstructed_amplitude_std_lst * 1.96,
                     color="#d62728", capsize=4, marker="o", markersize=10, alpha=0.5)
    plt.axhline(scenario_data[2], color="k", linestyle="--")
    # plt.plot(std_lst, reconstructed_amplitude_mean_lst)
    # plt.axhline(scenario_data[2], color='r', linestyle='--')
    # plt.plot(std_lst, scenario_data[2]+1.96*std_lst, 'go')
    plt.xlabel("Sinusoid white noise standard deviation [-]")
    plt.ylabel("Reconstructed amplitude [-]")
    plt.grid(True)
    f.subplots_adjust(left=0.1, top=0.94, right=0.98, bottom=0.13)
    return figure_number, reconstructed_amplitude_mean_lst


def discover_t_std(file="b0_a0_w0_r900.csv"):
    """
    Function to discover what is the sampling frequency and the deviation from that frequency in the form of a standard
    deviaton
    :param file: name of the file to be used for the computation of the aforementioned parameters.
    :return:
    """
    folder = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\" \
             "2nd Campaign\\Data\\2_Pre-processed_data_files"
    raw_data = pd.read_csv(os.path.join(folder, file))
    time_stamps = raw_data['Time (s)'][raw_data['Thrust (N)'].notna()]
    time_steps = [time_stamps.iloc[i] - time_stamps.iloc[i - 1] for i in range(1, len(time_stamps))]
    time_mu, time_std = norm.fit(time_steps)
    print(f"sampling_f={1/time_mu} and time_std={time_std}")
    return 1/time_mu, time_std


if __name__ == "__main__":
    # User input
    chosen_scenario = 3
    switch_plotting = False
    figure_number = 1
    std_start = 0
    std_step = 0.0001
    std_end = 0.02+std_step

    # Obtain scenario data
    scenario_data = select_scenario(chosen_scenario)
    scenario_data.append(switch_plotting)
    # figure_number, reconstructed_amplitude = obtain_reconstructed_amplitude(figure_number, *scenario_data)

    # Plot effect of noise
    sampling_f, time_std = discover_t_std()
    scenario_data[1] = sampling_f
    scenario_data[6] = time_std
    analyse_noise(figure_number, scenario_data, std_start, std_end, std_step)


