#!/usr/bin/env python3
"""
Carries out the identification or the reconstruction of the sinusoid from experimental data.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import signal
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from pyswarm import pso
from Blade_damage.user_input import dt, total_time
from helper_funcs import compute_BET_signals, pso_cost_function, obtain_wind_correction, experimental_data_extraction, \
    frequency_extraction

mpl.use('TkAgg')


def damaged_prop_signal_id(fn, content, signal_name, mean_wind_correction, BET_model_signal, switch_plot_fft=False,
                           switch_plot_sinusoid_id=False, id_type="PSO"):
    """
    Obtain the mean and the amplitude of the model and experimental signals using the specified technique (PSO or LS)
    :param fn: the number of the next figure to be plotted
    :param content: the dataframe containing the experimental information
    :param signal_name: whether the signal is thrust or torque
    :param mean_wind_correction: the bias correction function due to the wind impinging the test stand
    :param BET_model_signal: the BET signal
    :param switch_plot_fft: whether the fft of the BET signal should be plotted
    :param switch_plot_sinusoid_id: whether the identified sinusoid should be plotted
    :param id_type: the type of identification for the signal
    :return: the number of the next figure, the mean and amplitude of the BET signal, and the mean and amplitude of the
    experimental signal
    """
    time = content['Time (s)']

    # Choose the type of signal to reconstruct
    if signal_name == "T":
        plot_ylabel = 'Thrust (N)'
        wrench_signal = content[plot_ylabel]
    elif signal_name == "N":
        plot_ylabel = 'Torque (N·m)'
        wrench_signal = content[plot_ylabel]
    else:
        raise ValueError(f"The selected signal ({signal_name}) does not exist.")

    # Apply the wind corrections to the signal and detrend it
    wrench_signal = wrench_signal[wrench_signal.notna()] - mean_wind_correction
    wrench_signal_mean = wrench_signal.mean()
    wrench_signal_numpy = wrench_signal.to_numpy()
    detrended_wrench_signal = signal.detrend(wrench_signal_numpy) + wrench_signal_mean

    sampled_times = time[np.asarray(wrench_signal.keys())].to_numpy()
    sampled_times -= sampled_times[0]

    # Plot the data from validation and the detrended data. Figure 9.58 in thesis.
    if switch_plot_sinusoid_id:
        time_local = time - time.iloc[0]
        time_local = time_local[wrench_signal.notna().index]
        wrench_m, wrench_b = np.polyfit(time_local, wrench_signal, 1)
        detrended_wrench_m, detrended_wrench_b = np.polyfit(time_local, detrended_wrench_signal, 1)

        fig = plt.figure(fn)
        fn += 1
        plt.scatter(time_local, wrench_signal, color="#1f77b4", marker="o", label="Data")
        plt.scatter(time_local, detrended_wrench_signal, color="#d62728", marker="o", label="Detrended data")
        plt.plot(time_local, wrench_b+wrench_m*time_local, color="#1f77b4", linestyle="--", linewidth=4)
        plt.plot(time_local, detrended_wrench_b + detrended_wrench_m * time_local, color="#d62728", linestyle="--",
                 linewidth=4)
        plt.ylabel("Thrust [N]")
        plt.xlabel("Time [s]")
        plt.grid(True)
        fig.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
        fig.set_size_inches(19.24, 10.55)
        plt.legend(markerscale=2)

    # Compute the sinusoid
    BET_mean = np.mean(BET_model_signal)
    BET_amplitude = (np.max(BET_model_signal) - np.min(BET_model_signal)) / 2
    if np.max(BET_model_signal) - np.min(BET_model_signal) != 0:

        # Apply PSO
        if id_type == "PSO":
            fn, largest_frequency_wrench_signal = frequency_extraction(fn, BET_model_signal, dt,
                                                                       switch_plot_fft=switch_plot_fft, n_points=None)
            ub = [0.5, 2 * np.pi]
            lb = [0, 0]
            kwargs = {"sinusoid_f": largest_frequency_wrench_signal, "time_lst": sampled_times,
                      "data_lst": detrended_wrench_signal, "mean_sinusoid": wrench_signal_mean}
            xopt, fopt = pso(pso_cost_function, lb, ub, debug=True, swarmsize=5000, maxiter=20, kwargs=kwargs)
            wrench_signal_amplitude = xopt[0]
            wrench_signal_phase = xopt[1]

        # Apply Lomb-Scargle periodogram
        elif id_type == "LS":
            fn, largest_frequency_wrench_signal = frequency_extraction(fn, BET_model_signal, dt,
                                                                       switch_plot_fft=switch_plot_fft, n_points=None)
            t_fit = np.linspace(0, 1, 1000)
            ls = LombScargle(sampled_times, detrended_wrench_signal)
            y_fit = ls.model(t_fit, largest_frequency_wrench_signal/(2*np.pi))
            reconstructed_amplitude = (max(y_fit) - min(y_fit)) / 2
            wrench_signal_amplitude = reconstructed_amplitude
            wrench_signal_phase = np.arcsin(y_fit[0]/wrench_signal_amplitude)
        else:
            raise ValueError(f"The id_type {id_type} is not expected.")

        # Plot the identified sinusoid with the data used for the reconstruction
        if switch_plot_sinusoid_id:
            if id_type == "LS":
                plt.figure(fn)
                fn += 1
                frequency, power = LombScargle(sampled_times, detrended_wrench_signal).autopower(
                    maximum_frequency=largest_frequency_wrench_signal / (2 * np.pi) + 10)
                plt.plot(frequency, power)
                plt.grid(True)

            # Create same plot as before but know with the fitted sinusoid
            data_ps = np.polyfit(sampled_times, wrench_signal_numpy, 1)
            detrended_data_ps = np.polyfit(sampled_times, detrended_wrench_signal, 1)
            plt.figure(fn)
            fn += 1

            # Plotting the data gathered
            plt.scatter(sampled_times, wrench_signal_numpy, color="#1f77b4", marker="o",
                        label="Data")
            plt.plot(sampled_times, sampled_times * data_ps[0] + data_ps[1], color="#1f77b4", linestyle="--",
                     linewidth=4)

            # Plotting the detrended data gathered
            plt.scatter(sampled_times, detrended_wrench_signal, color="#d62728", marker="o",
                        label="Detrended data")
            plt.plot(sampled_times, sampled_times * detrended_data_ps[0] + detrended_data_ps[1], color="#d62728",
                     linestyle="--", linewidth=4)

            # Plotting the data from the model
            plt.plot(np.arange(0, total_time + dt, dt), BET_model_signal, "g--")

            # Plotting the approximated signal
            plt.plot(np.arange(0, np.max(sampled_times), dt), wrench_signal_mean +
                     wrench_signal_amplitude *
                     np.sin(largest_frequency_wrench_signal * np.arange(0, np.max(sampled_times), dt) +
                            wrench_signal_phase), "r-")
            plt.xlabel("Timestamp [s]")
            plt.ylabel(plot_ylabel)
            plt.grid(True)
            plt.legend(markerscale=2)
    else:
        wrench_signal_amplitude = 0

    return fn, BET_mean, BET_amplitude, wrench_signal_mean, wrench_signal_amplitude


if __name__ == "__main__":
    folder_files = "Data_pre-processing\\Data\\2_Pre-processed_data_files"
    folder_files_np = "Data_pre-processing\\Data\\2_Pre-processed_data_files\\No_propeller"
    folder_data_storage = "Data_storage"

    # User input
    blade_damage = 25  # 0, 10, 25
    alpha_angle = 0  # 0, 15, 30, 45, 60, 75, 90
    wind_speed = 2  # (0), 2, 4, 6, 9, 12
    rpm = 500  # 300, 500, 700, 900, 1100
    switch_plot_models = True
    figure_number = 2

    # Obtaining the average rpm
    figure_number, content_df, average_rpms, mean_wind_uncorrected_thrust, std_thrust,mean_wind_uncorrected_torque, \
    std_torque = experimental_data_extraction(figure_number, blade_damage, alpha_angle, wind_speed, rpm, folder_files,
                                              switch_plot_experimental_validation=False, switch_print=False)

    processed_content = pd.read_csv(os.path.join(folder_data_storage, "b0.csv"))

    figure_number, F_healthy_lst, M_healthy_lst, _, _ = \
        compute_BET_signals(figure_number, blade_damage, alpha_angle, wind_speed, average_rpms, dt, switch_plot_models)

    BET_thrust = -F_healthy_lst[-1, :]
    BET_torque = M_healthy_lst[-1, :]

    # Corrections
    figure_number, mean_wind_correction_thrust, mean_wind_correction_torque, _, _ = \
        obtain_wind_correction(figure_number, alpha_angle, wind_speed, folder_files_np)

    # Obtaining the mean and amplitude of the thrust
    figure_number, BET_mean_T, BET_amplitude_T, wrench_signal_mean_T, wrench_signal_amplitude_T = \
        damaged_prop_signal_id(figure_number, content_df, "T", mean_wind_correction_thrust, BET_thrust)

    # Obtaining the mean and amplitude of the torque
    figure_number, BET_mean_N, BET_amplitude_N, wrench_signal_mean_N, wrench_signal_amplitude_N = \
        damaged_prop_signal_id(figure_number, content_df, "N", mean_wind_correction_torque, BET_torque)
