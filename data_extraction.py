#!/usr/bin/env python3
"""
Retrieves the data from the pre-processed files, including the signal reconstruction. The data retrieved is saved in
files of the format "bx_rpms" within the "Data_storage" folder.
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
from Blade_damage.user_input import dt
from helper_funcs import compute_BET_signals, obtain_wind_correction, experimental_data_extraction
from damaged_prop_signal_id import damaged_prop_signal_id

mpl.use('TkAgg')

# Data filepath
folder_files = "Data_pre-processing\\Data\\2_Pre-processed_data_files"
folder_files_np = "Data_pre-processing\\Data\\2_Pre-processed_data_files\\No_propeller"


def data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                    switch_plot_experimental_validation, switch_plot_models, switch_wind_correction, data_file_name,
                    switch_plot_fft=False, switch_plot_sinusoid_id=False, id_type="PSO"):
    """
    Extract information from the wind tunnel signals and the models
    :param figure_number: number of the next figureto plot
    :param blade_damage_lst: the list of blade damages considered
    :param alpha_angle_lst: the list of angles of attack considered
    :param wind_speed_lst: the list of wind speeds considered
    :param rpm_lst: the list of rpm values considered
    :param switch_plot_experimental_validation: whether the thrust and torque wind data corrections should be plotted
    :param switch_plot_models: whether to plot the thrust and torque data from the models
    :param switch_wind_correction: whether the signals should be corrected with the data from the wind tunnel without
    the propeller
    :param data_file_name: the name of the file where the data table will be stored within Data_storage
    :param switch_plot_fft: whether to plot the FFT signal
    :param switch_plot_sinusoid_id: whether to plot the identified sinusoid with PSO or LS
    :param id_type: identification method used
    :return: the number of the next figure to be plotted
    """
    data_table = []

    # Different type of data needs to be extracted depending on whether the propeller has been damaged
    if blade_damage_lst[0] == 0:
        column_names = ["blade_damage", "alpha_angle", "wind_speed", "rpm", "average_rpms",
                        "mean_wind_uncorrected_thrust",
                        "std_thrust", "mean_wind_uncorrected_torque", "std_torque", "mean_wind_correction_thrust",
                        "std_wind_correction_thrust", "mean_wind_correction_torque", "std_wind_correction_torque",
                        "mean_wind_corrected_thrust", "mean_wind_corrected_torque", "Matlab_T", "Matlab_N", "BET_T",
                        "BET_N"]
    else:
        column_names = ["blade_damage", "alpha_angle", "wind_speed", "rpm", "average_rpms",
                        "mean_wind_uncorrected_thrust",
                        "std_thrust", "mean_wind_uncorrected_torque", "std_torque", "mean_wind_correction_thrust",
                        "std_wind_correction_thrust", "mean_wind_correction_torque", "std_wind_correction_torque",
                        "mean_wind_corrected_thrust", "mean_wind_corrected_torque", "amplitude_wind_corrected_thrust",
                        "amplitude_wind_corrected_torque", "BET_T", "BET_N", "BET_amplitude_T",
                        "BET_amplitude_N"]

    # Iterate over all the lists
    for blade_damage_counter, blade_damage in enumerate(blade_damage_lst):
        for alpha_angle_counter, alpha_angle in enumerate(alpha_angle_lst):
            for wind_speed_counter, wind_speed in enumerate(wind_speed_lst):
                for rpm_counter, rpm in enumerate(rpm_lst):
                    data_row = [blade_damage, alpha_angle, wind_speed, rpm]

                    # Obtain the information from the validation wind tunnel experiments
                    # Obtain the wind uncorrected thrust and torque
                    figure_number, content, average_rpms, mean_wind_uncorrected_thrust, std_thrust, \
                    mean_wind_uncorrected_torque, std_torque = \
                        experimental_data_extraction(figure_number, blade_damage,
                                                     alpha_angle, wind_speed, rpm, folder_files,
                                                     switch_plot_experimental_validation=False,
                                                     switch_print=False)

                    data_row += [average_rpms, mean_wind_uncorrected_thrust, std_thrust, mean_wind_uncorrected_torque,
                                 std_torque]

                    # Obtain the wind correction
                    figure_number, mean_wind_correction_thrust, mean_wind_correction_torque, \
                    std_wind_correction_thrust, std_wind_correction_torque = \
                        obtain_wind_correction(figure_number, alpha_angle, wind_speed, folder_files_np,
                                               switch_wind_correction=switch_wind_correction,
                                               switch_plot_experimental_validation=
                                               switch_plot_experimental_validation,
                                               switch_print_info=False)

                    data_row += [mean_wind_correction_thrust, std_wind_correction_thrust,
                                 mean_wind_correction_torque, std_wind_correction_torque]

                    # Obtain the data from the Matlab and the BET models
                    figure_number, F_healthy_lst, M_healthy_lst, Matlab_T, Matlab_N = \
                        compute_BET_signals(figure_number, blade_damage, alpha_angle, wind_speed, average_rpms,
                                            dt, switch_plot_models)
                    BET_thrust = -F_healthy_lst[-1, :]
                    BET_torque = M_healthy_lst[-1, :]

                    # Apply the wind correction
                    if blade_damage == 0:
                        mean_wind_corrected_thrust = mean_wind_uncorrected_thrust - mean_wind_correction_thrust
                        mean_wind_corrected_torque = mean_wind_uncorrected_torque - mean_wind_correction_torque

                        BET_T = np.mean(BET_thrust)
                        BET_N = np.mean(BET_torque)

                        data_row += [mean_wind_corrected_thrust, mean_wind_corrected_torque,
                                     Matlab_T, Matlab_N, BET_T, BET_N]
                    else:
                        # In the case that there is blade damage
                        # Obtaining the mean and amplitude of the thrust
                        figure_number, BET_mean_T, BET_amplitude_T, mean_wind_corrected_thrust, \
                        amplitude_wind_corrected_thrust = \
                            damaged_prop_signal_id(figure_number, content, "T", mean_wind_correction_thrust, BET_thrust,
                                                   switch_plot_fft=switch_plot_fft,
                                                   switch_plot_sinusoid_id=switch_plot_sinusoid_id, id_type=id_type)

                        # Obtaining the mean and amplitude of the torque
                        figure_number, BET_mean_N, BET_amplitude_N, mean_wind_corrected_torque, \
                        amplitude_wind_corrected_torque = \
                            damaged_prop_signal_id(figure_number, content, "N", mean_wind_correction_torque, BET_torque,
                                                   switch_plot_fft=switch_plot_fft,
                                                   switch_plot_sinusoid_id=switch_plot_sinusoid_id, id_type=id_type)

                        data_row += [mean_wind_corrected_thrust, mean_wind_corrected_torque,
                                     amplitude_wind_corrected_thrust, amplitude_wind_corrected_torque,
                                     BET_mean_T, BET_mean_N,
                                     BET_amplitude_T, BET_amplitude_N]

                    # Print the thrust and torque data means with the wind correction
                    print("\n Wind corrected thrust and torque")
                    print(f"The thrust mean: {mean_wind_corrected_thrust}")
                    print("------------------------------------------------")
                    print(f"The torque mean: {mean_wind_corrected_torque}")

                    data_table.append(data_row)

    # Store all the information in a dataframe and save in memory
    df = pd.DataFrame(data=data_table, columns=column_names)
    df.to_csv(os.path.join("Data_storage", f"{data_file_name}.csv"), index=False)
    return figure_number
