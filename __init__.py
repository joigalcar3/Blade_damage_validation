#!/usr/bin/env python3
"""
Provides the code for data extraction and analysis. There are different pre-defined inputs
depending on the damage and the wind speed.
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
from compute_rpm_data_statistics import compute_rpm_data_statistics
from plot_statistics import *
from user_inputs import *
from data_extraction import data_extraction


# Initial set-up
figure_number = 1
relative_comment = ""
if switch_error_percentage:
    relative_comment += "_relative"

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the wind is not equal to zero
if b == 0 and w != 0 and not user_choice:
    # Data extraction
    blade_damage_lst = [b]  # 0, 10, 25
    wind_speed_lst = [2, 4, 6, 9, 12]  # (0), 2, 4, 6, 9, 12
    switch_wind_correction = True  # whether the data should be corrected with the wind corrections
    data_file_name = f"b{blade_damage_lst[0]}"
    filename_input_data = f"b{b}"  # name of the input file of raw data
    filename_input_stat = f"b{b}_rpms"  # name of the input file of statistics
    filename_input_data_multiple_wind = [f"b{b}_a90_w0", filename_input_data]  # list of files provided as input
    filenames = [f"b{b}" for b in blade_damage_compare]

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the velocity is zero
elif b == 0 and w == 0 and not user_choice:
    alpha_angle_lst = [90]  # 0, 15, 30, 45, 60, 75, 90
    blade_damage_lst = [0]  # 0, 10, 25
    wind_speed_lst = [0]  # (0), 2, 4, 6, 9, 12
    switch_wind_correction = False
    data_file_name = f"b{b}_a{alpha_angle_lst[0]}_w0"
    filename_input_data = f"b{b}_a90_w0"
    filename_input_stat = f"b{b}_a90_w0_rpms"
    filename_input_data_multiple_wind = [filename_input_data, f"b{b}"]  # list of files provided as input
    filenames = [f"b{b}_a90_w0" for b in blade_damage_compare]

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is damaged and the wind is not equal to zero
if b != 0 and w != 0 and not user_choice:
    # Data extraction
    blade_damage_lst = [b]  # 0, 10, 25
    wind_speed_lst = [2, 4, 6, 9, 12]  # (0), 2, 4, 6, 9, 12
    switch_wind_correction = True
    data_file_name = f"b{blade_damage_lst[0]}"
    filename_input_data = f"b{b}"  # name of the input file of raw data
    filename_input_stat = f"b{b}_rpms"  # name of the input file of statistics
    filename_input_data_multiple_wind = [f"b{b}_a90_w0", filename_input_data]  # list of files provided as input
    filenames = [f"b{b}" for b in blade_damage_compare]

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is damaged and the velocity is zero
elif b != 0 and w == 0 and not user_choice:
    alpha_angle_lst = [90]  # 0, 15, 30, 45, 60, 75, 90
    blade_damage_lst = [b]  # 0, 10, 25
    wind_speed_lst = [0]  # (0), 2, 4, 6, 9, 12
    switch_wind_correction = False
    data_file_name = f"b{b}_a{alpha_angle_lst[0]}_w0"
    filename_input_data = f"b{b}_a90_w0"
    filename_input_stat = f"b{b}_a90_w0_rpms"
    filename_input_data_multiple_wind = [filename_input_data, f"b{b}"]  # list of files provided as input
    filenames = [f"b{b}_a90_w0" for b in blade_damage_compare]

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data with the parameters chosen by the user
elif user_choice:
    blade_damage_lst = [0]  # 0, 10, 25
    alpha_angle_lst = [0]  # 0, 15, 30, 45, 60, 75, 90
    wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
    switch_wind_correction = True
    # data_file_name = f"b{blade_damage_lst[0]}"
    data_file_name = f"b0_a{alpha_angle_lst[0]}_w0"
    filename_input_data = f"b{b}"  # name of the input file of raw data
    # filename_input_data = "b25_a90_w0"
    filename_input_stat = f"b{b}_rpms"  # name of the input file of statistics
    # filename_input_stat = "b25_a90_w0_rpms"
    filename_input_data_multiple_wind = [f"b{b}_a90_w0", filename_input_data]  # list of files provided as input
    filenames = [f"b{b}" for b in blade_damage_compare]
    # filenames = [f"b{b}_a90_w0" for b in blade_damage_compare]

else:
    raise ValueError(f"The user choice:{user_choice}, is invalid.")

# Data extraction
if switch_data_extraction:
    figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                    switch_plot_experimental_validation, switch_plot_models, switch_wind_correction,
                                    data_file_name, switch_plot_fft=switch_plot_fft,
                                    switch_plot_sinusoid_id=switch_plot_sinusoid_id, id_type=id_type)

# Data statistics
if switch_data_statistics:
    compute_rpm_data_statistics(data_file_name, blade_damage=b)

# Data analysis and plotting
if switch_data_analysis:
    if plot_single_windspeed:
        if plot_single_damage:
            plot_statistics(filename_input_stat, filename_input_data, b, wind_speed_lst, rpm_lst,
                            switch_plot_alpha_angles=switch_plot_alpha_angles, switch_amplitude=switch_amplitude,
                            switch_val_error_bars=switch_val_error_bars,
                            switch_error_percentage=switch_error_percentage, comment=comment)
        else:
            data_damage_comparison(figure_number, wind_speed_lst, rpm_lst, filenames,
                                   switch_error_percentage=switch_error_percentage,
                                   switch_val_error_bars=switch_val_error_bars,
                                   switch_plot_alpha_angles=switch_plot_alpha_angles,
                                   switch_subtract_no_damage=switch_subtract_no_damage,
                                   switch_plot_stds=switch_plot_stds)
    else:
        figure_number = plot_rpms_windspeeds(figure_number, filename_input_data_multiple_wind, b, model,
                                             wind_speed_lst, comment, switch_error_percentage, switch_amplitude)
