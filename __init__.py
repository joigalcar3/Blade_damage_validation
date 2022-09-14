#!/usr/bin/env python3
"""
Provides the code for extracting the required information and analysing it. There are different inputs depending
on the damage and the wind speed.
"""

from data_extraction import data_extraction
from data_analysis import data_analysis, data_damage_comparison

# User input
b = 25  # blade damage
w = 0  # wind speed
switch_error_percentage = False  # whether the error should be relative (True) or absolute (False)
user_choice = False  # whether the user choice should be run instead of the predetermined configurations
switch_data_extraction = True  # whether the data extraction from the pre-processed data should be carried out
switch_data_analysis = False  # whether the data should be analysed to create the plots
switch_val_error_bars = True  # whether the validation error bars should be plotted
switch_plot_fft = False  # whether to plot the fft of the BET signal for sinusoid identification
switch_plot_sinusoid_id = False  # whether to plot the identified sinusoid

switch_blade_damage_comparison = False  # whether the results with different blade damages should be plotted
blade_damage_compare = [0, 10, 25]  # the blade damages to compare

# Initial set-up and computations
figure_number = 1
relative_comment = ""
if switch_error_percentage:
    relative_comment += "_relative"

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the wind is not equal to zero
if b == 0 and w != 0 and not user_choice:
    # Data extraction
    if switch_data_extraction:
        blade_damage_lst = [b]  # 0, 10, 25
        alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]   # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2, 4, 6, 9, 12]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        data_file_name = f"b{blade_damage_lst[0]}"

        figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                        switch_plot_experimental_validation, switch_plot_models, switch_wind_correction,
                                        data_file_name, switch_plot_fft=switch_plot_fft,
                                        switch_plot_sinusoid_id=switch_plot_sinusoid_id)

    # Data analysis
    if switch_data_analysis:
        blade_damage = b
        wind_speed = w
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        filename_output = f"b0_w{wind_speed}_rpms"
        filename_input = f"b0"

        figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                      switch_plot_rpms, filename_input, filename_output,
                                      switch_error_percentage=switch_error_percentage, comment=relative_comment,
                                      switch_val_error_bars=switch_val_error_bars)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the velocity is zero
elif b == 0 and w == 0 and not user_choice:
    alpha_angle_lst_all = [90]  # 0, 15, 30, 45, 60, 75, 90
    if switch_data_extraction:
        blade_damage_lst = [0]  # 0, 10, 25
        wind_speed_lst = [0]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = False

        for angle in alpha_angle_lst_all:
            alpha_angle_lst = [angle]
            data_file_name = f"b0_a{alpha_angle_lst[0]}_w0"
            figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                            switch_plot_experimental_validation, switch_plot_models,
                                            switch_wind_correction, data_file_name, switch_plot_fft=switch_plot_fft,
                                            switch_plot_sinusoid_id=switch_plot_sinusoid_id)

    # Data analysis
    if switch_data_analysis:
        blade_damage = 0
        wind_speed = 0
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        for angle in alpha_angle_lst_all:
            filename_output = f"b0_w0_rpms"
            filename_input = f"b0"
            figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                          switch_plot_rpms, filename_input, filename_output,
                                          switch_error_percentage=switch_error_percentage, comment=relative_comment,
                                          switch_val_error_bars=switch_val_error_bars)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is damaged and the wind is not equal to zero
if b != 0 and w != 0 and not user_choice:
    # Data extraction
    if switch_data_extraction:
        blade_damage_lst = [b]  # 0, 10, 25
        alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]   # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2, 4, 6, 9, 12]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        data_file_name = f"b{blade_damage_lst[0]}"

        figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                        switch_plot_experimental_validation, switch_plot_models, switch_wind_correction,
                                        data_file_name, switch_plot_fft=switch_plot_fft,
                                        switch_plot_sinusoid_id=switch_plot_sinusoid_id)

    # Data analysis
    if switch_data_analysis:
        blade_damage = b
        wind_speed = w
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        filename_output = f"b{b}_w{wind_speed}_rpms"
        filename_input = f"b{b}_w{wind_speed}"

        if switch_blade_damage_comparison:
            filenames = [f"b{b}.csv" for b in blade_damage_compare]
            data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                                   filenames, switch_error_percentage=switch_error_percentage, comment=relative_comment)
        else:
            # Analysis of the mean values
            figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                          switch_plot_rpms, filename_input, filename_output,
                                          switch_error_percentage=switch_error_percentage, comment=relative_comment,
                                          switch_val_error_bars=switch_val_error_bars)

            # Analysis of the oscillation amplitudes
            comment = "_amp" + relative_comment
            figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                          switch_plot_rpms, filename_input, filename_output,
                                          switch_error_percentage=switch_error_percentage, comment=comment,
                                          switch_val_error_bars=switch_val_error_bars, switch_amplitude=True)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is damaged and the velocity is zero
elif b != 0 and w == 0 and not user_choice:
    alpha_angle_lst_all = [90]  # 0, 15, 30, 45, 60, 75, 90
    if switch_data_extraction:
        blade_damage_lst = [b]  # 0, 10, 25
        wind_speed_lst = [0]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = False

        for angle in alpha_angle_lst_all:
            alpha_angle_lst = [angle]
            data_file_name = f"b{b}_a{alpha_angle_lst[0]}_w0"
            figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                            switch_plot_experimental_validation, switch_plot_models,
                                            switch_wind_correction, data_file_name, switch_plot_fft=switch_plot_fft,
                                            switch_plot_sinusoid_id=switch_plot_sinusoid_id)

    # Data analysis
    if switch_data_analysis:
        blade_damage = b
        wind_speed = 0
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        for angle in alpha_angle_lst_all:
            if switch_blade_damage_comparison:
                filenames = [f"b{b}_a{angle}_w0.csv" for b in blade_damage_compare]
                data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                                       filenames, switch_error_percentage=switch_error_percentage,
                                       comment=relative_comment)
            else:
                filename_output = f"b{blade_damage}_w0_rpms"
                filename_input = f"b{blade_damage}_w0"
                figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                              switch_plot_rpms, filename_input, filename_output,
                                              switch_error_percentage=switch_error_percentage, comment=relative_comment,
                                              switch_val_error_bars=switch_val_error_bars)

                # Analysis of the oscillation amplitudes
                comment = "_amp" + relative_comment
                figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                              switch_plot_rpms, filename_input, filename_output,
                                              switch_error_percentage=switch_error_percentage, comment=comment,
                                              switch_val_error_bars=switch_val_error_bars, switch_amplitude=True)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data with the parameters chosen by the user
elif user_choice:
    if switch_data_extraction:
        blade_damage_lst = [0]  # 0, 10, 25
        alpha_angle_lst = [0]  # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        # data_file_name = f"b{blade_damage_lst[0]}"
        # data_file_name = f"b0_a{alpha_angle_lst[0]}_w0"
        data_file_name = "dummy2"

        figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                                        switch_plot_experimental_validation, switch_plot_models, switch_wind_correction,
                                        data_file_name, switch_plot_fft=switch_plot_fft,
                                        switch_plot_sinusoid_id=switch_plot_sinusoid_id)

    # Data analysis
    if switch_data_analysis:
        blade_damage = 0
        wind_speed = 2
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        switch_error_percentage = True
        filename_output = "b0_rpms"
        filename_input = "b0"
        comment = "dummy_"

        # data_file_name = f"b0_a15_w0_rpms"
        # filename = f"b0_a15_w0.csv"

        figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                      switch_plot_rpms, filename_input, filename_output, comment=comment,
                                      switch_error_percentage=switch_error_percentage,
                                      switch_val_error_bars=switch_val_error_bars)

