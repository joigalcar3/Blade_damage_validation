#!/usr/bin/env python3
"""
Provides the code for extracting the required information and analysing it. There are different inputs depending
on the damage and the wind speed.
"""

from data_extraction import data_extraction
from data_analysis import data_analysis, data_damage_comparison

# User input
b = 25
w = 2
switch_error_percentage = True
user_choice = False
switch_data_extraction = True
switch_data_analysis = False

switch_blade_damage_comparison = False
blade_damage_compare = [0, 25]

figure_number = 1
relative_comment = ""
if switch_error_percentage:
    relative_comment = "relative_"


# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the wind is not equal to zero
if b == 0 and w == 2 and not user_choice:
    # Data extraction
    if switch_data_extraction:
        blade_damage_lst = [0]  # 0, 10, 25
        alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]   # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        data_file_name = f"b{blade_damage_lst[0]}"

        figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                        switch_plot_experimental_validation, switch_plot_models, switch_wind_correction, data_file_name)

    # Data analysis
    if switch_data_analysis:
        blade_damage = 0
        wind_speed = 2
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        data_file_name = "b0_rpms"
        filename = "b0.csv"

        figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                      switch_plot_rpms, data_file_name, filename,
                                      switch_error_percentage=switch_error_percentage)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is not damaged and the velocity is zero
elif b == 0 and w == 0 and not user_choice:
    alpha_angle_lst_all = [0, 15, 30, 45, 60, 75, 90]  # 0, 15, 30, 45, 60, 75, 90
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
                            switch_plot_experimental_validation, switch_plot_models, switch_wind_correction,
                            data_file_name)

    # Data analysis
    if switch_data_analysis:
        blade_damage = 0
        wind_speed = 0
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        for angle in alpha_angle_lst_all:
            data_file_name = f"b0_a{angle}_w0_rpms"
            filename = f"b0_a{angle}_w0.csv"
            figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                          switch_plot_rpms, data_file_name, filename,
                                          switch_error_percentage=switch_error_percentage)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data when the drone is damaged and the wind is not equal to zero
if b != 0 and w == 2 and not user_choice:
    # Data extraction
    if switch_data_extraction:
        blade_damage_lst = [b]  # 0, 10, 25
        alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]   # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        data_file_name = f"b{blade_damage_lst[0]}"

        figure_number = data_extraction(figure_number, blade_damage_lst, alpha_angle_lst, wind_speed_lst, rpm_lst,
                        switch_plot_experimental_validation, switch_plot_models, switch_wind_correction, data_file_name)

    # Data analysis
    if switch_data_analysis:
        blade_damage = b
        wind_speed = 2
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        data_file_name = f"b{b}_rpms"
        filename = f"b{b}.csv"
        comment = f"b{b}\\{relative_comment}"

        if switch_blade_damage_comparison:
            filenames = [f"b{b}.csv" for b in blade_damage_compare]
            comment = f"b{'_'.join(str(i) for i in blade_damage_compare)}\\{relative_comment}"
            data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                                   filenames, switch_error_percentage=False, comment=comment)
        else:
            figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                          switch_plot_rpms, data_file_name, filename,
                                          switch_error_percentage=switch_error_percentage, comment=comment)

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
                                        data_file_name)

    # Data analysis
    if switch_data_analysis:
        blade_damage = 0
        wind_speed = 2
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_alpha_angles = True
        switch_plot_rpms = True
        switch_error_percentage = True
        data_file_name = "b0_rpms"
        filename = "b0.csv"
        comment = "dummy_"

        # data_file_name = f"b0_a15_w0_rpms"
        # filename = f"b0_a15_w0.csv"

        figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                      switch_plot_rpms, data_file_name, filename, comment=comment,
                                      switch_error_percentage=switch_error_percentage)

