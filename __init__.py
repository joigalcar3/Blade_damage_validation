#!/usr/bin/env python3
"""
Provides the code for extracting the required information and analysing it. There are different inputs depending
on the damage and the wind speed.
"""

from data_extraction import data_extraction
from data_analysis import data_analysis

# User input
b = 0
w = 0
user_choice = False
switch_data_extraction = False
switch_data_analysis = True
figure_number = 1


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
                                      switch_plot_rpms, data_file_name, filename)

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
                                          switch_plot_rpms, data_file_name, filename)

# ----------------------------------------------------------------------------------------------------------------------
# Code for the extraction of all the data with the parameters chosen by the user
elif user_choice:
    if switch_data_extraction:
        blade_damage_lst = [0]  # 0, 10, 25
        alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]  # 0, 15, 30, 45, 60, 75, 90
        wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
        rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
        switch_plot_experimental_validation = False
        switch_plot_models = False
        switch_wind_correction = True
        data_file_name = f"b{blade_damage_lst[0]}"
        # data_file_name = f"b0_a{alpha_angle_lst[0]}_w0"

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
        data_file_name = "b0_rpms"
        filename = "b0.csv"

        # data_file_name = f"b0_a15_w0_rpms"
        # filename = f"b0_a15_w0.csv"

        figure_number = data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles,
                                      switch_plot_rpms, data_file_name, filename)

