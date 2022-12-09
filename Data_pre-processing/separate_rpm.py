#!/usr/bin/env python3
"""
File to divide a single file into multiple ones with different rpm values.

A single file is usually divided into 5 files, one for each of the following rpm values: 300, 500, 700, 900 and 1100 rpm
"""

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"

import pandas as pd
import scipy.signal as ss
import numpy as np
from sklearn.neighbors import KernelDensity
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(1)

# Matplotlib settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)


def separate_rpm(filename, figure_number, original_folder, destination_folder, plotting=False):
    """
    Divides the data within a single file into multiple files with different values of propeller rotational speed.
    Normally, one file would be dividided into 5 files, each corresponding to rotational speed values of 300, 500, 700,
    900 and 1100 rpms.
    :param filename: name of the file
    :param figure_number: the figure number to be used for the next figure generation
    :param original_folder: the directory where the input file is located
    :param destination_folder: the directory where the output file should be located
    :param plotting: whether the ESC and rpm division should be plotted (used for the paper's figures)
    :return:
    """
    print(f"Separating rpms of: {filename}")
    # Obtain information data
    blade_damage_begin = filename.index("b") + 1
    blade_damage_end = filename.index("_a")
    alpha_angle_end = filename.index("_w")
    wind_speed_end = filename.index(".")

    blade_damage = int(filename[blade_damage_begin:blade_damage_end])
    alpha_angle = int(filename[blade_damage_end+2:alpha_angle_end])
    wind_speed = int(filename[alpha_angle_end+2:wind_speed_end])

    # Retrieval of data
    file_path = os.path.join(original_folder, filename)
    content = pd.read_csv(file_path, skiprows=1)

    # Filtering by ESC information
    # Obtaining the most common ESC values
    escs = content["ESC signal (µs)"]
    escs_counts = np.bincount(escs)
    escs_count_max = escs_counts[escs_counts > 1300]
    if len(escs_count_max) != 5:
        raise AttributeError(f"Only {len(escs_count_max)}/5 rpm values were found ")
    escs_common = np.where(escs_counts > 1300)[0]

    # Extracting the uninterrupted constant ESC intervals
    if plotting:
        fig = plt.figure(figure_number)
        figure_number += 1
        plt.plot(content['Time (s)'], escs, alpha=0.5, linewidth=4)
    constant_rpm_intervals = []
    constant_rpm_timestamps = []
    for esc in escs_common:
        esc_indeces = np.where(escs == esc)[0]
        left_index = esc_indeces[0]
        n_corrections_required = esc_indeces[-1] + 1 - len(esc_indeces)
        if n_corrections_required:
            current_corrections = 0
            for i in range(1, len(esc_indeces)):
                if esc_indeces[i] != esc_indeces[i - 1] + 1:
                    left_index = esc_indeces[i]
                    current_corrections += 1
                    if current_corrections == n_corrections_required:
                        break
        esc_section = escs[left_index:esc_indeces[-1] + 1]
        constant_rpm_intervals.append(esc_section)
        constant_rpm_timestamps.append(np.arange(left_index, esc_indeces[-1] + 1))
        if plotting:
            plt.plot(content['Time (s)'][np.arange(left_index, esc_indeces[-1] + 1)], esc_section, linestyle='--', label=f"{esc}",
                     linewidth=4)

    # Plotting
    rpms = [300, 500, 700, 900, 1100]
    if plotting:
        # Plotting the intervals in the ESC domain
        # plt.title("Selected timestamps for analysis")
        plt.ylabel("ESC value [µs]")
        plt.xlabel("Time [s]")
        plt.grid(True)
        plt.legend()
        fig.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
        fig.set_size_inches(19.24, 10.55)

        # Plotting the intervals in the motor rotations domain
        fig = plt.figure(figure_number)
        figure_number += 1
        plt.plot(content['Time (s)'], content["Motor Electrical Speed (rad/s)"], alpha=0.5, linewidth=4)
        for counter, rpm_interval in enumerate(constant_rpm_timestamps):
            plt.plot(content['Time (s)'][rpm_interval], content["Motor Electrical Speed (rad/s)"][rpm_interval], linestyle='--', label=str(rpms[counter]),
                     linewidth=4)
        # plt.title("Selected rpm intervals for analysis")
        plt.ylabel("Motor Electrical Speed [rad/s]")
        plt.xlabel("Time [s]")
        plt.grid(True)
        plt.legend()
        fig.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
        fig.set_size_inches(19.24, 10.55)

    # Saved rpms in different files
    for count, rpm_interval in enumerate(constant_rpm_timestamps):
        data_interval = content[rpm_interval[0]:rpm_interval[-1] + 1]
        rpm = rpms[count]
        destination_filename = f"b{blade_damage}_a{alpha_angle}_w{wind_speed}_r{rpm}.csv"
        filename = os.path.join(destination_folder, destination_filename)
        if os.path.exists(filename):
            raise ValueError(f"{filename} already exists")
        data_interval.to_csv(filename)
    return figure_number


if __name__ == "__main__":
    # User input
    of = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign\\" \
         "Data\\Test_files_correct_names"
    df = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign\\" \
         "Data\\Test_files_correct_names_separated"
    a = 90
    w = 12
    b = 25
    switch_plotting = True
    switch_multiple_file = False

    # Retrieval of data
    file = f"b{b}_a{a}_w{w}.csv"
    fn = 1

    if switch_multiple_file:
        files = os.listdir(of)
        for file in files:
            fn = separate_rpm(file, fn, of, df, plotting=switch_plotting)
    else:
        fn = separate_rpm(file, fn, of, df, plotting=switch_plotting)
