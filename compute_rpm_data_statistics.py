#!/usr/bin/env python3
"""
Computes the statistics for each rpm. This is done by compressing the information from all the angles with respect
to the flow for a single propeller rotation velocity. The main statistics are the mean and standard deviation of the
absolute and the relative error when comparing the experimental data obtained for each angle with that obtained from
the models.
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
from scipy.stats import norm
mpl.use('TkAgg')

# Create all possible data combinations
abbreviations = ["T", "N"]
wrench_names = {"T": "thrust", "N": "torque"}
models = ["Matlab", "BET"]
data_points = ["mu", "std", "m", "b", "tol"]
modifications = ["", "_rel", "_amp", "_amp_rel"]
column_names = ["rpm", "blade_damage", "wind_speed"]
for model in models:
    for modification in modifications:
        for abbreviation in abbreviations:
            for data_point in data_points:
                if model == "Matlab" and "amp" in modification:
                    continue
                column_names.append(f"{model}_{data_point}_{abbreviation}{modification}")


def compute_rpm_data_statistics(filename_input, blade_damage=None, comment="", filename_output=None):
    """
    File to compute the statistics for each rpm using the data for the different propeller incidence angles. It creates
    the data required to create, e.g., the plots T error vs rpms.
    The blade damage is constant and this is done for every rpm with constant wind speeds.
    :param filename_input: the name of the file that is fed as input
    :param blade_damage: the percentage of blade damage
    :param comment: whether the name of the output file should have any comment
    :param filename_output: the name of the output file
    :return: None
    """
    # Extraction of the processed data
    df = pd.read_csv(os.path.join("Data_storage", filename_input + ".csv"))
    rpm_lst = np.unique(df["rpm"])
    wind_speed_lst = np.unique(df["wind_speed"])

    if blade_damage is None:
        blade_damage = int(filename_input[1:])

    # Iterate over the rpms and wind speeds
    data_table = []
    for wind_speed in wind_speed_lst:
        for rpm in rpm_lst:
            counter = 3
            data_row = np.zeros(len(column_names))
            data_row[:counter] = [rpm, blade_damage, wind_speed]

            # Retrieving the data for the specific scenario
            data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) & (df["rpm"] == rpm)]

            # Obtaining the thrust plots
            # Experimental data
            for column_name in column_names:
                if "mu" not in column_name:
                    continue

                # Remove impossible combinations
                if ("amp" in column_name and blade_damage == 0) or (blade_damage != 0 and "Matlab" in column_name):
                    data_row[counter:counter + len(data_points)] = np.ones(len(data_points)) * -1
                    counter += len(data_points)
                    continue

                # Check for amplitude in the name
                if "amp" in column_name:
                    mean_or_amplitude_val = "amplitude"
                    mean_or_amplitude_mod = "_amplitude"
                else:
                    mean_or_amplitude_val = "mean"
                    mean_or_amplitude_mod = ""

                # Check for the wrench type
                if abbreviations[1] in column_name:
                    abbreviation = abbreviations[1]
                else:
                    abbreviation = abbreviations[0]

                # Check for the model type
                if models[0] in column_name:
                    model_name = models[0]
                else:
                    model_name = models[1]

                # Obtain a row of the dataframe we are building
                wrench_name = wrench_names[abbreviation]
                validation_data = data_rpm[f"{mean_or_amplitude_val}_wind_corrected_{wrench_name}"].to_numpy()
                model_data = data_rpm[f"{model_name}{mean_or_amplitude_mod}_{abbreviation}"].to_numpy()
                model_stats = compute_signal_stats(column_name, model_data, validation_data)
                data_row[counter:counter+len(data_points)] = model_stats
                counter += len(data_points)
            data_table.append(data_row)

    # Create output file name
    if filename_output is None:
        filename_output = f"b{blade_damage}{comment}_rpms.csv"

    # Create dataframe and store it
    df_rpms = pd.DataFrame(data=data_table, columns=column_names)
    df_rpms.to_csv(os.path.join("Data_storage", filename_output), index=False)


def compute_signal_stats(column_name, model_data, validation_data):
    """
    Given a signal data, its statistical values are computed:
        - The data points mean values (mu)
        - The data points standard deviation (std)
        - The data points slope of the fitted line (mu)
        - The data points fitted line intersection with the y-axis (b)
        - The error range where all the errors can be found (tol), important for plotting
    :param column_name: the name of the column where the data comes from
    :param model_data: the data obtained from the model
    :param validation_data: the data from the experiment
    :return: statistical information
    """
    # Computation of the models line approximations
    model_m, model_b = 0, 0
    if validation_data.shape[0] > 1:
        model_m, model_b = np.polyfit(validation_data, model_data, 1)

    # Computation of the probability statistics of the errors
    if "rel" in column_name:
        model_error = np.divide(validation_data - model_data,
                                np.maximum(np.abs(validation_data), 1e-10)) * 100
    else:
        model_error = validation_data - model_data

    model_error_range = max(model_error) - min(model_error)
    model_tolerance = abs(model_error_range) * 0.2
    model_mu, model_std = norm.fit(model_error)
    model_stats = [model_mu, model_std, model_m, model_b, model_tolerance]

    return model_stats


if __name__ == "__main__":
    fi = "b25_a90_w0"
    b = 25
    comm = "_a90_w0"
    compute_rpm_data_statistics(fi, blade_damage=b, comment=comm)
