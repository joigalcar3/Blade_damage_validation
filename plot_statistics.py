#!/usr/bin/env python3
"""
Provides the functions to plot the processed experimental and model data.
"""

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


# Imports
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

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

# Creating naming and dictionaries
abbreviations = ["T", "N"]
wrench_names = {"T": "thrust", "N": "torque"}
wrench_units = {"T": "N", "N": "Nm"}
data_points = ["mu", "std", "m", "b", "tol"]
models = ["Matlab", "BET"]

# Setting up color scheme
red_color_hex, red_color_rgb = "#d62728", [i / 255 for i in [214, 39, 40]]
blue_color_hex, blue_color_rgb = "#1f77b4", [i / 255 for i in [31, 119, 180]]
green_color_hex, green_color_rgb = "#2ca02c", [i / 255 for i in [44, 160, 44]]
orange_color_hex, orange_color_rgb = '#ff7f0e', [i / 255 for i in [255, 127, 14]]
purple_color_hex, purple_color_rgb = '#9467bd', [i / 255 for i in [148, 103, 189]]
brown_color_hex, brown_color_rgb = '#8c564b', [i / 255 for i in [140, 86, 75]]
pink_color_hex, pink_color_rgb = '#e377c2', [i / 255 for i in [227, 119, 194]]
colors_hex = [red_color_hex, blue_color_hex, green_color_hex, orange_color_hex, purple_color_hex, brown_color_hex,
              pink_color_hex]
markers = ["o", "v", "s", "P", "*", "x", "<"]


def plot_statistics(filename_input_stat, filename_input_data, blade_damage, wind_speed_lst, rpm_lst,
                    switch_plot_alpha_angles=True, switch_amplitude=False, switch_val_error_bars=True,
                    switch_error_percentage=False, comment=""):
    """
    Method that creates all the plots that compares angle of attack of the propeller rotational plane and rpms. Hence
    the blade damage and the wind speed is constant.
    :param filename_input_stat: the name of the csv file containing the pre-computed rpm stats
    :param filename_input_data: the name of the csv file containing the experimental and model data
    :param blade_damage: the percentage of blade damage
    :param wind_speed_lst: the list of wind speeds considered
    :param rpm_lst: the list of rpm considered
    :param switch_plot_alpha_angles: whether the angles of attack should be plotted with different transparency values
    for differentiation
    :param switch_amplitude: whether the amplitude of the sinusoid should be plotted instead of the bias
    :param switch_val_error_bars: whether the experimental 2 x sigma whiskers should be plotted
    :param switch_error_percentage: whether the relative error should be plotted instead of the absolute error
    :param comment: whether any text should be added at the end of the file name
    :return:
    """
    # Obtain data and its statistics
    df_stat = pd.read_csv(os.path.join("Data_storage", filename_input_stat + ".csv"))
    df_data = pd.read_csv(os.path.join("Data_storage", filename_input_data + ".csv"))
    active_models = models
    if blade_damage != 0:
        active_models = list(set(models) - {"Matlab"})

    # Define the figure number
    figure_number = 1

    # Whether the error computed is absolute or relative and the mean or amplitude are plotted
    dict_keys_txt, mean_or_amplitude, comment = check_amp_rel(switch_error_percentage, switch_amplitude, comment)

    # Plot for every wind speed and rpm in their respective lists
    for wind_speed in wind_speed_lst:
        for rpm in rpm_lst:
            # Retrieving the data for the specific scenario
            data_rpm_stat = df_stat[(df_stat["blade_damage"] == blade_damage) & (df_stat["wind_speed"] == wind_speed) &
                                    (df_stat["rpm"] == rpm)]
            data_rpm_data = df_data[(df_data["blade_damage"] == blade_damage) & (df_data["wind_speed"] == wind_speed) &
                                    (df_data["rpm"] == rpm)]

            # In the case that alpha_angle distinction is required, the used angles are extracted
            if switch_plot_alpha_angles:
                alpha_angle = data_rpm_data["alpha_angle"].to_numpy()
            else:
                alpha_angle = None

            # Loop for every wrench form
            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                wrench_name = wrench_names[abbreviation]

                # Models data
                models_stats = {}
                for model in active_models:
                    model_stats, model_name = create_model_stats(model, data_rpm_data, data_rpm_stat, mean_or_amplitude,
                                                                 abbreviation, wrench_name, switch_val_error_bars,
                                                                 switch_amplitude, dict_keys_txt)
                    models_stats[f"{model_name} model"] = model_stats

                # Location and name of the plots
                folder_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}")
                plot_name = os.path.join(folder_name, f"{wrench_name}_b{blade_damage}_w{wind_speed}_r{rpm}{comment}")
                if not os.path.exists(os.path.join("Plots_storage", folder_name)):
                    os.makedirs(os.path.join("Plots_storage", folder_name))

                # Produce the plots
                figure_number = data_pd_plotter(models_stats, figure_number, plot_name=plot_name,
                                                data_type=abbreviation[-1], alpha_angle=alpha_angle,
                                                switch_error_percentage=switch_error_percentage)

        # Plot the rpm comparison
        # Extract the models stats
        models_stats = {}
        data_stat = df_stat[(df_stat["blade_damage"] == blade_damage) & (df_stat["wind_speed"] == wind_speed)]
        data_data = df_data[(df_data["blade_damage"] == blade_damage) & (df_data["wind_speed"] == wind_speed)]
        for model in active_models:
            model_name = model
            if "Matlab" in model:
                model_name = "Gray-box"

            model_stats = {}
            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                rpms = data_stat["rpm"].to_numpy()
                mus = data_stat[f"{model}_mu_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                stds = data_stat[f"{model}_std_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                model_stats[abbreviation] = {"rpms": rpms, "mus": mus, "stds": stds, "blade_damage": blade_damage}
            models_stats[f"{model_name} model"] = model_stats

        if wind_speed != 0:
            plot_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}",
                                     f"error_b{blade_damage}_w{wind_speed}{comment}")
        else:
            angle_name = np.unique(data_data["alpha_angle"].to_numpy())[-1]
            plot_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}",
                                     f"error_b{blade_damage}_a{angle_name}_w{wind_speed}{comment}")
        figure_number = rpm_error_plotter(figure_number, models_stats, plot_name=plot_name,
                                          switch_error_percentage=switch_error_percentage)
    return figure_number


def rpm_error_plotter(figure_number, models_stats, plot_name="dummy2", switch_error_percentage=False,
                      ncol=1, switch_plot_stds=True, switch_subtract_no_damage=False):
    """
    Plot the evolution of the errors through different rpms with its mean and standard deviation
    :param figure_number: number of the figure to plot
    :param models_stats: the dictionary that contains the thrust and torque statistical information for plotting
    :param plot_name: the name of the plot that should be produced as output
    :param switch_error_percentage: whether the relative error is plotted instead of the absolute error
    :param ncol: number of columns in the legend when plotting
    :param switch_subtract_no_damage: whether the plotted data had the no damage data subtracted
    :param switch_plot_stds: whether the whiskers denoting the stds should be plotted
    :return:
    """
    # Create figure
    f, ax_lst = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.2}, num=figure_number)
    figure_number += 1
    unique_blade_damages = set([models_stats[i][j]["blade_damage"]
                                for i in models_stats.keys()
                                for j in models_stats[i].keys()])

    # Plot the errors
    for counter_ax, ax_name in enumerate(ax_lst):
        abbreviation = abbreviations[counter_ax]
        wrench_unit = wrench_units[abbreviation]
        for counter_model, model_name in enumerate(models_stats.keys()):
            wrench_data = models_stats[model_name][abbreviation]
            blade_damage = wrench_data["blade_damage"]
            rpms = wrench_data["rpms"]

            # Create label whether it is different damages or different models
            if "%" in model_name or len(unique_blade_damages) == 1:
                signal_label = model_name
            else:
                signal_label = f"{model_name}: {blade_damage}%"

            # Plot with and without whiskers
            if switch_plot_stds:
                ax_name.errorbar(rpms, wrench_data["mus"], yerr=wrench_data["stds"] * 1.96,
                                 color=colors_hex[counter_model], capsize=16, capthick=2, marker=markers[counter_model],
                                 markersize=10, alpha=0.5, label=signal_label)
            else:
                ax_name.plot(rpms, wrench_data["mus"], color=colors_hex[counter_model], marker=markers[counter_model],
                             markersize=20, alpha=0.5, label=signal_label)

        ax_name.grid(True)
        ax_name.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))

        # Include the y-label
        if switch_error_percentage:
            if switch_subtract_no_damage:
                ax_name.set_ylabel(f"{abbreviation} $\Delta$error [%]")
            else:
                ax_name.set_ylabel(f"{abbreviation} error [%]")
        else:
            if switch_subtract_no_damage:
                ax_name.set_ylabel(f"{abbreviation} $\Delta$error [{wrench_unit}]")
            else:
                ax_name.set_ylabel(f"{abbreviation} error [{wrench_unit}]")
        ax_name.yaxis.set_label_coords(-0.1, 0.5)

        # Include legend
        if counter_ax == 0:
            # ax_name.legend(markerscale=2, ncol=ncol, loc=1)   # position legend in the top right corner
            ax_name.legend(markerscale=2, ncol=ncol)      # dont provide lengend position
            # ax_name.legend(markerscale=2, ncol=ncol, loc="lower left")    # position the legend in the lower left

            # Create space on the upper space of the frame for the legend
            # ax_name.set_ylim((ax_name.get_ylim()[0], ax_name.get_ylim()[1] * 2))
        elif counter_ax == len(ax_lst) - 1:
            ax_name.set_xlabel("Propeller rotational speed [rad/s]")
            ax_name.set_xticks(rpms)

    # Set up image dimensions and save it
    f.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
    f.set_size_inches(19.24, 10.55)
    f.savefig(os.path.join("Plots_storage", f"{plot_name}.pdf"))
    return figure_number


def data_pd_plotter(models_stats, figure_number, plot_name="dummy", data_type="T", alpha_angle=None,
                    switch_error_percentage=False):
    """
    Plot the experimental and model data as well as the error in the form of probabilistic distribution
    :param models_stats: dictionary with the models data
    :param figure_number: number of the next figure to plot
    :param plot_name: name of the plot
    :param data_type: whether thrust or torque would be plotted
    :param alpha_angle: the alpha angles of the data
    :param switch_error_percentage: whether the error data should be relative instead of absolute
    :return:
    """
    # Computation of the constant line
    all_validation_data = np.concatenate([models_stats[key]["validation_data"] for key in models_stats.keys()])
    constant_x_min = np.min(all_validation_data) - 0.1 * abs(np.min(all_validation_data))
    constant_x_max = np.max(all_validation_data) + 0.1 * abs(np.max(all_validation_data))

    constant_line_coords = np.array([constant_x_min, constant_x_max])

    # Computation of the BET and model line approximations
    models_names = models_stats.keys()
    errors = np.zeros((len(models_names), len(models_stats[list(models_names)[0]]["validation_data"])))
    for counter, model_name in enumerate(models_names):
        # Retrieving model data
        model_data = models_stats[model_name]["model_data"]
        validation_data = models_stats[model_name]["validation_data"]

        # Computation of the models line approximations
        model_m, model_b = models_stats[model_name]["m"], models_stats[model_name]["b"]
        switch_plot_linear_fits = False
        if validation_data.shape[0] > 1:
            switch_plot_linear_fits = True

        model_y = model_m * constant_line_coords + model_b
        models_stats[model_name]["y"] = model_y

        # Computation of the probability statistics of the errors
        if switch_error_percentage:
            model_error = np.divide(validation_data - model_data,
                                    np.maximum(np.abs(validation_data), 1e-10)) * 100
        else:
            model_error = validation_data - model_data
        models_stats[model_name]["e"] = model_error
        errors[counter, :] = model_error

    # Computation of the x-axis range of the probability distribution
    maximum_tolerance = max([models_stats[key]["tol"] for key in models_stats.keys()])
    range_limit = max(abs(np.min(errors)), abs(np.max(errors))) + maximum_tolerance
    pdf_range = np.linspace(-range_limit, range_limit, 100)

    # Computation of the probability distributions of the errors
    for model_name in models_names:
        model_p = norm.pdf(pdf_range, models_stats[model_name]["mu"], models_stats[model_name]["std"])
        models_stats[model_name]["p"] = model_p

    # Plotting the data and the probability distributions together
    f_T, ax_T_lst = plt.subplots(2, 1, gridspec_kw={'wspace': 0.5, 'hspace': 0.55, 'height_ratios': [3, 1]},
                                 num=figure_number)
    figure_number += 1

    # Extracting the axes of the subplots
    ax_data = ax_T_lst[0]
    ax_pd = ax_T_lst[1]

    # Plotting the linear curve of 100% success with its 1.96 * std whiskers
    ax_data.plot(constant_line_coords, constant_line_coords, 'k--')
    for counter, model_name in enumerate(models_names):
        error_bar = models_stats[model_name]["error_bar"]
        if error_bar is not None:
            validation_data = models_stats[model_name]["validation_data"]
            ax_data.errorbar(validation_data, validation_data, yerr=error_bar * 1.96, linestyle="", color="k",
                             capsize=4)

    # In the case that the angles of the rotation plane with respect to the flow are provided, compute colors
    if alpha_angle is not None:
        red_color = np.asarray([red_color_rgb + [(a + 10) / 100] for a in alpha_angle])
        blue_color = np.asarray([blue_color_rgb + [(a + 10) / 100] for a in alpha_angle])
        green_color = np.asarray([green_color_rgb + [(a + 10) / 100] for a in alpha_angle])
    else:
        red_color = red_color_hex
        blue_color = blue_color_hex
        green_color = green_color_hex
    colors = [red_color, blue_color, green_color]

    for i, model_name in enumerate(models_stats.keys()):
        # Retrieve model data and probability distribution
        model_data = models_stats[model_name]["model_data"]
        model_error = models_stats[model_name]["e"]
        model_p = models_stats[model_name]["p"]
        validation_data = models_stats[model_name]["validation_data"]

        # Plot model data
        ax_data.scatter(validation_data, model_data, c=colors[i], marker=markers[i], label=model_name, s=100)
        if switch_plot_linear_fits:
            ax_data.plot(constant_line_coords, models_stats[model_name]["y"], color=colors_hex[i], linestyle="-")

        # Plot model probability distribution
        ax_pd.scatter(model_error, np.zeros(len(model_error)), c=colors[i], marker=markers[i], s=100)
        ax_pd.plot(pdf_range, model_p, color=colors_hex[i], linewidth=2, alpha=0.1)
        ax_pd.fill_between(pdf_range, model_p, color=colors_hex[i], alpha=0.1)

    # Polishing the data plot
    if data_type == "T":
        ax_data.set_xlabel("Experiments corrected thrust [N]")
        ax_data.set_ylabel("Model thrust [N]")
        if switch_error_percentage:
            ax_pd.set_xlabel("Thrust model relative error [%]")
        else:
            ax_pd.set_xlabel("Thrust model absolute error [N]")
    elif data_type == "N":
        ax_data.set_xlabel("Experiments corrected torque [Nm]")
        ax_data.set_ylabel("Model torque [Nm]")
        if switch_error_percentage:
            ax_pd.set_xlabel("Torque model relative error [%]")
        else:
            ax_pd.set_xlabel("Torque model absolute error [Nm]")
    ax_pd.set_yticks([])
    original_ylim = ax_data.get_ylim()
    ax_data.set_ylim(top=(original_ylim[-1] - original_ylim[0]) * 0.35 + original_ylim[-1])
    ax_data.legend(markerscale=2, scatterpoints=3, loc=2)
    ax_data.grid(True)
    ax_data.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax_data.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Polishing the probability distributions plot
    ax_pd.axvline(0, color="k", linestyle="--", alpha=0.2)

    ax_pd.set_ylim(bottom=ax_pd.get_ylim()[0] * 1.25)
    ax_pd.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    f_T.subplots_adjust(left=0.1, top=0.95, right=0.98, bottom=0.17)
    f_T.set_size_inches(19.24, 10.55)
    f_T.savefig(os.path.join("Plots_storage", f"{plot_name}.pdf"), bbox_inches='tight')

    return figure_number


def data_damage_comparison(figure_number, wind_speed_lst, rpm_lst, filenames, comment="", switch_plot_alpha_angles=True,
                           switch_error_percentage=False, switch_amplitude=False, switch_val_error_bars=True,
                           switch_subtract_no_damage=False, switch_plot_stds=False):
    """
    Method to create the same plots as in data analysis. However, instead of comparing Matlab and BET, it compares
    the performance of BET with different degrees of failure
    :param figure_number: the number of the next figure to plot
    :param wind_speed_lst: the list of speeds of the wind tunnel
    :param rpm_lst: the list of rpms to consider
    :param filenames: the names of the files that contain the data for the different blade damages
    :param comment: any comment to insert at the beginning of the plots file names
    :param switch_plot_alpha_angles: whether the angles of the plane of rotation with respect to the flow should be
    visible
    :param switch_error_percentage: whether the error should be plotted as relative instead of absolute
    :param switch_amplitude: whether the amplitude should be plotted instead of the mean
    :param switch_plot_rpms: to switch on a plot of the mean and standard error for the different rpms
    :param switch_plot_stds: whether the stds whickers are plotted
    :return:
    """
    # Obtaining the blade damages
    blade_damages = []
    for filename in filenames:
        if "w" in filename:
            blade_damage = int(filename[filename.index("b") + 1:filename.index("_")])
        else:
            blade_damage = int(filename[filename.index("b") + 1:])
        blade_damages.append(blade_damage)

    # Whether the error computed is absolute or relative and the mean or amplitude are plotted
    dict_keys_txt, mean_or_amplitude, comment = check_amp_rel(switch_error_percentage, switch_amplitude, comment)

    # Creating the folder where the plots will be stored
    blade_damages_str = "_".join(str(i) for i in blade_damages)
    for wind_speed in wind_speed_lst:
        folder_name = os.path.join(f"b{blade_damages_str}", f"w{wind_speed}")
        if not os.path.exists(os.path.join("Plots_storage", folder_name)):
            os.makedirs(os.path.join("Plots_storage", folder_name))

        # Looping over the desired rpms
        for rpm in rpm_lst:
            # Obtaining the thrust plots
            # Experimental data
            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                wrench_name = wrench_names[abbreviation]

                # Retrieving the data
                models_stats = {}
                for counter, filename in enumerate(filenames):
                    blade_damage = blade_damages[counter]

                    # Retrieving data files
                    filters = {"blade_damage": blade_damage, "wind_speed": wind_speed, "rpm": rpm}
                    data_rpm_data = filter_data(filename, filters)
                    data_rpm_stat = filter_data(filename + "_rpms", filters)

                    # Whether the different angles should be differentiated by transparency
                    if switch_plot_alpha_angles:
                        alpha_angle = data_rpm_data["alpha_angle"].to_numpy()
                    else:
                        alpha_angle = None

                    model_stats, _ = create_model_stats("BET", data_rpm_data, data_rpm_stat, mean_or_amplitude,
                                                        abbreviation, wrench_name, switch_val_error_bars, switch_amplitude,
                                                        dict_keys_txt)
                    models_stats[f"BET model: {blade_damage}%"] = model_stats

                # File name
                plot_name = os.path.join(folder_name,
                                         f"{wrench_name}_b{blade_damages_str}_w{wind_speed}_r{rpm}{comment}")

                # Produce the plots
                figure_number = data_pd_plotter(models_stats, figure_number, plot_name=plot_name,
                                                data_type=abbreviation[-1], alpha_angle=alpha_angle,
                                                switch_error_percentage=switch_error_percentage)

        # Retrieve the default scenario whose data will be subtracted to the rest
        mus_default = np.zeros(len(rpm_lst))
        stds_default = np.zeros(len(rpm_lst))
        if switch_subtract_no_damage:
            blade_damage = blade_damages[0]

            # Retrieving data files
            filters = {"blade_damage": blade_damage, "wind_speed": wind_speed}
            data_data = filter_data(filenames[0], filters)
            data_stat = filter_data(filenames[0] + "_rpms", filters)

            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                mus_default = data_stat[f"BET_mu_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                stds_default = data_stat[f"BET_std_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()

        # Plot the rpm comparison
        # Extract the models stats
        models_stats = {}
        for counter, filename in enumerate(filenames):
            if switch_subtract_no_damage and counter == 0:
                continue
            blade_damage = blade_damages[counter]

            # Retrieving data files
            filters = {"blade_damage": blade_damage, "wind_speed": wind_speed}
            data_data = filter_data(filename, filters)
            data_stat = filter_data(filename + "_rpms", filters)

            model_stats = {}
            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                rpms = data_stat["rpm"].to_numpy()
                mus = data_stat[f"BET_mu_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                stds = data_stat[f"BET_std_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                model_stats[abbreviation] = {"rpms": rpms, "mus": mus-mus_default, "stds": stds-stds_default,
                                             "blade_damage": blade_damage}
            models_stats[f"BET model: {blade_damage}%"] = model_stats

        # Creating the name of the file where to save the plots
        if wind_speed != 0:
            plot_name = os.path.join(folder_name, f"error_b{blade_damages_str}_w{wind_speed}{comment}")
        else:
            angle_name = np.unique(data_data["alpha_angle"].to_numpy())[-1]
            plot_name = os.path.join(folder_name,
                                     f"error_b{blade_damages_str}_a{angle_name}_w{wind_speed}{comment}")

        figure_number = rpm_error_plotter(figure_number, models_stats, plot_name=plot_name,
                                          switch_error_percentage=switch_error_percentage,
                                          switch_plot_stds=switch_plot_stds,
                                          switch_subtract_no_damage=switch_subtract_no_damage)

    return figure_number


def plot_rpms_windspeeds(figure_number, filenames, blade_damage, model, wind_speed_lst, comment, switch_error_percentage,
                      switch_amplitude):
    """
    Plot the wrenches (absolute or relative) error with respect to the rpms for the different wind speeds in a single
    plot
    :param figure_number: the number of the next plot
    :param filenames: the names of the file with the information regarding a single blade damage
    :param blade_damage: the blade damage
    :param model: the name of the model to be plotted "Matlab" or "BET"
    :param wind_speed_lst: the list of wind speeds to plot
    :param comment: whether a comment should be added at the end of the plot name
    :param switch_error_percentage: whether the error should be absolute or relative
    :param switch_amplitude: whether the mean or the amplitude of the signals errors should be plotted
    :return:
    """
    dict_keys_txt, mean_or_amplitude, comment = check_amp_rel(switch_error_percentage, switch_amplitude, comment)
    models_stats = {}
    for filename in filenames:
        for wind_speed in wind_speed_lst:
            filters = {"wind_speed": wind_speed, "blade_damage": blade_damage}
            data_stat = filter_data(filename + "_rpms", filters)
            if data_stat.empty:
                continue
            model_stats = {}

            # Extract the information for every windspeed
            for i in range(len(abbreviations)):
                abbreviation = abbreviations[i]
                rpms = data_stat["rpm"].to_numpy()
                mus = data_stat[f"{model}_mu_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                stds = data_stat[f"{model}_std_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy()
                model_stats[abbreviation] = {"rpms": rpms, "mus": mus, "stds": stds, "blade_damage": blade_damage}
            models_stats[f"w={wind_speed}"] = model_stats

    # Obtain the filename to save the figure
    plot_name = os.path.join(f"b{blade_damage}",
                             f"error_b{blade_damage}_m{model}{comment}")

    # Plot the figure
    figure_number = rpm_error_plotter(figure_number, models_stats, plot_name=plot_name,
                                      switch_error_percentage=switch_error_percentage, switch_plot_stds=False, ncol=2)
    return figure_number


def filter_data(filename, filters):
    """
    Function to import data and filter it
    :param filename: name of the file
    :param filters: filters to be applied in the form of a dictionary
    :return:
    """
    df = pd.read_csv(os.path.join("Data_storage", filename + ".csv"))
    if len(filters.keys()) == 0:
        return df
    filtered_data = df[[all(tot) for tot in zip(*[(df[key] == filters[key]) for key in filters.keys()])]]
    return filtered_data


def check_amp_rel(switch_error_percentage, switch_amplitude, comment):
    """
    Check whether it is desired to plot the amplitude or the mean and the absolute or relative error
    :param switch_error_percentage: whether the relative error should be computed instead of the absolute error
    :param switch_amplitude: whether the amplitude should be plotted instead of the mean
    :return:
    """
    # Whether the error computed is absolute or relative
    error_percentage_txt = ""
    if switch_error_percentage:
        error_percentage_txt = "_rel"
        comment = "_relative" + comment

    # Whether the bias or the amplitude are considered
    amplitude_txt = ""
    mean_or_amplitude_val = "mean"
    mean_or_amplitude_mod = ""
    if switch_amplitude:
        amplitude_txt = "_amp"
        mean_or_amplitude_val = "amplitude"
        mean_or_amplitude_mod = "_amplitude"
        comment = "_amp" + comment

    dict_keys_txt = {"rel": error_percentage_txt, "amp": amplitude_txt}
    mean_or_amplitude = {"val": mean_or_amplitude_val, "mod": mean_or_amplitude_mod}

    return dict_keys_txt, mean_or_amplitude, comment


def create_model_stats(model, data_rpm_data, data_rpm_stat, mean_or_amplitude, abbreviation, wrench_name,
                       switch_val_error_bars, switch_amplitude, dict_keys_txt):
    """
    Create dictionary with model statistical data, validation and model data, as well as error data.
    :param model: name of the model
    :param data_rpm_data: extracted data filtered by rpm
    :param data_rpm_stat: statistical data filtered by rpm
    :param mean_or_amplitude: whether mean or amplitude are plotted
    :param abbreviation: whether thrust or torque abbreviation
    :param wrench_name: the name of the wrench
    :param switch_val_error_bars: whether to experimental error should be computed
    :param switch_amplitude: whether the amplitude is computed instead of the mean
    :param dict_keys_txt: dictionary of dictionary key components
    :return:
    """
    # Extracting experimental data
    corrected_experimental_data_mean = \
        data_rpm_data[f"{mean_or_amplitude['val']}_wind_corrected_{wrench_name}"].to_numpy()
    error_bar = None
    if switch_val_error_bars and not switch_amplitude:
        experimental_data_std = data_rpm_data[f"std_{wrench_name}"].to_numpy()
        correction_experimental_data_std = data_rpm_data[f"std_wind_correction_{wrench_name}"].to_numpy()

        error_bar = experimental_data_std + correction_experimental_data_std

    # If the Matlab model is chosen, the name is Gray-box model instead
    model_name = model
    if "Matlab" in model:
        model_name = "Gray-box"

    # Extract all the information types in data_points
    model_stats = {}
    for data_point in data_points:
        model_stats[f"{data_point}"] = \
            data_rpm_stat[
                f"{model}_{data_point}_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy().item()
    model_stats["validation_data"] = corrected_experimental_data_mean
    model_stats["error_bar"] = error_bar
    model_stats["model_data"] = data_rpm_data[f"{model}{mean_or_amplitude['mod']}_{abbreviation}"].to_numpy()

    return model_stats, model_name


if __name__ == "__main__":
    # User_input
    plot_single_damage = False
    plot_single_windspeed = True
    switch_error_percentage = True
    switch_amplitude = False
    switch_val_error_bars = False
    switch_plot_alpha_angles = True
    switch_subtract_no_damage = True
    switch_plot_stds = False

    blade_damage = 25
    filename_input_data = f"b{blade_damage}"
    # filename_input_data = "b25_a90_w0"
    filename_input_stat = f"b{blade_damage}_rpms"
    # filename_input_stat = "b25_a90_w0_rpms"

    wind_speed_lst = [2, 4, 6, 9, 12]  # (0), 2, 4, 6, 9, 12
    rpm_lst = [300, 500, 700, 900, 1100]
    blade_damage_compare = [0, 10, 25]
    comment = ""
    figure_number = 1
    if plot_single_windspeed:
        if plot_single_damage:
            plot_statistics(filename_input_stat, filename_input_data, blade_damage, wind_speed_lst, rpm_lst,
                            switch_plot_alpha_angles=switch_plot_alpha_angles, switch_amplitude=switch_amplitude,
                            switch_val_error_bars=switch_val_error_bars, switch_error_percentage=switch_error_percentage,
                            comment=comment)
        else:
            filenames = [f"b{b}" for b in blade_damage_compare]
            # filenames = [f"b{b}_a90_w0" for b in blade_damage_compare]
            data_damage_comparison(figure_number, wind_speed_lst, rpm_lst, filenames,
                                   switch_error_percentage=switch_error_percentage,
                                   switch_val_error_bars=switch_val_error_bars,
                                   switch_plot_alpha_angles=switch_plot_alpha_angles,
                                   switch_subtract_no_damage=switch_subtract_no_damage,
                                   switch_plot_stds=switch_plot_stds)
    else:
        model = "BET"
        filename_input_data = [f"b{blade_damage}_a90_w0", filename_input_data]
        figure_number = plot_rpms_windspeeds(figure_number, filename_input_data, blade_damage, model, wind_speed_lst,
                                             comment, switch_error_percentage, switch_amplitude)
