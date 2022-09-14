#!/usr/bin/env python3
"""
Provides the functions to plot the processed experimental data
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
# mpl.use('Agg')
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)

abbreviations = ["T", "N"]
wrench_names = {"T": "thrust", "N": "torque"}
wrench_units = {"T": "N", "N": "Nm"}
data_points = ["mu", "std", "m", "b", "tol"]
models = ["Matlab", "BET"]
red_color_hex = "#d62728"
blue_color_hex = "#1f77b4"
green_color_hex = "#2ca02c"
colors_hex = [red_color_hex, blue_color_hex, green_color_hex]
blue_color_rgb = [i / 255 for i in [31, 119, 180]]
red_color_rgb = [i / 255 for i in [214, 39, 40]]
green_color_rgb = [i / 255 for i in [44, 160, 44]]
markers = ["o", "v", "s"]


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
    error_percentage_txt = dict_keys_txt["rel"]
    amplitude_txt = dict_keys_txt["amp"]
    mean_or_amplitude_mod = mean_or_amplitude["mod"]
    mean_or_amplitude_val = mean_or_amplitude["val"]

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

                # model_stats = create_model_stats(model, data_rpm_data, data_rpm_stat, mean_or_amplitude, abbreviation,
                #                                  wrench_name, switch_val_error_bars, switch_amplitude, dict_keys_txt)
                error_bar = None

                corrected_experimental_data_mean = \
                    data_rpm_data[f"{mean_or_amplitude_val}_wind_corrected_{wrench_name}"].to_numpy()
                if switch_val_error_bars and not switch_amplitude:
                    experimental_data_std = data_rpm_data[f"std_{wrench_name}"].to_numpy()
                    correction_experimental_data_std = data_rpm_data[f"std_wind_correction_{wrench_name}"].to_numpy()

                    error_bar = experimental_data_std + correction_experimental_data_std

                # Models data
                models_stats = {}
                for model in active_models:
                    model_stats = {}
                    model_name = model
                    if "Matlab" in model:
                        model_name = "Gray-box"
                    for data_point in data_points:
                        model_stats[f"{data_point}"] = \
                            data_rpm_stat[
                                f"{model}_{data_point}_{abbreviation}{amplitude_txt}{error_percentage_txt}"].to_numpy().item()
                    model_stats["validation_data"] = corrected_experimental_data_mean
                    model_stats["error_bar"] = error_bar
                    model_stats["model_data"] = data_rpm_data[f"{model}{mean_or_amplitude_mod}_{abbreviation}"].to_numpy()
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
                mus = data_stat[f"{model}_mu_{abbreviation}{amplitude_txt}{error_percentage_txt}"].to_numpy()
                stds = data_stat[f"{model}_std_{abbreviation}{amplitude_txt}{error_percentage_txt}"].to_numpy()
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


def rpm_error_plotter(figure_number, models_stats, plot_name="dummy2", switch_error_percentage=False):
    """
    Plot the evolution of the errors through different rpms with its mean and standard deviation
    :param figure_number: number of the figure to plot
    :param models_stats: the dictionary that contains the thrust and torque statistical information for plotting
    :param plot_name: the name of the plot that should be produced as output
    :param switch_error_percentage: whether the relative error is plotted instead of the absolute error
    :return:
    """
    f, ax_lst = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.2}, num=figure_number)
    figure_number += 1

    # Plot the thrust errors
    for counter_ax, ax_name in enumerate(ax_lst):
        abbreviation = abbreviations[counter_ax]
        wrench_unit = wrench_units[abbreviation]
        for counter_model, model_name in enumerate(models_stats.keys()):
            wrench_data = models_stats[model_name][abbreviation]
            blade_damage = wrench_data["blade_damage"]
            rpms = wrench_data["rpms"]
            ax_name.errorbar(rpms, wrench_data["mus"], yerr=wrench_data["stds"] * 1.96, color=colors_hex[counter_model],
                             capsize=4, marker=markers[counter_model], alpha=0.5,
                             label=f"{model_name}: {blade_damage}%")

        ax_name.grid(True)
        ax_name.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
        if switch_error_percentage:
            ax_name.set_ylabel(f"{abbreviation} error [%]")
        else:
            ax_name.set_ylabel(f"{abbreviation} error [{wrench_unit}]")
        ax_name.yaxis.set_label_coords(-0.1, 0.5)

        if counter_ax == 0:
            ax_name.legend(markerscale=3)
        elif counter_ax == len(ax_lst) - 1:
            ax_name.set_xlabel("Propeller rotational speed [rad/s]")
            ax_name.set_xticks(rpms)

    f.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
    f.set_size_inches(19.24, 10.55)
    f.savefig(os.path.join("Plots_storage", f"{plot_name}.png"))
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
    ax_data.legend(markerscale=3, scatterpoints=3, loc=2)
    ax_data.grid(True)
    ax_data.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax_data.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Polishing the probability distributions plot
    ax_pd.axvline(0, color="k", linestyle="--", alpha=0.2)

    ax_pd.set_ylim(bottom=ax_pd.get_ylim()[0] * 1.25)
    ax_pd.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    f_T.subplots_adjust(left=0.1, top=0.95, right=0.98, bottom=0.17)
    f_T.set_size_inches(19.24, 10.55)
    f_T.savefig(os.path.join("Plots_storage", f"{plot_name}.png"), bbox_inches='tight')

    return figure_number


def data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                           filenames, comment="", switch_error_percentage=False, switch_amplitude=False,
                           switch_val_error_bars=True):
    """
    Method to create the same plots as in data analysis. However, instead of comparing Matlab and BET, it compares
    the performance of BET with different degrees of failure
    :param figure_number: the number of the next figure to plot
    :param wind_speed: the speed of the wind tunnel
    :param rpm_lst: the list of rpms to consider
    :param switch_plot_alpha_angles: whether the angles of the plane of rotation with respect to the flow should be
    visible
    :param switch_plot_rpms: to switch on a plot of the mean and standard error for the different rpms
    :param filenames: the names of the files that contain the data for the different blade damages
    :param comment: any comment to insert at the beginning of the plots file names
    :param switch_error_percentage: whether the error should be plotted as relative instead of absolute
    :param switch_amplitude: whether the amplitude should be plotted instead of the mean
    :return:
    """
    # Obtaining the blade damages
    blade_damages = []
    for filename in filenames:
        if "w" in filename:
            blade_damage = int(filename[filename.index("b") + 1:filename.index("_")])
        else:
            blade_damage = int(filename[filename.index("b") + 1:filename.index(".")])
        blade_damages.append(blade_damage)

    # Whether the error computed is absolute or relative and the mean or amplitude are plotted
    dict_keys_txt, mean_or_amplitude, comment = check_amp_rel(switch_error_percentage, switch_amplitude, comment)

    # Creating the folder where the plots will be stored
    blade_damages_str = "_".join(str(i) for i in blade_damages)
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
                df_stat = pd.read_csv(os.path.join("Data_storage", filename + "_rpms.csv"))
                df_data = pd.read_csv(os.path.join("Data_storage", filename + ".csv"))

                # Retrieving specific data
                data_rpm_stat = df_stat[(df_stat["blade_damage"] == blade_damage) &
                                        (df_stat["wind_speed"] == wind_speed) & (df_stat["rpm"] == rpm)]
                data_rpm_data = df_data[(df_data["blade_damage"] == blade_damage) &
                                        (df_data["wind_speed"] == wind_speed) & (df_data["rpm"] == rpm)]

                # Whether the different angles should be differentiated by transparency
                if switch_plot_alpha_angles:
                    alpha_angle = data_rpm_stat["alpha_angle"].to_numpy()
                else:
                    alpha_angle = None

                model_stats = create_model_stats("BET", data_rpm_data, data_rpm_stat, mean_or_amplitude, abbreviation,
                                                 wrench_name, switch_val_error_bars, switch_amplitude, dict_keys_txt)
                models_stats[f"BET model: {blade_damage}%"] = model_stats

            # File name
            plot_name = os.path.join(folder_name,
                                     f"{wrench_name}_b{blade_damages_str}_w{wind_speed}_r{rpm}{comment}")

            # Produce the plots
            figure_number = data_pd_plotter(models_stats, figure_number, plot_name=plot_name,
                                            data_type=abbreviation[-1], alpha_angle=alpha_angle,
                                            switch_error_percentage=switch_error_percentage)

    if switch_plot_rpms:
        if wind_speed == 0:
            data_file_names = [filename[:filename.index(".")] + f"_rpms{comment}" for filename in filenames]
        else:
            data_file_names = [f"b{i}_rpms{comment}" for i in blade_damages]
        df_rpms_lst = []
        for data_file_name in data_file_names:
            df_rpms = pd.read_csv(os.path.join("Data_storage", f"{data_file_name}.csv"))
            df_rpms_lst.append(df_rpms)

        if wind_speed == 0:
            angle_name = int(data_file_name[data_file_name.index('a')+1:data_file_name.index('w')-1])
            plot_name = os.path.join(folder_name, f"error_b{blade_damages_str}_a{angle_name}_w{wind_speed}{comment}")
            figure_number = rpm_error_plotter(figure_number, df_rpms_lst,
                                              plot_name=plot_name, switch_Matlab=False,
                                              switch_error_percentage=switch_error_percentage)
        else:
            plot_name = os.path.join(folder_name,
                                     f"error_b{blade_damages_str}_w{wind_speed}{comment}")
            figure_number = rpm_error_plotter(figure_number, df_rpms_lst,
                                              plot_name=plot_name, switch_Matlab=False,
                                              switch_error_percentage=switch_error_percentage)
        figure_number -= 1

        return figure_number


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
    :param model:
    :param data_rpm_data:
    :param data_rpm_stat:
    :param mean_or_amplitude:
    :param abbreviation:
    :param wrench_name:
    :param switch_val_error_bars:
    :param switch_amplitude:
    :param dict_keys_txt:
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

    model_stats = {}
    model_name = model
    if "Matlab" in model:
        model_name = "Gray-box"
    for data_point in data_points:
        model_stats[f"{data_point}"] = \
            data_rpm_stat[
                f"{model_name}_{data_point}_{abbreviation}{dict_keys_txt['amp']}{dict_keys_txt['rel']}"].to_numpy().item()
    model_stats["validation_data"] = corrected_experimental_data_mean
    model_stats["error_bar"] = error_bar
    model_stats["model_data"] = data_rpm_data[f"{model}{mean_or_amplitude['mod']}_{abbreviation}"].to_numpy()

    return model_stats


if __name__ == "__main__":
    # filename_input_data = "b0"
    filename_input_data = "b25_a90_w0"
    # filename_input_stat = "b0_rpms"
    filename_input_stat = "b25_a90_w0_rpms"
    blade_damage = 25
    wind_speed_lst = [0]
    rpm_lst = [300, 500, 700, 900, 1100]
    comment = ""
    plot_statistics(filename_input_stat, filename_input_data, blade_damage, wind_speed_lst, rpm_lst,
                    switch_plot_alpha_angles=True, switch_amplitude=True, switch_val_error_bars=True,
                    switch_error_percentage=True, comment=comment)


