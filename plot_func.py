import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import plotly
import plotly.graph_objs as go

red_color_hex = "#d62728"
blue_color_hex = "#1f77b4"
green_color_hex = "#2ca02c"
colors_hex = [red_color_hex, blue_color_hex, green_color_hex]
blue_color_rgb = [i/255 for i in [31, 119, 180]]
red_color_rgb = [i/255 for i in [214, 39, 40]]
green_color_rgb = [i/255 for i in [44, 160, 44]]
markers = ["o", "v", "s"]


def data_pd_plotter(validations_data, models_data, figure_number, plot_name="dummy", data_type="T",
                    validation_data_std=None, alpha_angle=None, switch_error_percentage=False):
    """
    Plot the experimental and model data as well as the error in the form of probabilistic distribution
    :param validations_data: data from the experiments
    :param models_data: dictionary with the models data
    :param figure_number: number of the next figure to plot
    :param plot_name: name of the plot
    :param data_type: whether thrust or torque would be plotted
    :param validation_data_std: the standard deviation of the experimental data gathered
    :param alpha_angle: the alpha angles of the data
    :param switch_error_percentage: whether the error data should be relative instead of absolute
    :return:
    """
    # Computation of the constant line
    all_validation_data = np.array(list(validations_data.values())).flatten()
    constant_x_min = np.min(all_validation_data) - 0.1 * abs(np.min(all_validation_data))
    constant_x_max = np.max(all_validation_data) + 0.1 * abs(np.max(all_validation_data))

    constant_line_coords = np.array([constant_x_min, constant_x_max])

    # Computation of the BET and model line approximations
    models_names = models_data.keys()
    models_stats = {}
    tolerances = []
    errors = np.zeros((len(models_names), len(validations_data[list(models_names)[0]])))
    for counter, model_name in enumerate(models_names):
        # Retrieving model data
        model_data = models_data[model_name]
        validation_data = validations_data[model_name]

        # Computation of the models line approximations
        model_m, model_b = 0, 0
        switch_plot_linear_fits = False
        if validation_data.shape[0] > 1:
            model_m, model_b = np.polyfit(validation_data, model_data, 1)
            switch_plot_linear_fits = True

        model_y = model_m * constant_line_coords + model_b

        # Computation of the probability statistics of the errors
        if switch_error_percentage:
            model_error = np.divide(validation_data - model_data, np.maximum(np.abs(validation_data), 1e-10)) * 100
        else:
            model_error = validation_data - model_data
        model_error_range = max(model_error) - min(model_error)
        model_tolerance = abs(model_error_range) * 0.2
        model_mu, model_std = norm.fit(model_error)
        models_stats[model_name] = {"model_y": model_y, "model_error": model_error,
                                    "model_tolerance": model_tolerance, "model_mu": model_mu, "model_std": model_std}
        tolerances.append(model_tolerance)
        errors[counter, :] = model_error

    # Computation of the x-axis range of the probability distribution
    maximum_tolerance = max(tolerances)
    range_limit = max(abs(np.min(errors)), abs(np.max(errors))) + maximum_tolerance
    pdf_range = np.linspace(-range_limit, range_limit, 100)

    # Computation of the probability distributions of the errors
    for model_name in models_names:
        model_p = norm.pdf(pdf_range, models_stats[model_name]["model_mu"], models_stats[model_name]["model_std"])
        models_stats[model_name]["model_p"] = model_p

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
        error_bar = validation_data_std[model_name]
        if error_bar is not None:
            validation_data = validations_data[model_name]
            ax_data.errorbar(validation_data, validation_data, yerr=error_bar*1.96, linestyle="", color="k",
                             capsize=4)

    # In the case that the angles of the rotation plane with respect to the flow are provided, compute colors
    if alpha_angle is not None:
        red_color = np.asarray([red_color_rgb+[(a+10)/100] for a in alpha_angle])
        blue_color = np.asarray([blue_color_rgb+[(a+10)/100] for a in alpha_angle])
        green_color = np.asarray([green_color_rgb + [(a + 10) / 100] for a in alpha_angle])
    else:
        red_color = red_color_hex
        blue_color = blue_color_hex
        green_color = green_color_hex
    colors = [red_color, blue_color, green_color]

    for i, model_name in enumerate(models_data.keys()):
        # Retrieve model data and probability distribution
        model_data = models_data[model_name]
        model_error = models_stats[model_name]["model_error"]
        model_p = models_stats[model_name]["model_p"]
        validation_data = validations_data[model_name]

        # Plot model data
        ax_data.scatter(validation_data, model_data, c=colors[i], marker=markers[i], label=model_name, s=100)
        if switch_plot_linear_fits:
            ax_data.plot(constant_line_coords, models_stats[model_name]["model_y"], color=colors_hex[i], linestyle="-")

        # Plot model probability distribution
        ax_pd.scatter(model_error, np.zeros(len(model_error)), c=colors[i], marker=markers[i], s=100)
        ax_pd.plot(pdf_range, model_p, color=colors_hex[i], linewidth=2, alpha=0.1)
        ax_pd.fill_between(pdf_range, model_p, color=colors_hex[i], alpha=0.1)

    # Polishing the data plot
    if data_type == "T":
        ax_data.set_xlabel("Experiments corrected thrust [N]")
        ax_data.set_ylabel("Model thrust [N]")
    elif data_type == "N":
        ax_data.set_xlabel("Experiments corrected torque [Nm]")
        ax_data.set_ylabel("Model torque [Nm]")
    original_ylim = ax_data.get_ylim()
    ax_data.set_ylim(top=(original_ylim[-1]-original_ylim[0])*0.35+original_ylim[-1])
    ax_data.legend(markerscale=3, scatterpoints=3, loc=2)
    ax_data.grid(True)
    ax_data.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax_data.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Polishing the probability distributions plot
    ax_pd.axvline(0, color="k", linestyle="--", alpha=0.2)

    if data_type == "T":
        if switch_error_percentage:
            ax_pd.set_xlabel("Thrust model relative error [%]")
        else:
            ax_pd.set_xlabel("Thrust model absolute error [N]")
    elif data_type == "N":
        if switch_error_percentage:
            ax_pd.set_xlabel("Torque model relative error [%]")
        else:
            ax_pd.set_xlabel("Torque model absolute error [Nm]")
    ax_pd.set_yticks([])
    ax_pd.set_ylim(bottom=ax_pd.get_ylim()[0]*1.25)
    ax_pd.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    f_T.subplots_adjust(left=0.1, top=0.95, right=0.98, bottom=0.17)
    f_T.set_size_inches(19.24, 10.55)
    f_T.savefig(os.path.join("Plots_storage", f"{plot_name}.png"), bbox_inches='tight')

    return figure_number, models_stats


def rpm_error_plotter(figure_number, df_rpms_lst, plot_name="dummy2", switch_error_percentage=False,
                      switch_Matlab=True):
    """
    Plot the evolution of the errors through different rpms with its mean and standard deviation
    :param figure_number: number of the figure to plot
    :param df_rpms_lst: the list that contains the dataframes which caontain the thrust and torque model error means
    and standard deviations
    :param switch_same_model: whether the same model is plotted multiple times with different blade damages
    :return:
    """
    if type(df_rpms_lst) != list:
        df_rpms_lst = [df_rpms_lst]
    f, ax_lst = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.2}, num=figure_number)
    figure_number += 1

    # Plot the thrust errors
    ax_T = ax_lst[0]

    counter = 0
    for df_rpms in df_rpms_lst:
        blade_damage = df_rpms["blade_damage"][0]
        rpms = df_rpms["rpm"].to_numpy()
        if not blade_damage and switch_Matlab:
            ax_T.errorbar(rpms, df_rpms["Matlab_mu_T"].to_numpy(), yerr=df_rpms["Matlab_std_T"].to_numpy() * 1.96,
                          color=colors_hex[counter], capsize=4, marker=markers[counter], alpha=0.5,
                          label=f"Gray-box model: {blade_damage}%")
            counter += 1
        ax_T.errorbar(rpms, df_rpms["BET_mu_T"].to_numpy(), yerr=df_rpms["BET_std_T"].to_numpy() * 1.96,
                      color=colors_hex[counter], capsize=4, marker=markers[counter], alpha=0.5,
                      label=f"BET model: {blade_damage}%")
        counter += 1
    ax_T.grid(True)
    ax_T.legend(markerscale=3)
    ax_T.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
    if switch_error_percentage:
        ax_T.set_ylabel("T error [%]")
    else:
        ax_T.set_ylabel("T error [N]")
    ax_T.yaxis.set_label_coords(-0.1, 0.5)

    # Plot the torque errors
    ax_N = ax_lst[1]
    counter = 0
    for df_rpms in df_rpms_lst:
        blade_damage = df_rpms["blade_damage"][0]
        rpms = df_rpms["rpm"].to_numpy()
        if not blade_damage and switch_Matlab:
            ax_N.errorbar(rpms, df_rpms["Matlab_mu_N"].to_numpy(), yerr=df_rpms["Matlab_std_N"].to_numpy() * 1.96,
                          color=colors_hex[counter], capsize=4, marker=markers[counter], alpha=0.5)
            counter += 1
        ax_N.errorbar(rpms, df_rpms["BET_mu_N"].to_numpy(), yerr=df_rpms["BET_std_N"].to_numpy() * 1.96,
                      color=colors_hex[counter], capsize=4, marker=markers[counter], alpha=0.5)
        counter += 1
    ax_N.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))

    ax_N.grid(True)
    if switch_error_percentage:
        ax_N.set_ylabel("N error [%]")
    else:
        ax_N.set_ylabel("N error [Nm]")
    ax_N.set_xlabel("Propeller rotational speed [rad/s]")
    ax_N.yaxis.set_label_coords(-0.1, 0.5)
    ax_N.set_xticks(rpms)
    f.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
    f.set_size_inches(19.24, 10.55)
    f.savefig(os.path.join("Plots_storage", f"{plot_name}.png"))
    return figure_number


def plot_4D_data(filename, x_axis_data="alpha_angle", y_axis_data="wind_speed", z_axis_data="BET_T", color_data='rpm',
                 store_file_name="4D_b0_T"):
    """
    Method to plot the data in 3D.
    :param filename: name of the data file from which data should be retrieved
    :param x_axis_data: the data to be plotted in the x-axis
    :param y_axis_data: the data to be plotted in the y-axis
    :param z_axis_data: the data to be plotted in the z-axis
    :param color_data: the data to be used to determine the color
    :param store_file_name: the name of the html file where to store the plot
    :return:
    """
    folder_data_storage = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
                          "\\2nd Campaign\\Code\\Blade_damage_validation\\Data_storage"

    directory_path = os.path.join(folder_data_storage, filename)

    df = pd.read_csv(directory_path)

    # Set marker properties
    markercolor = df[color_data]

    # Make Plotly figure
    fig1 = go.Scatter3d(x=df[x_axis_data],
                        y=df[y_axis_data],
                        z=df[z_axis_data],
                        marker=dict(color=markercolor,
                                    opacity=1,
                                    reversescale=True,
                                    colorscale='Viridis',
                                    size=5),
                        line=dict(width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    mylayout = go.Layout(scene=dict(xaxis=dict( title=x_axis_data),
                                    yaxis=dict( title=y_axis_data),
                                    zaxis=dict(title=z_axis_data)),)

    #Plot and save html
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                         auto_open=True,
                         filename=(f"{store_file_name}.html"))

if __name__ == "__main__":
    filename = "b0.csv"
    x_axis_data = "alpha_angle"
    y_axis_data = "wind_speed"
    z_axis_data = "BET_T"
    color_data = 'rpm'
    store_file_name = "4D_b0_T"

    plot_4D_data(filename, x_axis_data, y_axis_data, z_axis_data, color_data, store_file_name)