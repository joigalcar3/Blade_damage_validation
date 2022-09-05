import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

red_color_hex = "#d62728"
blue_color_hex = "#1f77b4"
colors_hex = [red_color_hex, blue_color_hex]
blue_color_rgb = [i/255 for i in [31, 119, 180]]
red_color_rgb = [i/255 for i in [214, 39, 40]]
markers = ["o", "v"]


def data_pd_plotter(validation_data, models_data, figure_number, plot_name="dummy", data_type="T",
                    validation_data_std=None, alpha_angle=None, switch_error_percentage=False):
    """
    Plot the experimental and model data as well as the error in the form of probabilistic distribution
    :param validation_data: data from the experiments
    :param models_data: dictionary with the models data
    :param figure_number: number of the next figure to plot
    :param plot_name: name of the plot
    :param data_type: whether thrust or torque would be plotted
    :param validation_data_std: the standard deviation of the experimental data gathered
    :param alpha_angle: the alpha angles of the data
    :return:
    """
    # Computation of the constant line
    constant_x_min = np.min(validation_data) - 0.1 * abs(np.min(validation_data))
    constant_x_max = np.max(validation_data) + 0.1 * abs(np.max(validation_data))

    constant_line_coords = np.array([constant_x_min, constant_x_max])

    # Computation of the BET and model line approximations
    models_names = models_data.keys()
    models_stats = {}
    tolerances = []
    errors = np.zeros((len(models_names), len(validation_data)))
    for counter, model_name in enumerate(models_names):
        model_data = models_data[model_name]

        # Computation of the models line approximations
        model_m, model_b = np.polyfit(validation_data, model_data, 1)
        model_y = model_m * constant_line_coords + model_b

        # Computation of the probability statistics of the errors
        if switch_error_percentage:
            model_error = np.divide(validation_data - model_data, validation_data) * 100
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
    f_T, ax_T_lst = plt.subplots(2, 1, gridspec_kw={'wspace': 0.5, 'hspace': 0.55, 'height_ratios': [3, 1]})
    figure_number += 1

    # Extracting the axes of the subplots
    ax_data = ax_T_lst[0]
    ax_pd = ax_T_lst[1]

    # Plotting the linear curve of 100% success with its 1.96*std whiskers
    ax_data.plot(constant_line_coords, constant_line_coords, 'k--')
    if validation_data_std is not None:
        ax_data.errorbar(validation_data, validation_data, yerr=validation_data_std*1.96, linestyle="", color="k",
                         capsize=4)

    # In the case that the angles of the rotation plane with respect to the flow are provided, compute colors
    if alpha_angle is not None:
        red_color = np.asarray([red_color_rgb+[(a+10)/100] for a in alpha_angle])
        blue_color = np.asarray([blue_color_rgb+[(a+10)/100] for a in alpha_angle])
    else:
        red_color = red_color_hex
        blue_color = blue_color_hex
    colors = [red_color, blue_color]

    for i, model_name in enumerate(models_data.keys()):
        # Retrieve model data and probability distribution
        model_data = models_data[model_name]
        model_error = models_stats[model_name]["model_error"]
        model_p = models_stats[model_name]["model_p"]

        # Plot model data
        ax_data.scatter(validation_data, model_data, c=colors[i], marker=markers[i], label=model_name, s=100)
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


def rpm_error_plotter(figure_number, df_rpms, plot_name="dummy2", switch_error_percentage=False):
    """
    Plot the evolution of the errors through different rpms with its mean and standard deviation
    :param figure_number: number of the figure to plot
    :param df_rpms: the dataframe which contaons the thrust and torque model error means and standard deviations
    :return:
    """
    blade_damage = df_rpms["blade_damage"][0]

    f, ax_lst = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.2})
    figure_number += 1

    # Plot the thrust errors
    ax_T = ax_lst[0]

    rpms = df_rpms["rpm"].to_numpy()
    if not blade_damage:
        ax_T.errorbar(rpms, df_rpms["Matlab_mu_T"].to_numpy(), yerr=df_rpms["Matlab_std_T"].to_numpy() * 1.96,
                      color=red_color_hex, capsize=4, marker="o", alpha=0.5, label="Gray-box model")
    ax_T.errorbar(rpms, df_rpms["BET_mu_T"].to_numpy(), yerr=df_rpms["BET_std_T"].to_numpy() * 1.96,
                  color=blue_color_hex, capsize=4, marker="v", alpha=0.5, label="BET model")
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

    if not blade_damage:
        ax_N.errorbar(rpms, df_rpms["Matlab_mu_N"].to_numpy(), yerr=df_rpms["Matlab_std_N"].to_numpy() * 1.96,
                      color=red_color_hex, capsize=4, marker="o", alpha=0.5)
    ax_N.errorbar(rpms, df_rpms["BET_mu_N"].to_numpy(), yerr=df_rpms["BET_std_N"].to_numpy() * 1.96,
                  color=blue_color_hex, capsize=4, marker="v", alpha=0.5)
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

