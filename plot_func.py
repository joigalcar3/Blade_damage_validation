import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

blue_color_hex = "#1f77b4"
red_color_hex = "#d62728"
blue_color_rgb = [i/255 for i in [31, 119, 180]]
red_color_rgb = [i/255 for i in [214, 39, 40]]


def data_pd_plotter(validation_data, model_1_data, model_2_data, figure_number, plot_name="dummy", data_type="T",
                    validation_data_std=None, alpha_angle=None):
    """
    Plot the experimental and model data as well as the error in the form of probabilistic distribution
    :param validation_data: data from the experiments
    :param model_1_data: data from the gray-box aerodynamic model
    :param model_2_data: data from the BET model
    :param figure_number: number of the next figure to plot
    :param plot_name: name of the plot
    :param data_type: whether thrust or torque would be plotted
    :param validation_data_std: the standard deviation of the experimental data gathered
    :param alpha_angle: the alpha angles of the data
    :return:
    """
    # Computation of the constant line
    # constant_x_min = np.min(validation_data) - 0.1 * abs(np.min(validation_data)) if np.min(validation_data) < 0 else 0
    # constant_x_max = 0 if np.max(validation_data) < 0 else np.max(validation_data) + 0.1 * abs(np.max(validation_data))

    constant_x_min = np.min(validation_data) - 0.1 * abs(np.min(validation_data))
    constant_x_max = np.max(validation_data) + 0.1 * abs(np.max(validation_data))

    constant_line_coords = np.array([constant_x_min, constant_x_max])

    # Computation of the BET and model line approximations
    matlab_m, matlab_b = np.polyfit(validation_data, model_1_data, 1)
    bet_m, bet_b = np.polyfit(validation_data, model_2_data, 1)

    matlab_y = matlab_m * constant_line_coords + matlab_b
    bet_y = bet_m * constant_line_coords + bet_b

    # Computation of the probability distributions
    matlab_error = validation_data - model_1_data
    bet_error = validation_data - model_2_data

    matlab_error_range = max(matlab_error) - min(matlab_error)
    bet_error_range = max(bet_error) - min(bet_error)

    matlab_tolerance = abs(matlab_error_range) * 0.2
    bet_tolerance = abs(bet_error_range) * 0.2
    maximum_tolerance = max(matlab_tolerance, bet_tolerance)
    range_limit = max(abs(min(np.hstack((matlab_error, bet_error)))), abs(max(np.hstack((matlab_error, bet_error))))) +\
                  maximum_tolerance

    # pdf_range = np.linspace(min(np.hstack((matlab_error, bet_error))) - maximum_tolerance,
    #                         max(np.hstack((matlab_error, bet_error))) + maximum_tolerance, 100)

    pdf_range = np.linspace(-range_limit, range_limit, 100)

    matlab_mu, matlab_std = norm.fit(matlab_error)
    matlab_p = norm.pdf(pdf_range, matlab_mu, matlab_std)

    bet_mu, bet_std = norm.fit(bet_error)
    bet_p = norm.pdf(pdf_range, bet_mu, bet_std)

    # Plotting the data and the probability distributions together
    f_T, ax_T_lst = plt.subplots(2, 1, gridspec_kw={'wspace': 0.5, 'hspace': 0.55, 'height_ratios': [3, 1]})
    figure_number += 1

    # Plotting the data
    ax_data = ax_T_lst[0]
    ax_data.plot(constant_line_coords, constant_line_coords, 'k--')

    if validation_data_std is not None:
        ax_data.errorbar(validation_data, validation_data, yerr=validation_data_std*1.96, linestyle="", color="k",
                         capsize=4)

    if alpha_angle is not None:
        red_color = np.asarray([red_color_rgb+[(a+10)/100] for a in alpha_angle])
        blue_color = np.asarray([blue_color_rgb+[(a+10)/100] for a in alpha_angle])
    else:
        red_color = red_color_hex
        blue_color = blue_color_hex

    ax_data.scatter(validation_data, model_1_data, c=red_color, marker="o", label="Gray-box model", s=100)
    ax_data.scatter(validation_data, model_2_data, c=blue_color, marker="v", label="BET model", s=100)

    ax_data.plot(constant_line_coords, matlab_y, color=red_color_hex, linestyle="-")
    ax_data.plot(constant_line_coords, bet_y,  color=blue_color_hex, linestyle="-")

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

    # Plotting the probability distributions
    ax_pd = ax_T_lst[1]
    ax_pd.scatter(matlab_error, np.zeros(len(matlab_error)), c=red_color, marker="o", s=100)
    ax_pd.scatter(bet_error, np.zeros(len(bet_error)), c=blue_color, marker="v", s=100)

    ax_pd.plot(pdf_range, matlab_p, color=red_color_hex, linewidth=2, alpha=0.1)
    ax_pd.plot(pdf_range, bet_p, color=blue_color_hex, linewidth=2, alpha=0.1)

    ax_pd.fill_between(pdf_range, matlab_p, color=red_color_hex, alpha=0.1)
    ax_pd.fill_between(pdf_range, bet_p, color=blue_color_hex, alpha=0.1)

    ax_pd.axvline(0, color="k", linestyle="--", alpha=0.2)

    if data_type == "T":
        ax_pd.set_xlabel("Thrust model error [N]")
    elif data_type == "N":
        ax_pd.set_xlabel("Torque model error [Nm]")
    ax_pd.set_yticks([])
    ax_pd.set_ylim(bottom=ax_pd.get_ylim()[0]*1.25)
    ax_pd.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    f_T.subplots_adjust(left=0.1, top=0.95, right=0.98, bottom=0.17)
    f_T.set_size_inches(19.24, 10.55)
    f_T.savefig(os.path.join("Plots_storage", f"{plot_name}.png"), bbox_inches='tight')
    return figure_number, matlab_mu, matlab_std, bet_mu, bet_std


def rpm_error_plotter(figure_number, df_rpms, plot_name="dummy2"):
    """
    Plot the evolution of the errors through different rpms with its mean and standard deviation
    :param figure_number: number of the figure to plot
    :param df_rpms: the dataframe which contaons the thrust and torque model error means and standard deviations
    :return:
    """
    f, ax_lst = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0.5, 'hspace': 0.2})
    figure_number += 1

    # Plot the thrust errors
    ax_T = ax_lst[0]

    rpms = df_rpms["rpm"].to_numpy()
    ax_T.errorbar(rpms, df_rpms["Matlab_mu_T"].to_numpy(), yerr=df_rpms["Matlab_std_T"].to_numpy() * 1.96,
                  color=red_color_hex, capsize=4, marker="o", alpha=0.5, label="Gray-box model")
    ax_T.errorbar(rpms, df_rpms["BET_mu_T"].to_numpy(), yerr=df_rpms["BET_std_T"].to_numpy() * 1.96,
                  color=blue_color_hex, capsize=4, marker="v", alpha=0.5, label="BET model")
    ax_T.grid(True)
    ax_T.legend(markerscale=3)
    ax_T.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_T.set_ylabel("T error [N]")
    ax_T.yaxis.set_label_coords(-0.1, 0.5)

    # Plot the torque errors
    ax_N = ax_lst[1]

    ax_N.errorbar(rpms, df_rpms["Matlab_mu_N"].to_numpy(), yerr=df_rpms["Matlab_std_N"].to_numpy() * 1.96,
                  color=red_color_hex, capsize=4, marker="o", alpha=0.5)
    ax_N.errorbar(rpms, df_rpms["BET_mu_N"].to_numpy(), yerr=df_rpms["BET_std_N"].to_numpy() * 1.96,
                  color=blue_color_hex, capsize=4, marker="v", alpha=0.5)
    ax_N.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_N.grid(True)
    ax_N.set_ylabel("N error [Nm]")
    ax_N.set_xlabel("Propeller rotational speed [rad/s]")
    ax_N.yaxis.set_label_coords(-0.1, 0.5)
    ax_N.set_xticks(rpms)
    f.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.17)
    f.set_size_inches(19.24, 10.55)
    f.savefig(os.path.join("Plots_storage", f"{plot_name}.png"))
    return figure_number

