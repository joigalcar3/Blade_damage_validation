import pandas as pd
import os
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
from plot_func import data_pd_plotter, rpm_error_plotter

# filename = "b0.csv"
filename = "b0_a0_w0.csv"

figure_number = 1

df = pd.read_csv(os.path.join("Data_storage", filename))


# User_input
blade_damage = 0
wind_speed = 0
rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
switch_plot_alpha_angles = False
switch_plot_rpms = True
# data_file_name = "b0_rpms"
data_file_name = "b0_a0_w0_rpms"

if switch_plot_alpha_angles:
    column_names = ["rpm", "blade_damage", "wind_speed", "Matlab_mu_T", "Matlab_std_T", "BET_mu_T",
                    "BET_std_T", "Matlab_mu_N", "Matlab_std_N", "BET_mu_N", "BET_std_N"]
    data_table = []
    for rpm in rpm_lst:
        data_row = [rpm, blade_damage, wind_speed]

        # Retrieving the data
        data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) & (df["rpm"] == rpm)]
        alpha_angle = data_rpm["alpha_angle"].to_numpy()

        # Obtaining the thrust plots
        corrected_experimental_T_data_mean = data_rpm["mean_wind_corrected_thrust"].to_numpy()
        experimental_T_data_std = data_rpm["std_thrust"].to_numpy()
        correction_experimental_T_data_std = data_rpm["std_wind_correction_thrust"].to_numpy()
        matlab_T_data = data_rpm["Matlab_T"].to_numpy()
        bet_T_data = data_rpm["BET_T"].to_numpy()

        error_bars = experimental_T_data_std + correction_experimental_T_data_std

        figure_number, matlab_mu_T, matlab_std_T, bet_mu_T, bet_std_T = \
            data_pd_plotter(corrected_experimental_T_data_mean, matlab_T_data, bet_T_data,
                            figure_number, plot_name=f"thrust_b{blade_damage}_w{wind_speed}_r{rpm}",
                            data_type="T", validation_data_std=error_bars, alpha_angle=alpha_angle)

        data_row += [matlab_mu_T, matlab_std_T, bet_mu_T, bet_std_T]
        # Obtaining the torque plots
        corrected_experimental_N_data_mean = data_rpm["mean_wind_corrected_torque"].to_numpy()
        experimental_N_data_std = data_rpm["std_torque"].to_numpy()
        correction_experimental_N_data_std = data_rpm["std_wind_correction_torque"].to_numpy()
        matlab_N_data = data_rpm["Matlab_N"].to_numpy()
        bet_N_data = data_rpm["BET_N"].to_numpy()

        error_bars = experimental_N_data_std + correction_experimental_N_data_std

        figure_number, matlab_mu_N, matlab_std_N, bet_mu_N, bet_std_N = \
            data_pd_plotter(corrected_experimental_N_data_mean, matlab_N_data, bet_N_data,
                            figure_number, plot_name=f"torque_b{blade_damage}_w{wind_speed}_r{rpm}",
                            data_type="N", validation_data_std=error_bars, alpha_angle=alpha_angle)
        data_row += [matlab_mu_N, matlab_std_N, bet_mu_N, bet_std_N]
        data_table.append(data_row)

    df_rpms = pd.DataFrame(data=data_table, columns=column_names)
    df_rpms.to_csv(os.path.join("Data_storage", f"{data_file_name}.csv"), index=False)
if switch_plot_rpms:
    df_rpms = pd.read_csv(os.path.join("Data_storage", f"{data_file_name}.csv"))
    figure_number = rpm_error_plotter(figure_number, df_rpms, plot_name=f"error_b{blade_damage}_w{wind_speed}")

plt.show()





