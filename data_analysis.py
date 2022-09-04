import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
from plot_func import data_pd_plotter, rpm_error_plotter


def data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                  data_file_name, filename):
    df = pd.read_csv(os.path.join("Data_storage", filename))
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
        if wind_speed == 0:
            angle_name = int(data_file_name[data_file_name.index('a')+1:data_file_name.index('w')-1])
            figure_number = rpm_error_plotter(figure_number, df_rpms,
                                              plot_name=f"error_b{blade_damage}_a{angle_name}_w{wind_speed}")
        else:
            figure_number = rpm_error_plotter(figure_number, df_rpms, plot_name=f"error_b{blade_damage}_w{wind_speed}")
    return figure_number





