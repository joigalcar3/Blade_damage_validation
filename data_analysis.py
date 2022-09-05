import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
from plot_func import data_pd_plotter, rpm_error_plotter

abbreviations = ["T", "N"]
wrench_names = ["thrust", "torque"]


def data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                  data_file_name, filename, comment="", switch_error_percentage=False):
    df = pd.read_csv(os.path.join("Data_storage", filename))
    if not blade_damage:
        column_names = ["rpm", "blade_damage", "wind_speed", "Matlab_mu_T", "Matlab_std_T", "BET_mu_T",
                        "BET_std_T", "Matlab_mu_N", "Matlab_std_N", "BET_mu_N", "BET_std_N"]
    else:
        column_names = ["rpm", "blade_damage", "wind_speed", "BET_mu_T", "BET_std_T", "BET_mu_N", "BET_std_N"]
    data_table = []
    for rpm in rpm_lst:
        data_row = [rpm, blade_damage, wind_speed]

        # Retrieving the data
        data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) & (df["rpm"] == rpm)]
        if switch_plot_alpha_angles:
            alpha_angle = data_rpm["alpha_angle"].to_numpy()
        else:
            alpha_angle = None

        # Obtaining the thrust plots
        # Experimental data
        for i in range(len(abbreviations)):
            abbreviation = abbreviations[i]
            wrench_name = wrench_names[i]

            corrected_experimental_data_mean = data_rpm[f"mean_wind_corrected_{wrench_name}"].to_numpy()
            experimental_data_std = data_rpm[f"std_{wrench_name}"].to_numpy()
            correction_experimental_data_std = data_rpm[f"std_wind_correction_{wrench_name}"].to_numpy()

            error_bars = experimental_data_std + correction_experimental_data_std

            # Models data
            if not blade_damage:
                matlab_data = data_rpm[f"Matlab_{abbreviation}"].to_numpy()
                bet_data = data_rpm[f"BET_{abbreviation}"].to_numpy()
                models_data = {"Gray-box model": matlab_data, "BET model": bet_data}
            else:
                bet_data = data_rpm[f"BET_mean_{abbreviation}"].to_numpy()
                models_data = {"BET model": bet_data}

            figure_number, models_stats = \
                data_pd_plotter(corrected_experimental_data_mean, models_data,
                                figure_number,
                                plot_name=f"{comment}{wrench_name}_b{blade_damage}_w{wind_speed}_r{rpm}",
                                data_type=abbreviation, validation_data_std=error_bars, alpha_angle=alpha_angle,
                                switch_error_percentage=switch_error_percentage)

            bet_mu = models_stats["BET model"]["model_mu"]
            bet_std = models_stats["BET model"]["model_std"]
            if not blade_damage:
                matlab_mu = models_stats["Gray-box model"]["model_mu"]
                matlab_std = models_stats["Gray-box model"]["model_std"]
                data_row += [matlab_mu, matlab_std, bet_mu, bet_std]
            else:
                data_row += [bet_mu, bet_std]
        data_table.append(data_row)

        df_rpms = pd.DataFrame(data=data_table, columns=column_names)
        df_rpms.to_csv(os.path.join("Data_storage", f"{comment}{data_file_name}.csv"), index=False)

    if switch_plot_rpms:
        df_rpms = pd.read_csv(os.path.join("Data_storage", f"{comment}{data_file_name}.csv"))
        if wind_speed == 0:
            angle_name = int(data_file_name[data_file_name.index('a')+1:data_file_name.index('w')-1])
            figure_number = rpm_error_plotter(figure_number, df_rpms,
                                              plot_name=f"{comment}error_b{blade_damage}_a{angle_name}_w{wind_speed}",
                                              switch_error_percentage=switch_error_percentage)
        else:
            figure_number = rpm_error_plotter(figure_number, df_rpms,
                                              plot_name=f"{comment}error_b{blade_damage}_w{wind_speed}",
                                              switch_error_percentage=switch_error_percentage)
    return figure_number


def data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                           filenames, comment="", switch_error_percentage=False):
    for rpm in rpm_lst:
        # Obtaining the thrust plots
        # Experimental data
        for i in range(len(abbreviations)):
            abbreviation = abbreviations[i]
            wrench_name = wrench_names[i]

            # Retrieving the data
            blade_damages = []
            models_data = {}
            for filename in filenames:
                blade_damage = int(filename[filename.index("b")+1:filename.index(".")])
                blade_damages.append(blade_damage)
                df = pd.read_csv(os.path.join("Data_storage", filename))
                data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) &
                              (df["rpm"] == rpm)]

                # Whether the different angles should be differentiated by transparency
                if switch_plot_alpha_angles:
                    alpha_angle = data_rpm["alpha_angle"].to_numpy()
                else:
                    alpha_angle = None

                # Models data
                bet_data = data_rpm[f"BET_mean_{abbreviation}"].to_numpy()
                models_data[f"BET model: {blade_damage}%"] = bet_data

            # Experimental data
            corrected_experimental_data_mean = data_rpm[f"mean_wind_corrected_{wrench_name}"].to_numpy()
            experimental_data_std = data_rpm[f"std_{wrench_name}"].to_numpy()
            correction_experimental_data_std = data_rpm[f"std_wind_correction_{wrench_name}"].to_numpy()

            error_bars = experimental_data_std + correction_experimental_data_std

            # File name
            blade_damages_str = "_".join(str(i) for i in blade_damages)
            plot_name = f"{comment}{wrench_name}_b_{blade_damages_str}_w{wind_speed}_r{rpm}"

            figure_number, models_stats = \
                data_pd_plotter(corrected_experimental_data_mean, models_data, figure_number, plot_name=plot_name,
                                data_type=abbreviation, validation_data_std=error_bars, alpha_angle=alpha_angle,
                                switch_error_percentage=switch_error_percentage)






