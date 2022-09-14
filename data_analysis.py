import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
from plot_func import data_pd_plotter, rpm_error_plotter

abbreviations_mean = ["T", "N"]
abbreviations_amplitude = ["amplitude_T", "amplitude_N"]
wrench_names = ["thrust", "torque"]


def data_analysis(figure_number, blade_damage, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                  filename_input, filename_output, comment="", switch_error_percentage=False,
                  switch_val_error_bars=True, switch_amplitude=False):
    """
    Method to analyse the data, which results in 2 plots:
        - the creation of plots with constant blade damage, wind speed and rpm: it shows the model(s) vs experimental
        value with the corresponding errors and Gaussian fitted curve. For a single scenario one can see the effect of
        the angles
        - the creations of plots with the Gaussian mean and standard deviation vs the different rpms. One can see the
        effect of the rpm in the accuracy of the models.
    :param figure_number: the number of the next figure to plot
    :param blade_damage: the percentage of blade damage
    :param wind_speed: the speed of the wind
    :param rpm_lst: the list of rpms considered
    :param switch_plot_alpha_angles: the switch to whether the angles should be differentiated with the transparent
    value of the data point
    :param switch_plot_rpms: the switch to plot the figure with the mean and standard deviation for the different rpms
    :param filename_input: the name of the input file to process
    :param filename_output: the name of the output file
    :param comment: whether the output name of the figure and data files should be modified
    :param switch_error_percentage: whether the error should be expressed in relative instead of absolute form
    :param switch_val_error_bars: whether the experimental standard deviation should plotted in the for of error bars
    :param switch_amplitude: in the case of blade damage, the signals are oscillatory. This switch activates the
    analysis of the amplitude instead of the mean.
    :return:
    """
    # Extraction of the processed data
    df = pd.read_csv(os.path.join("Data_storage", filename_input + ".csv"))

    # In the case that there is no damage, there is the Matlab signal
    if not blade_damage:
        column_names = ["rpm", "blade_damage", "wind_speed", "Matlab_mu_T", "Matlab_std_T", "BET_mu_T",
                        "BET_std_T", "Matlab_mu_N", "Matlab_std_N", "BET_mu_N", "BET_std_N"]
    else:
        column_names = ["rpm", "blade_damage", "wind_speed", "BET_mu_T", "BET_std_T", "BET_mu_N", "BET_std_N"]

    # Iterate over the rpms
    data_table = []
    for rpm in rpm_lst:
        data_row = [rpm, blade_damage, wind_speed]

        # Retrieving the data for the specific scenario
        data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) & (df["rpm"] == rpm)]

        # In the case that alpha_angle distinction is required, the used angles are extracted
        if switch_plot_alpha_angles:
            alpha_angle = data_rpm["alpha_angle"].to_numpy()
        else:
            alpha_angle = None

        # Making a distinction whether the mean or amplitude are used
        if switch_amplitude:
            abbreviations = abbreviations_amplitude
            mean_or_amplitude = "amplitude"
        else:
            abbreviations = abbreviations_mean
            mean_or_amplitude = "mean"

        # Obtaining the thrust plots
        # Experimental data
        for i in range(len(abbreviations)):
            abbreviation = abbreviations[i]
            wrench_name = wrench_names[i]

            corrected_experimental_data_mean = data_rpm[f"{mean_or_amplitude}_wind_corrected_{wrench_name}"].to_numpy()
            error_bar = None
            if switch_val_error_bars and not switch_amplitude:
                experimental_data_std = data_rpm[f"std_{wrench_name}"].to_numpy()
                correction_experimental_data_std = data_rpm[f"std_wind_correction_{wrench_name}"].to_numpy()

                error_bar = experimental_data_std + correction_experimental_data_std

            # Models data
            if not blade_damage:
                matlab_data = data_rpm[f"Matlab_{abbreviation}"].to_numpy()
                bet_data = data_rpm[f"BET_{abbreviation}"].to_numpy()
                models_data = {"Gray-box model": matlab_data, "BET model": bet_data}
                validation_data = {"Gray-box model": corrected_experimental_data_mean,
                                   "BET model": corrected_experimental_data_mean}
                error_bars = {"Gray-box model": error_bar,
                              "BET model": error_bar}
            else:
                bet_data = data_rpm[f"BET_{abbreviation}"].to_numpy()
                models_data = {"BET model": bet_data}
                validation_data = {"BET model": corrected_experimental_data_mean}
                error_bars = {"BET model": error_bar}

            folder_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}")
            plot_name = os.path.join(folder_name, f"{wrench_name}_b{blade_damage}_w{wind_speed}_r{rpm}{comment}")
            if not os.path.exists(os.path.join("Plots_storage", folder_name)):
                os.makedirs(os.path.join("Plots_storage", folder_name))

            figure_number, models_stats = \
                data_pd_plotter(validation_data, models_data,
                                figure_number,
                                plot_name=plot_name,
                                data_type=abbreviation[-1], validation_data_std=error_bars, alpha_angle=alpha_angle,
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
        df_rpms.to_csv(os.path.join("Data_storage", f"{filename_output}{comment}.csv"), index=False)

    if switch_plot_rpms:
        df_rpms = pd.read_csv(os.path.join("Data_storage", f"{filename_output}{comment}.csv"))
        if wind_speed == 0:
            angle_name = int(filename_output[filename_output.index('a')+1:filename_output.index('w')-1])
            plot_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}",
                                     f"error_b{blade_damage}_a{angle_name}_w{wind_speed}{comment}")
        else:
            plot_name = os.path.join(f"b{blade_damage}", f"w{wind_speed}",
                                     f"error_b{blade_damage}_w{wind_speed}{comment}")
        figure_number = rpm_error_plotter(figure_number, df_rpms,
                                          plot_name=plot_name,
                                          switch_error_percentage=switch_error_percentage)
    return figure_number


def data_damage_comparison(figure_number, wind_speed, rpm_lst, switch_plot_alpha_angles, switch_plot_rpms,
                           filenames, comment="", switch_error_percentage=False, switch_amplitude=False):
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

    # Creating the folder where the plots will be stored
    blade_damages_str = "_".join(str(i) for i in blade_damages)
    folder_name = os.path.join(f"b{blade_damages_str}", f"w{wind_speed}")
    if not os.path.exists(os.path.join("Plots_storage", folder_name)):
        os.makedirs(os.path.join("Plots_storage", folder_name))

    for rpm in rpm_lst:
        if switch_amplitude:
            abbreviations = abbreviations_amplitude
            mean_or_amplitude = "amplitude"
        else:
            abbreviations = abbreviations_mean
            mean_or_amplitude = "mean"

        # Obtaining the thrust plots
        # Experimental data
        for i in range(len(abbreviations)):
            abbreviation = abbreviations[i]
            wrench_name = wrench_names[i]

            # Retrieving the data
            models_data = {}
            validation_data = {}
            error_bars = {}
            for counter, filename in enumerate(filenames):
                blade_damage = blade_damages[counter]
                df = pd.read_csv(os.path.join("Data_storage", filename))
                data_rpm = df[(df["blade_damage"] == blade_damage) & (df["wind_speed"] == wind_speed) &
                              (df["rpm"] == rpm)]

                # Models data
                bet_data = data_rpm[f"BET_{abbreviation}"].to_numpy()
                models_data[f"BET model: {blade_damage}%"] = bet_data

                # Experimental data
                corrected_experimental_data_mean = data_rpm[f"{mean_or_amplitude}_wind_corrected_{wrench_name}"].to_numpy()
                experimental_data_std = data_rpm[f"std_{wrench_name}"].to_numpy()
                correction_experimental_data_std = data_rpm[f"std_wind_correction_{wrench_name}"].to_numpy()

                error_bar = experimental_data_std + correction_experimental_data_std
                validation_data[f"BET model: {blade_damage}%"] = corrected_experimental_data_mean
                error_bars[f"BET model: {blade_damage}%"] = error_bar

            # Whether the different angles should be differentiated by transparency
            if switch_plot_alpha_angles:
                alpha_angle = data_rpm["alpha_angle"].to_numpy()
            else:
                alpha_angle = None

            # File name
            plot_name = os.path.join(folder_name,
                                     f"{wrench_name}_b{blade_damages_str}_w{wind_speed}_r{rpm}{comment}")

            figure_number, models_stats = \
                data_pd_plotter(validation_data, models_data, figure_number, plot_name=plot_name,
                                data_type=abbreviation, validation_data_std=error_bars, alpha_angle=alpha_angle,
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






