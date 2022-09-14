import pandas as pd
from scipy import signal
from pyswarm import pso
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
from frequency_extraction import frequency_extraction
from helper_funcs import compute_BET_signals, pso_cost_function, obtain_wind_correction, experimental_data_extraction


def damaged_prop_signal_id(fn, content, signal_name, mean_wind_correction, BET_model_signal, switch_plot_fft=False,
                           switch_plot_sinusoid_id=False):
    """
    Obtain the mean and the amplitude of the model and experimental signals
    :param fn: the number of the next figure to be plotted
    :param content: the dataframe containing the experimental information
    :param signal_name: whether the signal is thrust or torque
    :param mean_wind_correction: the bias correction function due to the wind impinging the test stand
    :param BET_model_signal: the BET signal
    :param switch_plot_fft: whether the fft of the BET signal should be plotted
    :param switch_plot_sinusoid_id: whether the identified sinusoid should be plotted
    :return:
    """
    time = content['Time (s)']

    if signal_name == "T":
        wrench_signal = content['Thrust (N)']
    elif signal_name == "N":
        wrench_signal = content['Torque (N·m)']
    else:
        raise ValueError(f"The selected signal ({signal_name}) does not exist.")

    wrench_signal = wrench_signal[wrench_signal.notna()] - mean_wind_correction
    wrench_signal_mean = wrench_signal.mean()
    wrench_signal_numpy = wrench_signal.to_numpy()
    detrended_wrench_signal = signal.detrend(wrench_signal_numpy) + wrench_signal_mean

    sampled_times = time[np.asarray(wrench_signal.keys())].to_numpy()
    sampled_times -= sampled_times[0]

    BET_mean = np.mean(BET_model_signal)
    BET_amplitude = (np.max(BET_model_signal) - np.min(BET_model_signal)) / 2
    if np.max(BET_model_signal) - np.min(BET_model_signal) != 0:
        fn, largest_frequency_wrench_signal = frequency_extraction(fn, BET_model_signal,
                                                                   dt, switch_plot_fft=switch_plot_fft, n_points=None)
        ub = [0.5, 2 * np.pi]
        lb = [0, 0]
        kwargs = {"sinusoid_f": largest_frequency_wrench_signal, "time_lst": sampled_times,
                  "data_lst": detrended_wrench_signal, "mean_sinusoid": wrench_signal_mean}
        xopt, fopt = pso(pso_cost_function, lb, ub, debug=True, swarmsize=5000, maxiter=20, kwargs=kwargs)

        if switch_plot_sinusoid_id:
            plt.figure(fn)
            fn += 1
            plt.plot(sampled_times, wrench_signal_numpy, "bo", label="Data")  # Plotting the data gathered
            plt.plot(sampled_times, detrended_wrench_signal, "ro", label="Detrended data")  # Plotting the detrended data gathered
            plt.plot(np.arange(0, total_time + dt, dt), BET_model_signal, "g--")  # Plotting the data from the model
            plt.plot(np.arange(0, np.max(sampled_times), dt), wrench_signal_mean +
                     xopt[0] * np.sin(largest_frequency_wrench_signal * np.arange(0, np.max(sampled_times), dt) + xopt[1]),
                     "r-")  # Plotting the approximated signal
            plt.grid(True)

        wrench_signal_amplitude = xopt[0]
    else:
        wrench_signal_amplitude = 0

    return fn, BET_mean, BET_amplitude, wrench_signal_mean, wrench_signal_amplitude


if __name__ == "__main__":
    folder_files = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
                   "\\2nd Campaign\\Data\\2_Pre-processed_data_files"
    folder_files_np = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
                      "\\2nd Campaign\\Data\\2_Pre-processed_data_files\\No_propeller"
    folder_data_storage = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
                          "\\2nd Campaign\\Code\\Blade_damage_validation\\Data_storage"

    # User input
    blade_damage = 25  # 0, 10, 25
    alpha_angle = 0  # 0, 15, 30, 45, 60, 75, 90
    wind_speed = 2  # (0), 2, 4, 6, 9, 12
    rpm = 500  # 300, 500, 700, 900, 1100
    switch_plot_models = True
    figure_number = 2

    # Obtaining the average rpm
    figure_number, content_df, average_rpms, mean_wind_uncorrected_thrust, std_thrust,mean_wind_uncorrected_torque, \
    std_torque = experimental_data_extraction(figure_number, blade_damage, alpha_angle, wind_speed, rpm, folder_files,
                                              switch_plot_experimental_validation=False, switch_print=False)

    processed_content = pd.read_csv(os.path.join(folder_data_storage, "b0.csv"))

    figure_number, F_healthy_lst, M_healthy_lst, _, _ = \
        compute_BET_signals(figure_number, blade_damage, alpha_angle, wind_speed, average_rpms, dt, switch_plot_models)

    BET_thrust = -F_healthy_lst[-1, :]
    BET_torque = M_healthy_lst[-1, :]

    # Corrections
    figure_number, mean_wind_correction_thrust, mean_wind_correction_torque, _, _ = \
        obtain_wind_correction(figure_number, alpha_angle, wind_speed, folder_files_np)

    # Obtaining the mean and amplitude of the thrust
    figure_number, BET_mean_T, BET_amplitude_T, wrench_signal_mean_T, wrench_signal_amplitude_T = \
        damaged_prop_signal_id(figure_number, content_df, "T", mean_wind_correction_thrust, BET_thrust)

    # Obtaining the mean and amplitude of the torque
    figure_number, BET_mean_N, BET_amplitude_N, wrench_signal_mean_N, wrench_signal_amplitude_N = \
        damaged_prop_signal_id(figure_number, content_df, "N", mean_wind_correction_torque, BET_torque)
