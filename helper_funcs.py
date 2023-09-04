#!/usr/bin/env python3
"""
File that contains the help functions for the data extraction, statistical analysis and plotting.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from Blade_damage.Propeller import Propeller
from Blade_damage.user_input import n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub,\
    propeller_mass, percentage_hub_m, angle_first_blade, start_twist, finish_twist, switch_chords_twist_plotting,\
    n_blade_segment_lst, attitude, cla_coeffs, cda_coeffs, pqr, rho, total_time
from Blade_damage.helper_func import compute_average_chords, FM_time_simulation


def experimental_data_extraction(figure_number, blade_damage, alpha_angle, wind_speed, rpm, folder_files,
                                 switch_plot_experimental_validation=False, switch_print=False):
    """
    Extracting the desired experimental data
    :param figure_number: the number of the next figure to plot
    :param blade_damage: percentage of lost blade
    :param alpha_angle: the angle of the propeller rotational plane
    :param wind_speed: the speed of the wind
    :param rpm: the rpms at which the propeller was rotating
    :param folder_files: where all the exprimental data files are located
    :param switch_plot_experimental_validation: whether the experimental validation data (rpm, thrust and torque)
    should be plotted
    :param switch_print: whether information should be printed to the screen
    :return: number of the next figure, the raw file content, average rpms registered in the file run, mean and standard
    deviation of the thrust and the torque
    """
    # Obtain the information from the validation wind tunnel experiments
    # Obtain the wind uncorrected thrust and torque
    filename = f"b{blade_damage}_a{alpha_angle}_w{wind_speed}_r{rpm}.csv"

    content = pd.read_csv(os.path.join(folder_files, filename))

    rpms = content["Motor Electrical Speed (rad/s)"]
    thrust = content['Thrust (N)']
    torque = content['Torque (N·m)']
    average_rpms = rpms.mean()
    mean_wind_uncorrected_thrust = thrust[thrust.notna()].mean()
    mean_wind_uncorrected_torque = torque[torque.notna()].mean()
    std_thrust = thrust[thrust.notna()].std()
    std_torque = torque[torque.notna()].std()

    if switch_print:
        print(f"The rpm mean: {average_rpms}")
        print("\n Experimental thrust and torque")
        print(f"The thrust mean: {mean_wind_uncorrected_thrust}")
        print(f"The thrust standard deviation: {std_thrust}")
        print("------------------------------------------------")
        print(f"The torque mean: {mean_wind_uncorrected_torque}")
        print(f"The torque standard deviation: {std_torque}")

    if switch_plot_experimental_validation:
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(rpms)
        plt.title("Motor Electrical Speed (rad/s)")
        plt.xlabel("Samples [-]")
        plt.ylabel("RPMS [rad/s]")
        plt.grid(True)

        plt.figure(figure_number)
        figure_number += 1
        plt.plot(thrust, 'ro')
        plt.title("Uncorrected Thrust (N)")
        plt.xlabel("Samples [-]")
        plt.ylabel("T [N]")
        plt.grid(True)

        plt.figure(figure_number)
        figure_number += 1
        plt.plot(torque, 'ro')
        plt.title("Uncorrected Torque (Nm)")
        plt.xlabel("Samples [-]")
        plt.ylabel("\\tau [Nm]")
        plt.grid(True)

    return figure_number, content, average_rpms, mean_wind_uncorrected_thrust, std_thrust, mean_wind_uncorrected_torque, std_torque


def compute_BET_signals(figure_number, blade_damage, alpha_angle, wind_speed, rpms, dt, switch_plot_models):
    """
    Computes the thrust and torque of the healthy remaining component of the propeller
    :param figure_number: the number of the next figure to plot
    :param blade_damage: percentage of lost blade
    :param alpha_angle: the angle of the propeller rotational plane
    :param wind_speed: the speed of the wind
    :param rpms: the rpms at which the propeller was rotating
    :param dt: time step for BET time simulation of forces and moments
    :param switch_plot_models: whether to plot the force and moment diagrams from the BET model
    :return: the number of the next figure, the thrust and the torque of the BET and gray-box aerodynamic model (Matlab)
    """
    # Obtaining the information from the BET model
    # Obtain the information from the blade damage model
    # Create the propeller and the blades
    percentage_broken_blade_length = [blade_damage, 0, 0]
    propeller = Propeller(1, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub,
                          propeller_mass, percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                          broken_percentage=percentage_broken_blade_length,
                          plot_chords_twist=switch_chords_twist_plotting)
    propeller.create_blades()

    # ----------------------------------------------------------------------------------------------------------------------
    # Compute the location of the center of gravity of the propeller and the BladeSection chords
    _ = propeller.compute_cg_location()
    _, _ = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst,
                                  n_blade_segment_lst[0])

    # Put all the forces and moments together. The output is the actual thrust and moments generated by the prop
    # Local input
    n_blade_segment = 100
    body_velocity = np.array(
        [[wind_speed * np.cos(np.deg2rad(alpha_angle)), 0, -wind_speed * np.sin(np.deg2rad(alpha_angle))]]).T
    if abs(body_velocity[0, 0]) < 1e-12: body_velocity[0, 0] = 0
    if abs(body_velocity[1, 0]) < 1e-12: body_velocity[1, 0] = 0
    if abs(body_velocity[2, 0]) < 1e-12: body_velocity[2, 0] = 0

    # Computation of forces and moments from the mass and aerodynamic effects
    propeller_func_input = {"number_sections": n_blade_segment, "omega": rpms, "cla_coeffs": cla_coeffs,
                            "cda_coeffs": cda_coeffs, "body_velocity": body_velocity, "pqr": pqr, "rho": rho,
                            "attitude": attitude, "total_time": total_time, "dt": dt,
                            "n_points": int(total_time / dt + 1), "rotation_angle": 0}
    F_healthy_lst, M_healthy_lst = FM_time_simulation(propeller, propeller.compute_mass_aero_healthy_FM,
                                                      propeller_func_input, mass_aero="t",
                                                      switch_plot=switch_plot_models)

    # Add 3 to the figure counter if plotting
    if switch_plot_models: figure_number += 3

    # Also return the information of the Matlab model when needed
    T, N = 0, 0
    if blade_damage == 0:
        T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, rpms)

    return figure_number, F_healthy_lst, M_healthy_lst, T, N


def pso_cost_function(x, sinusoid_f, time_lst, data_lst, mean_sinusoid):
    """
    The cost function for the particle swarm optimization
    :param x: the parameter to optimize
    :param sinusoid_f: the frequency of the sinusoid that is expected to be found
    :param time_lst: the time stamps at which experimental data was collected
    :param data_lst: the collected experimental data
    :param mean_sinusoid: the mean of the sinusoidal signal
    :return: the current cost value
    """
    A = x[0]  # the first parameter to optimize is the amplitude of the sinusoid
    phase = x[1]  # the second parameter to optimize is the phase of the sinusoid
    current_prediction = mean_sinusoid + A * np.sin(sinusoid_f * time_lst + phase)
    cost_value = np.sqrt(np.sum(np.square(current_prediction - np.array(data_lst))) / len(current_prediction))
    return cost_value


def frequency_extraction(figure_number, signal, dt, switch_plot_fft=False, n_points=None):
    """
    Extract the frequency of a signal using the Fast Fourier Transform
    :param figure_number: the number of the next figure to plot
    :param signal: the signal from which the frequency should be extracted
    :param dt: the time that has passed between sample and sample
    :param switch_plot_fft: whether to plot the FFT
    :param n_points: number of the signal datapoints to use for the FFT
    :return: number of the next figure and the frequency with the largest value from the FFT.
    """
    # When the number is not specified, then the whole signal is chosen
    if n_points is None:
        n_points = signal.shape[0]

    # Output in the form of cycles/s. Transformed to rad/s
    yf = fft(signal)[0:n_points // 2]
    xf = fftfreq(n_points, dt)[:n_points // 2] * 2 * np.pi

    # Plot the FFT signal
    if switch_plot_fft:
        plt.figure(figure_number)
        figure_number += 1
        plt.plot(xf, 2.0 / n_points * np.abs(yf), color="r", marker="o")
        plt.ylabel("Amplitude [-]")
        plt.xlabel("Frequency [rad/s]")
        plt.grid()

    # Extract what is the frequency with the largest amplitude from the FFT
    largest_frequency = xf[np.where(abs(yf) == max(abs(yf[1:])))[0].item()]

    return figure_number, largest_frequency


def obtain_wind_correction(figure_number, alpha_angle, wind_speed, folder_files_np,
                           switch_wind_correction=True, switch_plot_experimental_validation=False,
                           switch_print_info=False):
    """
    Computes the wind corrections for a signal.
    :param figure_number: number of the next figure to be plotted
    :param alpha_angle: the angle of the propeller plane with respect to the airflow
    :param wind_speed: the speed of the wind
    :param folder_files_np: the directory where the no propeller files are stored
    :param switch_wind_correction: whether there should be a wind correction
    :param switch_plot_experimental_validation: whether the thrust and torque wind data corrections should be plotted
    :param switch_print_info: whether the retrieved wind information should be displayed
    :return: number of the next figure, the mean and standard deviation of the thrust and torque wind corrections
    """
    if switch_wind_correction:
        # Obtain forces and moments of the wind on the test stand without propeller
        wind_correction_filename = f"a{alpha_angle}_w{wind_speed}.csv"
        wind_correction_filepath = os.path.join(folder_files_np, wind_correction_filename)
        wind_correction_content = pd.read_csv(wind_correction_filepath, skiprows=1)
        wind_correction_thrust = wind_correction_content['Thrust (N)']
        wind_correction_torque = wind_correction_content['Torque (N·m)']

        # Compute the mean and the standard deviation
        mean_wind_correction_thrust = wind_correction_thrust[wind_correction_thrust.notna()].mean()
        mean_wind_correction_torque = wind_correction_torque[wind_correction_torque.notna()].mean()
        std_wind_correction_thrust = wind_correction_thrust[wind_correction_thrust.notna()].std()
        std_wind_correction_torque = wind_correction_torque[wind_correction_torque.notna()].std()

        # Print the wind correction results
        if switch_print_info:
            print("\n Wind corrections thrust and torque")
            print(f"The thrust mean: {mean_wind_correction_thrust}")
            print(f"The thrust standard deviation: {std_wind_correction_thrust}")
            print("------------------------------------------------")
            print(f"The torque mean: {mean_wind_correction_torque}")
            print(f"The torque standard deviation: {std_wind_correction_torque}")

        # Plot the thrust and torque wind corrections information
        if switch_plot_experimental_validation:
            plt.figure(figure_number)
            figure_number += 1
            plt.plot(wind_correction_thrust, "ro")
            plt.title("Wind thrust corrections (N)")
            plt.xlabel("Samples [-]")
            plt.ylabel("T [N]")
            plt.grid(True)

            plt.figure(figure_number)
            figure_number += 1
            plt.plot(wind_correction_torque, "ro")
            plt.title("Wind Torque corrections (Nm)")
            plt.xlabel("Samples [-]")
            plt.ylabel("\\tau [Nm]")
            plt.grid(True)
        return figure_number, mean_wind_correction_thrust, mean_wind_correction_torque, \
               std_wind_correction_thrust, std_wind_correction_torque
    return figure_number, 0, 0, 0, 0
