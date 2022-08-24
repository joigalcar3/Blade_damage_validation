# from ......Simulator.Blade_damage import Propeller
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import sys


folder_files = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files"
folder_files_np = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files\\No_propeller"
folder_blade_damage = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Simulator" \
                      "\\Blade_damage"
sys.path.append(folder_blade_damage)

from Propeller import Propeller

figure_number = 1


# User_input
blade_damage = 0
alpha_angle = 90
wind_speed = 4
rpm = 300
switch_plot_experimental_validation = True

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
    plt.show()

    plt.figure(figure_number)
    figure_number += 1
    plt.plot(torque, 'ro')
    plt.title("Uncorrected Torque (Nm)")
    plt.xlabel("Samples [-]")
    plt.ylabel("\\tau [Nm]")
    plt.grid(True)
    plt.show()

# Obtain the wind correction
wind_correction_filename = f"a{alpha_angle}_w{wind_speed}.csv"
wind_correction_filepath = os.path.join(folder_files_np, wind_correction_filename)
wind_correction_content = pd.read_csv(wind_correction_filepath, skiprows=1)
wind_correction_thrust = wind_correction_content['Thrust (N)']
wind_correction_torque = wind_correction_content['Torque (N·m)']

mean_wind_correction_thrust = wind_correction_thrust[wind_correction_thrust.notna()].mean()
mean_wind_correction_torque = wind_correction_torque[wind_correction_torque.notna()].mean()
std_wind_correction_thrust = wind_correction_thrust[wind_correction_thrust.notna()].std()
std_wind_correction_torque = wind_correction_torque[wind_correction_torque.notna()].std()
print("\n Wind corrections thrust and torque")
print(f"The thrust mean: {mean_wind_correction_thrust}")
print(f"The thrust standard deviation: {std_wind_correction_thrust}")
print("------------------------------------------------")
print(f"The torque mean: {mean_wind_correction_torque}")
print(f"The torque standard deviation: {std_wind_correction_torque}")

if switch_plot_experimental_validation:
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(wind_correction_thrust, "ro")
    plt.title("Wind thrust corrections (N)")
    plt.xlabel("Samples [-]")
    plt.ylabel("T [N]")
    plt.grid(True)
    plt.show()

    plt.figure(figure_number)
    figure_number += 1
    plt.plot(wind_correction_torque, "ro")
    plt.title("Wind Torque corrections (Nm)")
    plt.xlabel("Samples [-]")
    plt.ylabel("\\tau [Nm]")
    plt.grid(True)
    plt.show()

# Apply the wind correction
mean_wind_corrected_thrust = mean_wind_uncorrected_thrust-mean_wind_correction_thrust
mean_wind_corrected_torque = mean_wind_uncorrected_torque-mean_wind_correction_torque
print("\n Wind corrected thrust and torque")
print(f"The thrust mean: {mean_wind_corrected_thrust}")
print("------------------------------------------------")
print(f"The torque mean: {mean_wind_corrected_torque}")


# Obtain the information from the blade damage model
# Create the propeller and the blades
from user_input import *

propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=switch_chords_twist_plotting)
propeller.create_blades()

# ----------------------------------------------------------------------------------------------------------------------
# Compute the location of the center of gravity of the propeller and the BladeSection chords
cg_location = propeller.compute_cg_location()
average_chords, segment_chords = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst, n_blade_segment_lst[0])
