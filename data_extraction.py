import pandas as pd
import os
import matplotlib as mpl
mpl.use('TkAgg')
from Blade_damage.Propeller import Propeller
from Blade_damage.user_input import *
from Blade_damage.helper_func import *


folder_files = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files"
folder_files_np = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files\\No_propeller"
folder_blade_damage = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Simulator" \
                      "\\Blade_damage"

figure_number = 1


# User_input
blade_damage_lst = [0]  # 0, 10, 25
alpha_angle_lst = [0]   # 0, 15, 30, 45, 60, 75, 90
wind_speed_lst = [2]  # (0), 2, 4, 6, 9, 12
rpm_lst = [300, 500, 700, 900, 1100]  # 300, 500, 700, 900, 1100
switch_plot_experimental_validation = False
switch_plot_models = False
switch_wind_correction = False
data_file_name = f"b{blade_damage_lst[0]}"
# data_file_name = f"b0_a{alpha_angle_lst[0]}_w0"

# Initialising parameters
# The data is going to be structured in an array of 5 dimensions, which are explained as follows
# 1. blade damage
# 2. angle of attack
# 3. speed of the wind
# 4. the rotation speed of the propeller
# 5. the type of information:
#   - the first index is going to be the value obtained from experiments
#   - the second value is the experimental corrected value
#   - the third value is obtained from the Matlab model
#   - the fourth value is obtained from the BET model


data_table = []

if blade_damage_lst[0] == 0:
    column_names = ["blade_damage", "alpha_angle", "wind_speed", "rpm", "average_rpms", "mean_wind_uncorrected_thrust",
                    "std_thrust", "mean_wind_uncorrected_torque", "std_torque", "mean_wind_correction_thrust",
                    "std_wind_correction_thrust", "mean_wind_correction_torque", "std_wind_correction_torque",
                    "mean_wind_corrected_thrust", "mean_wind_corrected_torque", "Matlab_T", "Matlab_N", "BET_T",
                    "BET_N"]
else:
    column_names = ["blade_damage", "alpha_angle", "wind_speed", "rpm", "average_rpms", "mean_wind_uncorrected_thrust",
                    "std_thrust", "mean_wind_uncorrected_torque", "std_torque", "mean_wind_correction_thrust",
                    "std_wind_correction_thrust", "mean_wind_correction_torque", "std_wind_correction_torque",
                    "mean_wind_corrected_thrust", "mean_wind_corrected_torque", "BET_T_mean", "BET_T_amplitude",
                    "BET_N_mean", "BET_N_amplitude"]

for blade_damage_counter, blade_damage in enumerate(blade_damage_lst):
    for alpha_angle_counter, alpha_angle in enumerate(alpha_angle_lst):
        for wind_speed_counter, wind_speed in enumerate(wind_speed_lst):
            for rpm_counter, rpm in enumerate(rpm_lst):
                data_row = [blade_damage, alpha_angle, wind_speed, rpm]
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

                data_row += [average_rpms, mean_wind_uncorrected_thrust, std_thrust, mean_wind_uncorrected_torque,
                             std_torque]

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

                # Obtain the wind correction
                if switch_wind_correction:
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

                    data_row += [mean_wind_correction_thrust, std_wind_correction_thrust, mean_wind_correction_torque,
                                 std_wind_correction_torque]

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

                    # Apply the wind correction
                    mean_wind_corrected_thrust = mean_wind_uncorrected_thrust-mean_wind_correction_thrust
                    mean_wind_corrected_torque = mean_wind_uncorrected_torque-mean_wind_correction_torque
                    print("\n Wind corrected thrust and torque")
                    print(f"The thrust mean: {mean_wind_corrected_thrust}")
                    print("------------------------------------------------")
                    print(f"The torque mean: {mean_wind_corrected_torque}")

                    data_row += [mean_wind_corrected_thrust, mean_wind_corrected_torque]
                else:
                    data_row += [0, 0, 0, 0, mean_wind_uncorrected_thrust, mean_wind_uncorrected_torque]

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
                cg_location = propeller.compute_cg_location()
                average_chords, segment_chords = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst, n_blade_segment_lst[0])


                body_velocity = np.array([[wind_speed*np.cos(np.deg2rad(alpha_angle)), 0, -wind_speed*np.sin(np.deg2rad(alpha_angle))]]).T

                if blade_damage == 0:
                    T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, average_rpms)

                    data_row += [T, N]

                # ----------------------------------------------------------------------------------------------------------------------
                # Put all the forces and moments together. The output is the actual thrust and moments generated by the prop
                # Local input
                n_blade_segment = 100
                dt = 0.001

                # Computations
                n_points = int(total_time/dt + 1)
                F_healthy_lst = np.zeros((3, n_points))
                M_healthy_lst = np.zeros((3, n_points))
                rotation_angle_lst = np.zeros(n_points)
                rotation_angle = 0
                propeller.set_rotation_angle(0)
                for i in range(n_points):
                    if not i % 10:
                        print(f'Iteration {i} out of {n_points-1}')

                    F, M = propeller.compute_mass_aero_healthy_FM(n_blade_segment, average_rpms, attitude, cla_coeffs, cda_coeffs,
                                                                  body_velocity, pqr, rho)

                    F_healthy_lst[:, i] = F.flatten()
                    M_healthy_lst[:, i] = M.flatten()
                    rotation_angle_lst[i] = rotation_angle
                    rotation_angle = propeller.update_rotation_angle(average_rpms, dt)

                # Plot the forces and moments
                if switch_plot_models:
                    plot_FM(np.arange(0, total_time+dt, dt), rotation_angle_lst, F_healthy_lst,
                            M_healthy_lst, mass_aero='t')

                modelled_healty_T = -np.mean(F_healthy_lst[-1, :])
                modelled_healty_T_amplitude = np.max(F_healthy_lst[-1, :])-np.min(F_healthy_lst[-1, :])
                modelled_healty_N = np.mean(M_healthy_lst[-1, :])
                modelled_healty_N_amplitude = np.max(M_healthy_lst[-1, :]) - np.min(M_healthy_lst[-1, :])

                if blade_damage == 0:
                    data_row += [modelled_healty_T, modelled_healty_N]
                else:
                    data_row += [modelled_healty_T, modelled_healty_T_amplitude,
                                 modelled_healty_N, modelled_healty_N_amplitude]

                data_table.append(data_row)

df = pd.DataFrame(data=data_table, columns=column_names)
df.to_csv(os.path.join("Data_storage", f"{data_file_name}.csv"), index=False)
