from Blade_damage.Propeller import Propeller
from Blade_damage.user_input import *
from Blade_damage.helper_func import *
import pandas as pd


def compute_BET_signals(figure_number, blade_damage, alpha_angle, wind_speed, rpms, dt, switch_plot_models):
    """
    Computes the thrust and torque of the healthy remaining component of the propeller
    :param figure_number: the number of the next figure to plot
    :param blade_damage: percentage of lost blade
    :param alpha_angle: the angle of the propeller rotational plane
    :param wind_speed: the speed of the wind
    :param rpms: the rpms at which the propeller was rotating
    :param switch_plot_models: whether to plot the force and moment diagrams from the BET model
    :return:
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

    # Computations
    n_points = int(total_time / dt + 1)
    F_healthy_lst = np.zeros((3, n_points))
    M_healthy_lst = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)
    rotation_angle = 0
    propeller.set_rotation_angle(0)
    for i in range(n_points):
        if not i % 10:
            print(f'Iteration {i} out of {n_points - 1}')

        F, M = propeller.compute_mass_aero_healthy_FM(n_blade_segment, rpms, attitude, cla_coeffs, cda_coeffs,
                                                      body_velocity, pqr, rho)

        F_healthy_lst[:, i] = F.flatten()
        M_healthy_lst[:, i] = M.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(rpms, dt)

    # Plot the forces and moments
    if switch_plot_models:
        plot_FM(np.arange(0, total_time + dt, dt), rotation_angle_lst, F_healthy_lst,
                M_healthy_lst, mass_aero='t')
        figure_number += 3

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
    :return:
    """
    A = x[0]
    phase = x[1]
    current_prediction = mean_sinusoid + A * np.sin(sinusoid_f * time_lst + phase)
    cost_value = np.sum(np.abs(current_prediction - np.array(data_lst)))
    return cost_value


def obtain_wind_correction(alpha_angle, wind_speed, folder_files_np):
    """
    Computes the wind corrections for a signal
    :param alpha_angle: the angle of the propeller plane with respect to the airflow
    :param wind_speed: the speed of the wind
    :param folder_files_np: the directory where the no propeller files are stored
    :return:
    """
    # Corrections
    wind_correction_filename = f"a{alpha_angle}_w{wind_speed}.csv"
    wind_correction_filepath = os.path.join(folder_files_np, wind_correction_filename)
    wind_correction_content = pd.read_csv(wind_correction_filepath, skiprows=1)
    wind_correction_thrust = wind_correction_content['Thrust (N)']
    wind_correction_torque = wind_correction_content['Torque (N·m)']

    mean_wind_correction_thrust = wind_correction_thrust[wind_correction_thrust.notna()].mean()
    mean_wind_correction_torque = wind_correction_torque[wind_correction_torque.notna()].mean()

    return mean_wind_correction_thrust, mean_wind_correction_torque
