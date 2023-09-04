#!/usr/bin/env python3
"""
File where the user can write the inputs regarding the data extraction, data statistics generation and data plotting.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


# General inputs
b = 10                           # blade damage
w = 2                            # wind speed
alpha_angle_lst = [0, 15, 30, 45, 60, 75, 90]  # 0, 15, 30, 45, 60, 75, 90
rpm_lst = [700]                                # 300, 500, 700, 900, 1100
user_choice = False              # whether the user choice should be run instead of the predetermined configurations
switch_val_error_bars = True     # whether the validation error bars should be plotted

# Data extraction inputs
switch_data_extraction = False                # activate data extraction from the pre-processed data
switch_plot_experimental_validation = False   # whether the thrust and torque wind data corrections should be plotted
switch_plot_models = False                    # whether to plot the thrust and torque data from the models

switch_plot_fft = False          # whether to plot the fft of the BET signal for sinusoid identification
switch_plot_sinusoid_id = False  # whether to plot the identified sinusoid
id_type = "LS"                   # whether the signal reconstruction should be done with PSO or LS (LombScargle)

# Data statistics computation
switch_data_statistics = False   # activate the computation of data statistics

# General plotting inputs
switch_data_analysis = True         # whether the data should be analysed to create the plots
switch_error_percentage = True      # whether the error should be relative (True) or absolute (False)
plot_single_damage = True           # whether only one damage should be plotted or multiple for comparison
plot_single_windspeed = True        # whether the specified windspeed should be plotted or all of them for comparison
switch_amplitude = True             # activate the plot of the amplitude instead of the mean
switch_plot_alpha_angles = True     # activate the plot of the angles with transparencies
switch_plot_stds = True             # activate that the standard deviation whiskers are plotted
comment = "_PSO"

# Multiple damage input
switch_blade_damage_comparison = False  # whether the results with different blade damages should be plotted
blade_damage_compare = [0, 10, 25]      # the blade damages to compare
switch_subtract_no_damage = False       # activate that the 0% damage is subtracted from the damaged scenarios

# Multiple wind speeds input
model = "BET"                           # name of the model being analysed from the data
