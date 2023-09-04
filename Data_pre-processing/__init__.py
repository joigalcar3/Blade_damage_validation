#!/usr/bin/env python3
"""
Script to run the data pre-processing pipeline
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
import shutil

from .change_file_names import change_file_name
from .separate_rpm import separate_rpm

# Folders to work with
in_folder = "Data_pre-processing\\Data\\1_Raw_data_files"
temp_folder = "Data_pre-processing\\Data\\2_Pre-processed_data_files\\Temp"
out_folder = "Data_pre-processing\\Data\\2_Pre-processed_data_files"

shutil.rmtree(temp_folder)
shutil.rmtree(out_folder)
os.makedirs(temp_folder)

# Correct names from comments
filenames = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]
for filename in filenames:
    change_file_name(in_folder, filename, temp_folder)

print("Correct names: finished!")

# Separate by rpm values
switch_plotting = False
figure_number = 1

filenames = [f for f in os.listdir(temp_folder) if os.path.isfile(os.path.join(temp_folder, f))]
for filename in filenames:
    figure_number = separate_rpm(filename, figure_number, temp_folder, out_folder, plotting=switch_plotting)

print("RPM value separation: finished!")

shutil.rmtree(temp_folder)
