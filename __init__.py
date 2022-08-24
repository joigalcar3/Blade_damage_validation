import os
import shutil
from change_file_names import change_file_name
from separate_rpm import separate_rpm

# Folders to work with
in_folder = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign" \
     "\\Data\\1_Raw_data_files"
temp_folder = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files\\Temp"
out_folder = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data" \
              "\\2nd Campaign\\Data\\2_Pre-processed_data_files"

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
