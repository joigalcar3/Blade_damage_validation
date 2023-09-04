#!/usr/bin/env python3
"""
Script to change the default filenames that are obtained from the Tyto stand by using the comments on the outputted file
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
import pandas as pd
import shutil


def change_file_name(folder, file, new_folder):
    """
    Function to change the name of the file
    :param folder: directory of the original file
    :param file: name of the original file
    :param new_folder: directory where the new file should be saved
    :return: None
    """
    file_path = os.path.join(folder, file)
    contents = pd.read_csv(file_path)
    comments = contents.keys()

    # Extract the information from the original file comments
    alpha_angle = -1
    wind_speed = -1
    blade_damage = -1
    for comment in comments:
        if "alpha" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" d")
            alpha_angle = int(comment[index_equal+1:index_space])
        if "wind_speed" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" m")
            wind_speed = int(comment[index_equal+1:index_space])
        if "blade_damage" in comment:
            index_equal = comment.index("=")
            index_space = comment.index(" %")
            blade_damage = int(comment[index_equal+1:index_space])

    # Information in the case the propeller was dismounted and the forces and moments that the wind exert on the stand
    # were to be recorded (the so called wind corrections)
    if "no propeller" in comments[0]:
        new_file_name = f"a{alpha_angle}_w{wind_speed}.csv"
        new_folder = os.path.join(os.path.dirname(new_folder), "No_propeller")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_file_path = os.path.join(new_folder, new_file_name)
    else:
        if blade_damage == -1:
            blade_damage = 0
        new_file_name = f"b{blade_damage}_a{alpha_angle}_w{wind_speed}.csv"
        new_file_path = os.path.join(new_folder, new_file_name)

    # Check for file existance
    if os.path.exists(new_file_path):
        raise ValueError(f"{new_file_name} already exists")

    # Copy file to new location
    shutil.copy2(file_path, new_file_path)


if __name__ == "__main__":
    wd = "Data_pre-processing\\Data\\Test files"
    nwd = "Data_pre-processing\\Data\\Test_files_correct_names"
    filenames = os.listdir(wd)
    for f in filenames:
        change_file_name(wd, f, nwd)
