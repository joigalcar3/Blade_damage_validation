#!/usr/bin/env python3
"""
Script for computing the propeller incidence angles in the wind tunnel with the OptiTrack system.
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
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector provided."""
    return vector / np.linalg.norm(vector)

# Location of point 1 of baseline
x_coord_baseline_root = float(input("Please, provide the x coordinate of the baseline root."))  # 0.125649 at OJF
y_coord_baseline_root = float(input("Please, provide the y coordinate of the baseline root."))  # -2.323987 at OJF

# Location of point 2 of the baseline
x_coord_baseline_target = float(input("Please, provide the x coordinate of the baseline target."))  # 0.875930 at OJF
y_coord_baseline_target = float(input("Please, provide the y coordinate of the baseline target."))  # -2.334541 at OJF

# Baseline vector
baseline_root = np.array([x_coord_baseline_root, y_coord_baseline_root])
baseline_target = np.array([x_coord_baseline_target, y_coord_baseline_target])
baseline = baseline_target-baseline_root

# Obtain the test stand positions
nostop = True
while nostop:
    # Location of point 1 of test stand
    x_coord_test_stand_root = float(input("Please, provide the x coordinate of the test_stand root."))
    y_coord_test_stand_root = float(input("Please, provide the y coordinate of the test_stand root."))

    # Location of point 2 of test stand
    x_coord_test_stand_target = float(input("Please, provide the x coordinate of the test_stand target."))
    y_coord_test_stand_target = float(input("Please, provide the y coordinate of the test_stand target."))

    # Test stand vector
    test_stand_root = np.array([x_coord_test_stand_root, y_coord_test_stand_root])
    test_stand_target = np.array([x_coord_test_stand_target, y_coord_test_stand_target])
    test_stand = test_stand_target - test_stand_root

    # Compute propeller incidence angle
    v1_u = unit_vector(baseline)
    v2_u = unit_vector(test_stand)
    angle = abs(np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))
    alternative = 180 - angle

    print(f"The angle between the baseline and the test stand is {min(angle, alternative)}")

    cont = input("CONTINUE?")
    if cont == "n":
        nostop = False

