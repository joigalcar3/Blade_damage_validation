#!/usr/bin/env python3
"""
Script for computing the propeller incidence angles in the wind tunnel with the OptiTrack system
"""

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


# Imports
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

# Location of point 1 of baseline
# x_coord_baseline_root = float(input("Please, provide the x coordinate of the baseline root."))
# y_coord_baseline_root = float(input("Please, provide the y coordinate of the baseline root."))
x_coord_baseline_root = 0.125649
y_coord_baseline_root = -2.323987

# Location of point 2 of the baseline
# x_coord_baseline_target = float(input("Please, provide the x coordinate of the baseline target."))
# y_coord_baseline_target = float(input("Please, provide the y coordinate of the baseline target."))
x_coord_baseline_target = 0.875930
y_coord_baseline_target = -2.334541

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
    alternative = 180-angle

    print(f"The angle between the baseline and the test stand is {min(angle, alternative)}")

    cont = input("CONTINUE?")
    if cont == "n":
        nostop = False

