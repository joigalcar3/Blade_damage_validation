#!/usr/bin/env python3
"""
Contains function that calls for the specific scenario to simulate for the Lomb-Scargle periodogram testing.
"""

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


def select_scenario(chosen_scenario):
    """
    Function that contains the parameters for each of the scenarios
    :param chosen_scenario: number of the chosen scenario
    :return:
    """
    # Scenario 1: Figure 9.30 in thesis - \Delta Fz
    if chosen_scenario == 1:
        sinusoid_f = 95.5
        sampling_f = 6.68
        amplitude = 0.01
        bias = 0.05
        phase = 180
        std = 0.015
        time_std = 0.005658869367416815
        max_time = 20
        switch_time_innaccuracy = True

    # Scenario 2: easy scenario
    elif chosen_scenario == 2:
        sinusoid_f = 50
        sampling_f = 6.5
        amplitude = 3.5
        bias = 5
        phase = 0
        std = 0
        time_std = 1/sampling_f * 0.1
        max_time = 20
        switch_time_innaccuracy = False

    # Scenario 3: figure E.31 in thesis
    elif chosen_scenario == 3:
        sinusoid_f = 47.7464829275686
        sampling_f = 6.68
        amplitude = 0.001
        bias = 5
        phase = 180
        std = 0.015
        time_std = 0.005658869367416815
        max_time = 20
        switch_time_innaccuracy = True

    return [sinusoid_f, sampling_f, amplitude, bias, phase, std, time_std, max_time, switch_time_innaccuracy]
