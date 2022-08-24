import pandas as pd
import scipy.signal as ss
import numpy as np
from sklearn.neighbors import KernelDensity
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
figure_number = 1

# User input
folder = "C:\\Users\\jialv\\OneDrive\\2020-2021\\Thesis project\\3_Execution_phase\\Wind tunnel data\\2nd Campaign\\Data\\Test_files_correct_names"
alpha_angle = 45
wind_speed = 0
blade_damage = 0
switch_plotting = True

# Retrieval of data
filename = f"a{alpha_angle}_w{wind_speed}_b{blade_damage}.csv"
file_path = os.path.join(folder, filename)
content = pd.read_csv(file_path, skiprows=1)

# Work with the rotational speed of the motors
rpms = content["Motor Electrical Speed (rad/s)"]
rpms_300 = rpms[rpms<320]

plt.figure(figure_number)
figure_number += 1
plt.plot(rpms)
plt.grid(True)
plt.show()

# Implementing the savgol filter
def apply_savgol_filter(signal, times, n_points, order):
    filtered_signal = signal
    for i in range(times):
        filtered_signal = ss.savgol_filter(filtered_signal, n_points, order)
    return filtered_signal

savitzky_filter_4_51 = apply_savgol_filter(rpms, 4, 51, 1)
savitzky_filter_4_51_gradient = np.gradient(savitzky_filter_4_51)

savitzky_filter_4_101 = apply_savgol_filter(rpms, 4, 101, 1)
savitzky_filter_4_101_gradient = np.gradient(savitzky_filter_4_101)

savitzky_filter_1_101 = apply_savgol_filter(rpms, 1, 101, 1)
savitzky_filter_1_101_gradient = np.gradient(savitzky_filter_1_101)

plt.figure(figure_number)
figure_number += 1
plt.plot(np.gradient(rpms), label="rpms")
plt.plot(savitzky_filter_4_51_gradient, label="savitzky_filter_4_51_gradient")
plt.plot(savitzky_filter_4_101_gradient, label="savitzky_filter_4_101_gradient")
plt.grid(True)
plt.legend()
plt.show()


# Apply convolutional filter
convolution_filter_size = 401
savitzky_filter_gradient_switch = savitzky_filter_1_101_gradient
savitzky_filter_gradient_switch[savitzky_filter_gradient_switch > 0] = 1
savitzky_filter_gradient_switch[savitzky_filter_gradient_switch < 0] = 0
filter_array = np.hstack((np.zeros(int((convolution_filter_size-1)/2+1)), np.ones(int((convolution_filter_size-1)/2))))
convolved_signal = ss.convolve(savitzky_filter_gradient_switch, filter_array)

plt.figure(figure_number)
figure_number += 1
plt.plot(savitzky_filter_gradient_switch)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figure_number)
figure_number += 1
plt.plot(convolved_signal)
plt.grid(True)
plt.legend()
plt.show()


# Butterworth filter
# Setting standard filter requirements.
order = 3
cutoff = 2
fs = 0.5
b, a = ss.butter(order, cutoff, btype='low', analog=False)

# Plotting the frequency response.
w, h = ss.freqz(b, a, worN=8000)
plt.figure(figure_number)
figure_number += 1
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Filtering and plotting
y = ss.lfilter(b, a, rpms)

plt.subplot(2, 1, 2)
plt.plot(rpms, 'b-', label='data')
plt.plot(y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
