import numpy as np
from SoundLights.dataset_functions import import_json_to_dataframe

import pandas as pd
import os
import matplotlib.pyplot as plt

responses_ARAUS = pd.read_csv(
    os.path.join("data/csv_files", "SoundLights_ARAUS.csv"),
    dtype={"info.participant": str},
)  # , dtype = {'participant':str}

data = responses_ARAUS["info.wav_gain"].values
max = np.max(data)
p96 = np.percentile(data, 96)
p97 = np.percentile(data, 97)
p98 = np.percentile(data, 98)
p99 = np.percentile(data, 99)
p100 = np.percentile(data, 100)
print(p96, p97, p98, p99, p100)
data_norm = np.sort(data / 10)
print(data_norm[24400:].size)
print(np.where(data > 30))
""" hist_values, bin_edges, _ = plt.hist(
    data_norm, bins=200
)  # Adjust the number of bins as needed

plt.xlabel("Values")
plt.ylabel("Frequency")
title = "Histogram"
plt.title(title)
# plt.show() """

# Set the length of the arrays
length = 1440000
gain = 2
# Generate random values between ±0.3
array_1 = np.random.uniform(low=-0.3, high=0.3, size=length)
array_1_2 = array_1 * gain
# Generate random values between ±0.6
array_2 = np.random.uniform(low=-0.6, high=0.6, size=length)
array_2_2 = array_2 * gain
fs = 48000

from maad.spl import pressure2leq
from maad.util import mean_dB

raw_1 = mean_dB(pressure2leq(array_1, fs, 0.125))
raw_2 = mean_dB(pressure2leq(array_2, fs, 0.125))
raw_1_2 = mean_dB(pressure2leq(array_1_2, fs, 0.125))
raw_2_2 = mean_dB(pressure2leq(array_2_2, fs, 0.125))
print(raw_1, raw_2, raw_1_2, raw_2_2)

mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
array_3 = np.random.normal(mean, std_dev, length)
raw_3 = mean_dB(pressure2leq(array_3, fs, 0.125))
raw_3_2 = mean_dB(pressure2leq(array_3 * gain, fs, 0.125))
mean = 0.1  # Mean of the distribution
std_dev = 0.5  # Standard deviation of the distribution
array_4 = np.random.normal(mean, std_dev, length)
raw_4 = mean_dB(pressure2leq(array_4, fs, 0.125))
raw_4_2 = mean_dB(pressure2leq(array_4 * gain, fs, 0.125))
print(raw_3, raw_4, raw_3_2, raw_4_2)

from Mosqito.loadFiles import load

audio_path_original = (
    "data/soundscapes_augmented/ARAUS_fold0_01/fold_0_participant_10001_stimulus_02.wav"
)
audio_original, fs = load(audio_path_original, wav_calib=1.0, ch=1)
audio_path_6 = "data/ARAUS-extended_soundscapes/ARAUS_fold0_01/fold_0_participant_10001_stimulus_02.wav"
audio_6, fs = load(audio_path_6, wav_calib=1.0, ch=1)
raw_5 = mean_dB(pressure2leq(audio_original, fs, 0.125))
raw6 = mean_dB(pressure2leq(audio_6, fs, 0.125))
print(raw_5, raw6)
