import sys
import os

from SoundLights.dataset.dataset_functions import (
    generate_features,
    generate_features_internal,
)


# Inputs
audios_folders = "data/fold0_variations_freesound_audios/"
saving_path = "data/fold0_variations_Freesound_data/"
folders = [
    # "fold0_ARAUS_0_25",
    "fold0_ARAUS_0_5",
    "fold0_ARAUS_2",
    "fold0_ARAUS_4",
    "fold0_ARAUS_6",
    "fold0_ARAUS_normalized",
    "fold0_ARAUS_original_random",
]
csv_name = "data/csv_files/SoundLights_Freesound.csv"

for folder in folders:
    audios_path = audios_folders + folder + "/"
    save_in = saving_path + folder + "/"
    # Call function
    generate_features_internal(audios_path, csv_name, save_in, ["Freesound"])
