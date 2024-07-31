import pandas as pd
import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.dataset.dataset_functions import (
    generate_features,
)

############### Code to generate fold0 variations (wav_gain variations) ###############
# Inputs
csvPath = "data/ARAUS_extended.csv"
audios_path = "data/soundscapes_augmented/ARAUS_fold0_01/"  # Fold_0 folder
saving_folder_path = "data/variations_fold0/"

# Calculate
df = pd.read_csv(csvPath)
variations = ["x0_5", "x2", "x4", "x6", "random"]
for variation in variations:
    print(variation)
    saving_path = saving_folder_path + "variation_" + variation + "/"
    print("Saving path ", saving_path)

    if variation == "x0_5":
        # wav_gain variation of x0.5 in linear scale = -6dB in decibels scale
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            0.5,
        )
    elif variation == "x2":
        # wav_gain variation of x2 in linear scale = +6dB in decibels scale
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            2,
        )
    elif variation == "x4":
        # wav_gain variation of x4 in linear scale = +12dB in decibels scale
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            4,
        )
    elif variation == "x6":
        # wav_gain variation of x6 in linear scale = +18dB in decibels scale
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            6,
        )
    elif variation == "random":
        # random wav_gain variation
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            9999,
        )

#######################################################################################

# NOTE:
# Both this section of code generates single JSONS for each analysed audio. To get general
# JSON or CSV file, use import_jsons_to_json()  or import_jsons_to_dataframe() functions,
# respectively, from lib/dataset/dataset_functions.py
