import sys
import os
from scipy.io.wavfile import WavFileWarning
import warnings
import pandas as pd

# Suppress WavFileWarning
warnings.filterwarnings("ignore", category=WavFileWarning)

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.dataset.dataset_functions import (
    generate_features,
)


############### Code to generate ARAUS extended dataset ##################################
# Inputs
audios_path = "data/soundscapes_augmented/"
csv_path = "data/responses_adapted.csv"
saving_path = (
    "data/ARAUS_extended/"  # Specify saving path for JSONS files (one per audio)
)


csv_file = pd.read_csv(csv_path)
# Call function
generate_features(
    audios_path,
    csv_file,
    saving_path,
    ["ARAUS", "Freesound", "embedding"],
    "ARAUS_original",
    6.44,
    1,
)

#########################################################################################


# Code to generate features for new data (listenig tests audios) ########################
# Inputs
audios_path = "data/listening_test_audios/"
csv_path = "data/responses_fold_Fs.csv"
saving_path = "data/fold_Fs/"


csv_file = pd.read_csv(csv_path, delimiter=";")
# Call function
generate_features(
    audios_path,
    csv_file,
    saving_path,
    ["ARAUS", "Freesound", "embedding"],
    "new_data",
    6.44,
    1,
)

#########################################################################################

# NOTE:
# Both these sections of code generate single JSONS for each analysed audio. To get general
# JSON or CSV file, use import_jsons_to_json()  or import_jsons_to_dataframe() functions,
# respectively, from lib/dataset/dataset_functions.py, like:
# from lib.dataset.dataset_functions import import_json_to_dataframe
# import_json_to_dataframe("path/to/json",True,"path/to/new/csv")
