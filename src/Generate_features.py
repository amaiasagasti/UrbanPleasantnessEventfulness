import sys
import os
from scipy.io.wavfile import WavFileWarning
import warnings
import pandas as pd

# Suppress WavFileWarning
warnings.filterwarnings("ignore", category=WavFileWarning)

from SoundLights.dataset.dataset_functions import (
    generate_features,
)


# Inputs
audios_path = "data/listening_test_audios/"
csv_path = "data/main_files/answers_listening_tests.csv"
saving_path = "data/listening_test_data/"


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
