import sys
import os
from scipy.io.wavfile import WavFileWarning
import warnings

# Suppress WavFileWarning
warnings.filterwarnings("ignore", category=WavFileWarning)

from SoundLights.dataset.dataset_functions import (
    generate_features,
    generate_features_internal,
    generate_features_new_audios,
)

sys.path.append("..")


# Inputs
audios_path = "data/listening_test_audios/"
csv_path = "data/csv_files/listening_test_data.csv"
saving_path = "data/listening_test_data/"

# Call function
generate_features_new_audios(
    audios_path,
    csv_path,
    saving_path,
    ["ARAUS", "Freesound", "embedding"],
)
