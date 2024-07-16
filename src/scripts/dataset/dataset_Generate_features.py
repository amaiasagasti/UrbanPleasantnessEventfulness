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
from SoundLights.dataset.dataset_functions import (
    generate_features,
)


################# ARAUS extended dataset ############################
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

######################################################################
