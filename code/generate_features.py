import sys
from Mosqito.loadFiles import load
from SoundLights.features import extract_ARAUS_features, extract_Freesound_features
import pandas as pd
import os
import json
from SoundLights.dataset_functions import generate_features

sys.path.append("..")


# Inputs
audios_path = "data/soundscapes_augmented/"
ARAUScsv_path = "data/responses_SoundLights.csv"
saving_path = "data/"

# Call function
generate_features(audios_path, ARAUScsv_path, saving_path)
