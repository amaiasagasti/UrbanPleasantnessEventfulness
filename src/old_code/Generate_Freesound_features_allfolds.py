import sys
import os

sys.path.append(os.getcwd())
from src.SoundLights.dataset.dataset_functions import (
    generate_features,
    generate_features_internal,
)


# Inputs
audios_path = "data/regenerated/ARAUS_fold0_01/"
csv_path = "data/csv_files/SoundLights_Freesound.csv"
saving_path = "data/checking/"
generate_features_internal(audios_path, csv_path, saving_path, ["Freesound"])
