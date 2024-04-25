import sys
from SoundLights.dataset_functions import generate_features

sys.path.append("..")


# Inputs
audios_path = "data/soundscapes_augmented/"
ARAUScsv_path = "data/csv_files/responses_SoundLights.csv"
saving_path = "data/"

# Call function
generate_features(audios_path, ARAUScsv_path, saving_path, ["ARAUS", "Freesound"])
