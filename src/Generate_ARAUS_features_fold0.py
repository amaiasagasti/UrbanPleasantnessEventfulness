import sys
from SoundLights.dataset_functions import generate_features, generate_features_internal

sys.path.append("..")


# Inputs
audios_path = "data/soundscapes_augmented/"
csvs_path = "data/csv_files/"
saving_path = "data/fold0_ARAUS/"
csv_names = [
    "SoundLights_ARAUS_fold0_equal",
    "SoundLights_ARAUS_fold0_random",
    "SoundLights_ARAUS_fold0_0_25",
    "SoundLights_ARAUS_fold0_0_5",
    "SoundLights_ARAUS_fold0_2",
    "SoundLights_ARAUS_fold0_4",
]
print("hola")
for csv_name in csv_names:
    complete_path = csvs_path + csv_name + ".csv"
    save_in = saving_path + csv_name + "/"
    # Call function
    generate_features_internal(audios_path, complete_path, save_in, ["ARAUS"])
