""" import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from maad.util import mean_dB
from maad.spl import pressure2leq
from Mosqito.loadFiles import load
from src.SoundLights.dataset.wav_files import save_wav

# Path to folder containing original augmented soundscapes
audioFolderPath = "data/soundscapes_augmented/ARAUS_fold0_01"
# Path to original responeses.csv
csvPath = "data/csv_files/responses.csv"
ARAUScsv = pd.read_csv(csvPath)
# Path to save
savingPath = "data/fold0_variations_freesound_audios/fold0_ARAUS_6/"  # CHANGE NAME FOR DIFFERENT VARIATIONS !!!
# Type of variation of fold0 audios
type_variation = (
    "x6"  # CHANGE VARIATIONS "normalization", "x0_25", "x0_5", "x2", "x4", "x6" !!!
)
count_clip = 0
count_total = 0
clipping = []

files = sorted(os.listdir(audioFolderPath))
for file in files:
    print(file)

    if file.endswith(".mp3") or file.endswith(".wav"):
        audio_path = audioFolderPath + "/" + file
        # Load the stereo audio file
        audio_r, fs = load(audio_path, wav_calib=1.0, ch=1)
        audio_l, fs = load(audio_path, wav_calib=1.0, ch=0)
        print(np.max([audio_l, audio_r]))
        if type_variation == "normalization":
            max = np.max([np.abs(audio_l), np.abs(audio_r)])
            # Normalize
            audio_r = audio_r / max
            audio_l = audio_l / max
            adapted_signal = np.column_stack((audio_l, audio_r))
        else:
            # Find the row in responses.csv corresponding to current audio
            file_split = file.split("_")
            file_fold = int(file_split[1])
            file_participant = "ARAUS_" + file_split[3]
            file_stimulus = int(file_split[5].split(".")[0])
            audio_info_aug = ARAUScsv[ARAUScsv["fold_r"] == file_fold]
            audio_info_aug = audio_info_aug[
                audio_info_aug["stimulus_index"] == file_stimulus
            ]
            audio_info_aug = audio_info_aug[
                audio_info_aug["participant"] == file_participant
            ]
            # Get the original Leq of this audio
            true_Leq = audio_info_aug["Leq_R_r"].values[0]
            # Calculate gain from true Leq and "raw" Leq
            rawR_Leq = mean_dB(pressure2leq(audio_r, fs, 0.125))
            difference = true_Leq - rawR_Leq
            gain = 10 ** (difference / 20)
            # Normalisation gain to avoid a lot of clipping
            norm_gain = 6.44
            # Apply gain to audio
            safe_gain = gain / norm_gain
            if type_variation == "x0_25":
                adapted_audio_r = audio_r * safe_gain * 0.25
                adapted_audio_l = audio_l * safe_gain * 0.25
            elif type_variation == "x0_5":
                adapted_audio_r = audio_r * safe_gain * 0.5
                adapted_audio_l = audio_l * safe_gain * 0.5
            elif type_variation == "x2":
                adapted_audio_r = audio_r * safe_gain * 2
                adapted_audio_l = audio_l * safe_gain * 2
            elif type_variation == "x4":
                adapted_audio_r = audio_r * safe_gain * 4
                adapted_audio_l = audio_l * safe_gain * 4
            elif type_variation == "x6":
                adapted_audio_r = audio_r * safe_gain * 6
                adapted_audio_l = audio_l * safe_gain * 6
            adapted_signal = np.column_stack((adapted_audio_l, adapted_audio_r))
            max = np.max(adapted_audio_r)
            min = np.min(adapted_audio_r)
            # Clipping?
            if max > 1 or min < -1:
                count_clip = count_clip + 1
                adapted_signal = np.clip(adapted_signal, -1, 1)
        # Save audio
        if not os.path.exists(savingPath):
            os.makedirs(savingPath)
        savingPathComplete = savingPath + file
        save_wav(adapted_signal, fs, savingPathComplete)

        count_total = count_total + 1
        print("Done audio ", count_total, "/240") """

""" from SoundLights.dataset.dataset_functions import (
    import_jsons_to_json,
    import_dataframe_to_json,
)
import pandas as pd

csvPath = "data/csv_files/SoundLights_Freesound.csv"
df = pd.read_csv(csvPath)
import_dataframe_to_json(df, True, "SoundLights_Freesound") """

""" import json

# Open the JSON file
with open("data/main_files/SoundLights_complete.json", "r") as f:
    # Load the JSON data
    data_complete = json.load(f)
with open("data/main_files/SoundLights_CLAP.json", "r") as f:
    # Load the JSON data
    data_CLAP = json.load(f)

for i in data_complete:
    data_complete[i]["embeddings"] = data_CLAP[i]["embeddings"]

json_path = str("data/main_files/SoundLights_complete2.json")
with open(json_path, "w") as file:
    json.dump(data_complete, file, indent=4) """

""" import json
from SoundLights.dataset.dataset_functions import import_json_to_dataframe

# Open the JSON file

df = import_json_to_dataframe("data/main_files/SoundLights_CLAP.json")
df.to_csv("data/main_files/SoundLights_CLAP.csv", index=False) """

""" import pandas as pd
from SoundLights.dataset.dataset_functions import generate_features_general

csvPath = "data/main_files/answers_listening_tests.csv"
df = pd.read_csv(csvPath, delimiter=";")

generate_features_general(
    "data/prueba_audios_test/",
    df,
    "data/trying_test/",
    ["ARAUS", "Freesound", "embedding"],
    "new_data",
    6.44,
) """

import pandas as pd
from SoundLights.dataset.dataset_functions import (
    generate_features_general,
    generate_features,
)

csvPath = "data/main_files/responses_SoundLights2.csv"
df = pd.read_csv(csvPath)

generate_features_general(
    "data/prueba_audios/",
    df,
    "data/trying_Original2/",
    ["ARAUS", "Freesound", "embedding"],
    "ARAUS_original",
    6.44,
)

"""generate_features(
    "data/prueba_audios_changed_gain/",
    csvPath,
    "data/trying_Original_functiongenerate/",
    ["ARAUS", "Freesound", "embedding"],
)"""
