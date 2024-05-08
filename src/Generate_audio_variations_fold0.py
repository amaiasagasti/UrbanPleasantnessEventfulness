import numpy as np
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
        print("Done audio ", count_total, "/240")
