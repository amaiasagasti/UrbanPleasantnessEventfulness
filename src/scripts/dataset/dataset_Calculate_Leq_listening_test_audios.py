import sys
import os
import pandas as pd
from maad.util import mean_dB
from maad.spl import pressure2leq


# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from SoundLights.dataset.Mosqito.loadFiles import load

############### Code to calculate raw Leq of listening test audios ###############
# Inputs
audioFolderPath = "data/listening_test_audios/"
saving_path = "data/output_delete/raw_Leq_listening_test.csv"

# Calculate
files = sorted(os.listdir(audioFolderPath))
df = pd.DataFrame(columns=["info.file", "info.Leq_R", "info.Leq_L"])
for file in files:
    if file.endswith(".mp3") or file.endswith(".wav"):
        audio_path = audioFolderPath + file
        # Load the stereo audio file
        audio_r, fs = load(audio_path, wav_calib=1.0, ch=1)
        audio_l, fs = load(audio_path, wav_calib=1.0, ch=0)
        # Calculate Leq
        R_Leq = mean_dB(pressure2leq(audio_r, fs, 0.125))
        L_Leq = mean_dB(pressure2leq(audio_l, fs, 0.125))
        # Add to dataframe
        new_row = {"info.file": file, "info.Leq_R": R_Leq, "info.Leq_L": L_Leq}
        df = pd.concat([df, pd.DataFrame([new_row])])

# Save
df.to_csv(saving_path, index=False)
##################################################################################
