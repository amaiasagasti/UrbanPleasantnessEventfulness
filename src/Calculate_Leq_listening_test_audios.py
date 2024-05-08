import sys
import os
import pandas as pd
from maad.util import mean_dB
from maad.spl import pressure2leq


sys.path.append(os.getcwd())
from src.Mosqito.loadFiles import load

audioFolderPath = "data/listening_test_audios/"
files = sorted(os.listdir(audioFolderPath))
df = pd.DataFrame(columns=["info.file", "info.Leq_R", "info.Leq_L"])
for file in files:
    if file.endswith(".mp3") or file.endswith(".wav"):
        print("File ", file)
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
# df.to_csv("output.csv", index=False)

""" audio_tone_path = "data/listening_test_audios/1000Hz-calibration.wav"
audio_tone, fs = load(audio_tone_path, wav_calib=1.0, ch=1)
audio_tone_Leq = mean_dB(pressure2leq(audio_tone, fs, 0.125))
print("Calibration tone Leq =", audio_tone_Leq) """
