import sys
from Mosqito.loadFiles import load
from SoundLights.features import extract_ARAUS_features
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import essentia.standard as es

sys.path.append("..")

data_augmented = pd.read_csv("data/responses_SoundLights.csv")

column_names = [
    "participant",
    "fold_r",
    "soundscape",
    "masker",
    "smr",
    "stimulus_index",
    "wav_gain",
    "time_taken",
    "is_attention",
    "pleasant",
    "eventful",
    "chaotic",
    "vibrant",
    "uneventful",
    "calm",
    "annoying",
    "monotonous",
    "appropriate",
    "Savg_r",
    "Smax_r",
    "S05_r",
    "S10_r",
    "S20_r",
    "S30_r",
    "S40_r",
    "S50_r",
    "S60_r",
    "S70_r",
    "S80_r",
    "S90_r",
    "S95_r",
    "Navg_r",
    "Nrmc_r",
    "Nmax_r",
    "N05_r",
    "N10_r",
    "N20_r",
    "N30_r",
    "N40_r",
    "N50_r",
    "N60_r",
    "N70_r",
    "N80_r",
    "N90_r",
    "N95_r",
    "Favg_r",
    "Fmax_r",
    "F05_r",
    "F10_r",
    "F20_r",
    "F30_r",
    "F40_r",
    "F50_r",
    "F60_r",
    "F70_r",
    "F80_r",
    "F90_r",
    "F95_r",
    "LAavg_r",
    "LAmin_r",
    "LAmax_r",
    "LA05_r",
    "LA10_r",
    "LA20_r",
    "LA30_r",
    "LA40_r",
    "LA50_r",
    "LA60_r",
    "LA70_r",
    "LA80_r",
    "LA90_r",
    "LA95_r",
    "LCavg_r",
    "LCmin_r",
    "LCmax_r",
    "LC05_r",
    "LC10_r",
    "LC20_r",
    "LC30_r",
    "LC40_r",
    "LC50_r",
    "LC60_r",
    "LC70_r",
    "LC80_r",
    "LC90_r",
    "LC95_r",
    "Ravg_r",
    "Rmax_r",
    "R05_r",
    "R10_r",
    "R20_r",
    "R30_r",
    "R40_r",
    "R50_r",
    "R60_r",
    "R70_r",
    "R80_r",
    "R90_r",
    "R95_r",
    "Tgavg_r",
    "Tavg_r",
    "Tmax_r",
    "T05_r",
    "T10_r",
    "T20_r",
    "T30_r",
    "T40_r",
    "T50_r",
    "T60_r",
    "T70_r",
    "T80_r",
    "T90_r",
    "T95_r",
    "M00005_0_r",
    "M00006_3_r",
    "M00008_0_r",
    "M00010_0_r",
    "M00012_5_r",
    "M00016_0_r",
    "M00020_0_r",
    "M00025_0_r",
    "M00031_5_r",
    "M00040_0_r",
    "M00050_0_r",
    "M00063_0_r",
    "M00080_0_r",
    "M00100_0_r",
    "M00125_0_r",
    "M00160_0_r",
    "M00200_0_r",
    "M00250_0_r",
    "M00315_0_r",
    "M00400_0_r",
    "M00500_0_r",
    "M00630_0_r",
    "M00800_0_r",
    "M01000_0_r",
    "M01250_0_r",
    "M01600_0_r",
    "M02000_0_r",
    "M02500_0_r",
    "M03150_0_r",
    "M04000_0_r",
    "M05000_0_r",
    "M06300_0_r",
    "M08000_0_r",
    "M10000_0_r",
    "M12500_0_r",
    "M16000_0_r",
    "M20000_0_r",
    "Leq_R_r",
    "masker_bird",
    "masker_construction",
    "masker_silence",
    "masker_traffic",
    "masker_water",
    "masker_wind",
    "P_ground_truth",
    "E_ground_truth",
]

new_df = pd.DataFrame(columns=column_names)

# For each audio...
audios_path = "data/soundscapes_augmented/"
for file in sorted(os.listdir(audios_path)):
    if file.endswith(".mp3") or file.endswith(".wav"):

        print("File ", file)
        audio_path = audios_path + file

        # Find the row in ARAUS dataset that the audio filename matches
        file_split = file.split("_")
        file_fold = int(file_split[1])
        file_participant = "ARAUS_" + file_split[3]
        file_stimulus = int(file_split[5].split(".")[0])
        audio_info_aug = data_augmented[data_augmented["fold_r"] == file_fold]
        audio_info_aug = audio_info_aug[
            audio_info_aug["stimulus_index"] == file_stimulus
        ]
        audio_info_aug = audio_info_aug[
            audio_info_aug["participant"] == file_participant
        ]  # Row of info from the ARAUS csv that corresponds to the evaluated soundscape_augmented audio

        # Add info about audio to add dataframe
        output_row = {}
        output_row["participant"] = file_participant
        output_row["fold_r"] = audio_info_aug["fold_r"].values[0]
        output_row["soundscape"] = audio_info_aug["soundscape"].values[0]
        output_row["masker"] = audio_info_aug["masker"].values[0]
        output_row["smr"] = audio_info_aug["smr"].values[0]
        output_row["stimulus_index"] = audio_info_aug["stimulus_index"].values[0]
        output_row["wav_gain"] = audio_info_aug["wav_gain"].values[0]
        output_row["time_taken"] = audio_info_aug["time_taken"].values[0]
        output_row["is_attention"] = audio_info_aug["is_attention"].values[0]
        output_row["pleasant"] = audio_info_aug["pleasant"].values[0]
        output_row["eventful"] = audio_info_aug["eventful"].values[0]
        output_row["chaotic"] = audio_info_aug["chaotic"].values[0]
        output_row["vibrant"] = audio_info_aug["vibrant"].values[0]
        output_row["uneventful"] = audio_info_aug["uneventful"].values[0]
        output_row["calm"] = audio_info_aug["calm"].values[0]
        output_row["annoying"] = audio_info_aug["annoying"].values[0]
        output_row["monotonous"] = audio_info_aug["monotonous"].values[0]
        output_row["appropriate"] = audio_info_aug["appropriate"].values[0]
        output_row["P_ground_truth"] = audio_info_aug["P_ground_truth"].values[0]
        output_row["E_ground_truth"] = audio_info_aug["E_ground_truth"].values[0]
        output_row["Leq_R_r"] = audio_info_aug["Leq_R_r"].values[0]
        output_row["masker_bird"] = audio_info_aug["masker_bird"].values[0]
        output_row["masker_construction"] = audio_info_aug[
            "masker_construction"
        ].values[0]
        output_row["masker_silence"] = audio_info_aug["masker_silence"].values[0]
        output_row["masker_traffic"] = audio_info_aug["masker_traffic"].values[0]
        output_row["masker_water"] = audio_info_aug["masker_water"].values[0]
        output_row["masker_wind"] = audio_info_aug["masker_wind"].values[0]

        # Get signal
        gain = audio_info_aug["wav_gain"].values[0]
        signalR, fs = load(
            audio_path, wav_calib=gain, ch=1
        )  # R - SINGLE CHANNEL SIGNAL

        # Extract psychoacoustic features for signal
        list = [
            "loudness",
            "sharpness",
            "LA",
            "LC",
            "frequency",
            "roughness",
            "fluctuation",
        ]
        audio_acoustic_features = extract_ARAUS_features(signalR, fs, list)

        # Add acoustic features to output row
        output_row.update(audio_acoustic_features)

        # Save
        new_df = pd.concat(
            [new_df, pd.DataFrame([output_row])], ignore_index=True
        )  # Save row of data

        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")


new_df.to_csv("data/generated_features_SoundLights.csv", index=False)
