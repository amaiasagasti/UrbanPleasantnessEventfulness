import pandas as pd
import json
import os
from Mosqito.loadFiles import load
from SoundLights.features import extract_ARAUS_features, extract_Freesound_features


def generate_features(audioFolderPath, csvPath, savingPath):
    """
    Function to generate most acoustic and psychoacoustic features
    found in ARAUS dataset and the signal processing features extracted
    with Freesound Feature Extractor.

    Input:
    audioFolderPath: relative path to the folder that contains the
        ARAUS augmented soundscapes (.wav files)
    csvPath: absolute path to ARAUS csv file that contains wav_gains
        generated with code found in Adequate_responses_csv
    savingPath: save where output JSON is desired to be saved

    Returns:
        output: JSON file containing all features to the corresponding
            audio files. It can be imported as a Pandas dataframe
            using import_json_to_dataframe() from dataset_functions.py
    """
    output = {}
    files_count = 0
    ARAUScsv = pd.read_csv(csvPath)

    # Find the first and last WAV files for json name
    first_wav = None
    last_wav = None

    # Go over each audio file
    files = sorted(os.listdir(audioFolderPath))
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav"):
            print("File ", file)

            # Find the first and last WAV files for json name
            if first_wav is None:
                first_wav = file
            last_wav = file

            # Find the row in ARAUS dataset that the audio filename matches to get wav gain
            audio_path = audioFolderPath + file
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
            gain = audio_info_aug["wav_gain"].values[0]

            # Add basic info about audio to dictionary
            audio_info = {}
            audio_info["info"] = {
                "file": file,
                "participant": file_participant,
                "fold": int(audio_info_aug["fold_r"].values[0]),
                "soundscape": audio_info_aug["soundscape"].values[0],
                "masker": audio_info_aug["masker"].values[0],
                "smr": int(audio_info_aug["smr"].values[0]),
                "stimulus_index": int(audio_info_aug["stimulus_index"].values[0]),
                "wav_gain": audio_info_aug["wav_gain"].values[0],
                "time_taken": audio_info_aug["time_taken"].values[0],
                "is_attention": int(audio_info_aug["is_attention"].values[0]),
                "pleasant": int(audio_info_aug["pleasant"].values[0]),
                "eventful": int(audio_info_aug["eventful"].values[0]),
                "chaotic": int(audio_info_aug["chaotic"].values[0]),
                "vibrant": int(audio_info_aug["vibrant"].values[0]),
                "uneventful": int(audio_info_aug["uneventful"].values[0]),
                "calm": int(audio_info_aug["calm"].values[0]),
                "annoying": int(audio_info_aug["annoying"].values[0]),
                "monotonous": int(audio_info_aug["monotonous"].values[0]),
                "appropriate": int(audio_info_aug["appropriate"].values[0]),
                "P_ground_truth": audio_info_aug["P_ground_truth"].values[0],
                "E_ground_truth": audio_info_aug["E_ground_truth"].values[0],
                "Leq_R_r": audio_info_aug["Leq_R_r"].values[0],
                "masker_bird": int(audio_info_aug["masker_bird"].values[0]),
                "masker_construction": int(
                    audio_info_aug["masker_construction"].values[0]
                ),
                "masker_silence": int(audio_info_aug["masker_silence"].values[0]),
                "masker_traffic": int(audio_info_aug["masker_traffic"].values[0]),
                "masker_water": int(audio_info_aug["masker_water"].values[0]),
                "masker_wind": int(audio_info_aug["masker_wind"].values[0]),
            }

            ## PSYCHOACOUSTIC FEATURES EXTRACTION ########################################################
            # Get signal
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
            # Add to dictionary
            audio_info["ARAUS"] = audio_acoustic_features
            ################################################################################################

            ## NON-PSYCHOACOUSTIC FEATURES EXTRACTION ######################################################
            # Extract features for signal
            audio_freesound_features = extract_Freesound_features(audio_path)
            # Add to dictionary
            audio_info["freesound"] = audio_freesound_features
            ################################################################################################

            # Add this audio's dict to general dictionary
            output[int(files_count)] = audio_info

            # Save info in JSON
            csv_base_name = file.split(".")[0]
            json_name = str(savingPath + str(csv_base_name) + ".json")
            with open(json_name, "w") as json_file:
                json.dump(audio_info, json_file, indent=4)

            print("Done audio ", files_count)
            files_count = files_count + 1
            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Save in json
    first_wav = first_wav.split(".")[0].split("_")
    last_wav = last_wav.split(".")[0].split("_")
    csv_base_name = (
        "f"
        + first_wav[1]
        + "p"
        + first_wav[3]
        + "s"
        + first_wav[5]
        + "_"
        + "f"
        + last_wav[1]
        + "p"
        + last_wav[3]
        + "s"
        + last_wav[5]
    )

    json_name = savingPath + "SoundsDB_" + csv_base_name + ".json"
    with open(json_name, "w") as json_file:
        json.dump(output, json_file, indent=4)

    return output


def import_json_to_dataframe(json_path: str):

    # Load the JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    # Generate column names list of strings
    for file in data:
        df_row = pd.json_normalize(data[file])
        column_names = df_row.columns.tolist()

    # Generate empty dataframe with column names
    df = pd.DataFrame(columns=column_names)

    # Add each entry in JSON to row of dataframe
    for file in data:
        df_row = pd.json_normalize(data[file])
        df = pd.concat([df, df_row])

    return df


def import_jsons_to_dataframe(jsons_path: list):
    jsons = sorted(os.listdir(jsons_path))
    dfs = []

    for json_file in jsons:
        if json_file.endswith(".json"):
            json_path = jsons_path + json_file
            print(json_path)
            # Load the JSON data
            with open(json_path, "r") as file:
                data = json.load(file)
            for file in data:
                # Add each entry in JSON to row of dataframe
                df_row = pd.json_normalize(data[file])
                dfs.append(df_row)
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    return df
