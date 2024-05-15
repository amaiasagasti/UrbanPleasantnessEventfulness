import pandas as pd
import json
import os
import numpy as np
from CLAP.src.laion_clap import CLAP_Module
from Mosqito.loadFiles import load
from SoundLights.dataset.features import (
    extract_ARAUS_features,
    extract_Freesound_features,
    extract_CLAP_embeddings,
    calculate_P_E,
)
from SoundLights.dataset.wav_files import save_wav, delete_wav


def generate_features(audioFolderPath: str, csvPath: str, savingPath: str, type: list):
    """
    Function to generate ARAUS-extended from ARAUS dataset csv (original column names)
    most acoustic and psychoacoustic features found in ARAUS dataset and the signal
    processing features extracted with Freesound Feature Extractor.

    Input:
    audioFolderPath: relative path to the folder that contains the
        ARAUS augmented soundscapes (.wav files)
    csvPath: absolute path to ARAUS csv file that contains wav_gains
        generated with code found in Adequate_responses_csv
    savingPath: save where output JSON is desired to be saved
    type: type of features to generate ["ARAUS", "Freesound"]

    Returns:
        output: ARAUS-extended JSON file containing all features to the
            corresponding audio files. It is stored in /data folder directly.
            It can be imported as a Pandas dataframe using
            import_json_to_dataframe() from dataset_functions.py.
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
                "wav_gain": float(audio_info_aug["wav_gain"].values[0]),
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
            if "ARAUS" in type:
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
            if "Freesound" in type:
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


def import_dataframe_to_json(df, save: bool, json_name: str):
    print("Converting dataframe to json")
    json_file = {}
    columns = df.columns.tolist()
    for index, row in df.iterrows():
        row_json = {}
        # For each row of the dataframe
        for column_name in columns:
            keys = column_name.split(".")
            count_keys = len(keys)
            if count_keys == 1:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                row_json[keys[0]] = row[column_name]
            if count_keys == 2:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                row_json[keys[0]][keys[1]] = row[column_name]
            elif count_keys == 3:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                row_json[keys[0]][keys[1]][keys[2]] = row[column_name]
            elif count_keys == 4:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]] = row[column_name]
            elif count_keys == 5:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                if keys[4] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = row[column_name]
            elif count_keys == 6:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                if keys[4] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = {}
                if keys[5] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][keys[5]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][keys[5]] = row[
                    column_name
                ]
        json_file[index] = row_json
    if save:
        json_path = str("data/" + str(json_name) + ".json")
        with open(json_path, "w") as file:
            json.dump(json_file, file, indent=4)


def import_jsons_to_json(jsons_path: list, save: bool, json_name: str):
    jsons = sorted(os.listdir(jsons_path))
    single_json = {}
    count = 0
    for json_file in jsons:
        if json_file.endswith(".json"):
            json_path = jsons_path + json_file
            print(json_path)
            # Load the JSON data
            with open(json_path, "r") as file:
                data = json.load(file)
                single_json[count] = data
            count = count + 1
    if save:
        with open(jsons_path + json_name, "w") as file:
            json.dump(single_json, file, indent=4)
    return single_json


def generate_features_internal(
    audioFolderPath: str, csvPath: str, savingPath: str, type: list
):
    """
    Function to generate dataset variations from ARAUS-extended dataset csv (ARAUS-extended
    column names) the desired acoustic and psychoacoustic features found in ARAUS dataset
    and the signal processing features extracted with Freesound Feature Extractor.

    Input:
    audioFolderPath: relative path to the folder that contains the
        ARAUS augmented soundscapes (.wav files)
    csvPath: absolute path to ARAUS-extended csv file
    savingPath: save where output JSON is desired to be saved
    type: type of features to generate ["ARAUS", "Freesound"]

    Returns:
        output: version/subset of ARAUS-extended JSON file containing selected features of
            the corresponding audio files. It is directly saved in /data. It can be imported
            as a Pandas dataframe using import_json_to_dataframe() from dataset_functions.py
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

            # Check if this audio had already been processed by checking if json exists
            individual_json_path = savingPath + "individual_jsons/"
            csv_base_name = file.split(".")[0]
            json_name = str(individual_json_path + str(csv_base_name) + ".json")
            if os.path.exists(json_name):
                continue

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
            audio_info_aug = ARAUScsv[ARAUScsv["info.fold"] == file_fold]
            audio_info_aug = audio_info_aug[
                audio_info_aug["info.stimulus_index"] == file_stimulus
            ]
            audio_info_aug = audio_info_aug[
                audio_info_aug["info.participant"] == file_participant
            ]
            gain = audio_info_aug["info.wav_gain"].values[0]

            # Add basic info about audio to dictionary
            audio_info = {}
            audio_info["info"] = {
                "file": file,
                "participant": file_participant,
                "fold": int(audio_info_aug["info.fold"].values[0]),
                "soundscape": audio_info_aug["info.soundscape"].values[0],
                "masker": audio_info_aug["info.masker"].values[0],
                "smr": int(audio_info_aug["info.smr"].values[0]),
                "stimulus_index": int(audio_info_aug["info.stimulus_index"].values[0]),
                "wav_gain": float(audio_info_aug["info.wav_gain"].values[0]),
                "time_taken": audio_info_aug["info.time_taken"].values[0],
                "is_attention": int(audio_info_aug["info.is_attention"].values[0]),
                "pleasant": int(audio_info_aug["info.pleasant"].values[0]),
                "eventful": int(audio_info_aug["info.eventful"].values[0]),
                "chaotic": int(audio_info_aug["info.chaotic"].values[0]),
                "vibrant": int(audio_info_aug["info.vibrant"].values[0]),
                "uneventful": int(audio_info_aug["info.uneventful"].values[0]),
                "calm": int(audio_info_aug["info.calm"].values[0]),
                "annoying": int(audio_info_aug["info.annoying"].values[0]),
                "monotonous": int(audio_info_aug["info.monotonous"].values[0]),
                "appropriate": int(audio_info_aug["info.appropriate"].values[0]),
                "P_ground_truth": audio_info_aug["info.P_ground_truth"].values[0],
                "E_ground_truth": audio_info_aug["info.E_ground_truth"].values[0],
                "Leq_R_r": audio_info_aug["info.Leq_R_r"].values[0],
                "masker_bird": int(audio_info_aug["info.masker_bird"].values[0]),
                "masker_construction": int(
                    audio_info_aug["info.masker_construction"].values[0]
                ),
                "masker_silence": int(audio_info_aug["info.masker_silence"].values[0]),
                "masker_traffic": int(audio_info_aug["info.masker_traffic"].values[0]),
                "masker_water": int(audio_info_aug["info.masker_water"].values[0]),
                "masker_wind": int(audio_info_aug["info.masker_wind"].values[0]),
            }

            ## PSYCHOACOUSTIC FEATURES EXTRACTION ########################################################
            if "ARAUS" in type:
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
            if "Freesound" in type:
                # Extract features for signal
                audio_freesound_features = extract_Freesound_features(audio_path)
                # Add to dictionary
                audio_info["freesound"] = audio_freesound_features
            ################################################################################################

            # Add this audio's dict to general dictionary
            output[int(files_count)] = audio_info

            # Save info in JSON
            if not os.path.exists(individual_json_path):
                os.makedirs(individual_json_path)
            with open(json_name, "w") as json_file:
                json.dump(audio_info, json_file, indent=4)

            print("Done audio ", files_count)
            files_count = files_count + 1
            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Save in json
    if first_wav == None:
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
        # Check if the saving directory exists, create it if it doesn't
        json_name = savingPath + "SoundsDB_" + csv_base_name + ".json"
        with open(json_name, "w") as json_file:
            json.dump(output, json_file, indent=4)

    return output


def prepare_data_models(dataframe, features_evaluated, maskers_gain: float = 5):

    # Drop string columns
    """dataframe = dataframe.drop("info.file", axis=1)
    dataframe = dataframe.drop("info.participant", axis=1)"""

    # Maskers colum, increase values
    dataframe["info.masker_bird"] = dataframe["info.masker_bird"] * maskers_gain
    dataframe["info.masker_construction"] = (
        dataframe["info.masker_construction"] * maskers_gain
    )
    dataframe["info.masker_traffic"] = dataframe["info.masker_traffic"] * maskers_gain
    dataframe["info.masker_water"] = dataframe["info.masker_water"] * maskers_gain
    dataframe["info.masker_wind"] = dataframe["info.masker_wind"] * maskers_gain

    # For fold 0, group data
    dataframe_fold0 = dataframe[dataframe["info.fold"] == 0]
    # Drop string columns
    dataframe_fold0 = dataframe_fold0.drop("info.file", axis=1)
    dataframe_fold0 = dataframe_fold0.drop("info.participant", axis=1)
    dataframe_fold0 = dataframe_fold0.groupby(
        ["info.soundscape", "info.masker", "info.smr"]
    ).mean()  # For the test set, the same 48 stimuli were shown to all participants so we take the mean of their ratings as the ground truth
    dataframe_filtered = dataframe[
        dataframe["info.fold"] != 0
    ]  # Filter rows where 'fold' column is not equal to 0
    dataframe = pd.concat(
        [dataframe_fold0, dataframe_filtered], ignore_index=True
    )  # Join together

    # Drop columns with all equal values or std=0
    std = np.std(dataframe[features_evaluated], axis=0)
    columns_to_mantain_arg = np.where(std >= 0.00001)[0]
    columns_to_drop_arg = np.where(std <= 0.00001)[0]
    columns_to_mantain = [features_evaluated[i] for i in columns_to_mantain_arg]
    columns_to_drop = [features_evaluated[i] for i in columns_to_drop_arg]
    # print(features_evaluated[np.where(std == 0)[0]])
    dataframe.drop(columns=columns_to_drop, inplace=True)

    return dataframe, columns_to_mantain


def generate_features_new_audios(
    audioFolderPath: str, csvPath: str, saving_path: str, type: list
):
    """
    IN PROCESS
    Function to generate features from any input set of audios the desired
    acoustic and psychoacoustic features found in ARAUS dataset and the signal
    processing features extracted with Freesound Feature Extractor.

    Input:
    audioFolderPath: relative path to the folder that contains the
        ARAUS augmented soundscapes (.wav files)
    csvPath: absolute path to csv file that contains gain
    savingPath: save where output JSON is desired to be saved
    type: type of features to generate ["ARAUS", "Freesound"]

    Returns:
        output: version/subset of ARAUS-extended JSON file containing selected features of
            the corresponding audio files. It is directly saved in /data. It can be imported
            as a Pandas dataframe using import_json_to_dataframe() from dataset_functions.py
    """
    output = {}
    files_count = 0
    csv_file = pd.read_csv(csvPath, delimiter=";")

    # Find the first and last WAV files for json name
    first_wav = None
    last_wav = None

    # Run only once
    if "embedding" in type:
        # Load the model
        print("------- code starts -----------")
        model = CLAP_Module(enable_fusion=True)
        print("------- clap module -----------")
        model.load_ckpt("data/models/630k-fusion-best.pt")
        print("------- model loaded -----------")

    # Go over each audio file
    files = sorted(os.listdir(audioFolderPath))
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav"):
            print("File ", file)

            # Find the first and last WAV files for json name
            if first_wav is None:
                first_wav = file
            last_wav = file

            # Check if this audio had already been processed by checking if json exists
            individual_json_path = saving_path + "individual_jsons/"
            csv_base_name = file.split(".")[0]
            json_name = str(individual_json_path + str(csv_base_name) + ".json")
            if os.path.exists(json_name):
                continue

            audio_path = audioFolderPath + file
            # Find the row in csv that the audio filename matches to get wav gain
            audio_info = csv_file[csv_file["info.file"] == file]
            gain = float(audio_info["info.wav_gain"].values[0].replace(",", "."))

            # Calculate mean Pleasantness and Eventfulness values
            P, E = calculate_P_E(audio_info)

            # Add basic info about audio to dictionary
            audio_info_json = {}
            audio_info_json["info"] = {
                "file": file,
                "fold": int(6),
                "wav_gain": gain,
                "Leq_R_r": float(
                    audio_info["info.Leq_R_r"].values[0].replace(",", ".")
                ),
                "P_ground_truth": P,
                "E_ground_truth": E,
                "masker_bird": int(audio_info["info.masker_bird"].values[0]),
                "masker_construction": int(
                    audio_info["info.masker_construction"].values[0]
                ),
                "masker_silence": int(audio_info["info.masker_silence"].values[0]),
                "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
                "masker_water": int(audio_info["info.masker_water"].values[0]),
                "masker_wind": int(audio_info["info.masker_wind"].values[0]),
            }

            audio_r, fs = load(audio_path, wav_calib=gain, ch=1)  # R
            audio_l, fs = load(audio_path, wav_calib=gain, ch=0)  # L

            if "Freesound" or "embedding" in type:
                # Normalisation gain to avoid a lot of clipping (because audio variables
                # are in Pascal peak measure, we need "digital version")
                norm_gain = 6.44
                # Apply norm gain to audio
                adapted_audio_r = audio_r / norm_gain
                adapted_audio_l = audio_l / norm_gain
                adapted_signal = np.column_stack((adapted_audio_l, adapted_audio_r))
                max_gain = np.max(adapted_audio_r)
                min_gain = np.min(adapted_audio_r)
                # Clipping?
                if max_gain > 1 or min_gain < -1:
                    adapted_signal = np.clip(adapted_signal, -1, 1)
                # Save audio provisionally
                provisional_savingPath = saving_path + "provisional/"
                if not os.path.exists(provisional_savingPath):
                    os.makedirs(provisional_savingPath)
                provisional_saving_path_complete = provisional_savingPath + file
                save_wav(adapted_signal, fs, provisional_saving_path_complete)
                # This audio is used to generate Freesound or CLAP embedding group features

            ## PSYCHOACOUSTIC FEATURES EXTRACTION ########################################################
            if "ARAUS" in type:
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
                audio_acoustic_features = extract_ARAUS_features(audio_r, fs, list)
                # Add to dictionary
                audio_info_json["ARAUS"] = audio_acoustic_features
            ################################################################################################

            ## NON-PSYCHOACOUSTIC FEATURES EXTRACTION ######################################################
            if "Freesound" in type:
                # Extract features for signal
                audio_freesound_features = extract_Freesound_features(
                    provisional_saving_path_complete
                )
                # Add to dictionary
                audio_info_json["freesound"] = audio_freesound_features

            ################################################################################################

            ## EMBEDDING EXTRACTION ########################################################################
            if "embedding" in type:
                embedding = extract_CLAP_embeddings(
                    provisional_saving_path_complete, model
                )
                audio_info_json["CLAP"] = embedding
            ################################################################################################

            # Delete provisional audio
            if os.path.exists(provisional_savingPath):
                delete_wav(provisional_saving_path_complete)

            # Add this audio's dict to general dictionary
            output[int(files_count)] = audio_info_json

            # Save info in individual JSON for current audio
            if not os.path.exists(individual_json_path):
                os.makedirs(individual_json_path)
            with open(json_name, "w") as json_file:
                json.dump(audio_info_json, json_file, indent=4)

            print("Done audio ", files_count)
            files_count = files_count + 1
            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Save in json
    if first_wav != None:
        first_wav = first_wav.split(".")[0]
        last_wav = last_wav.split(".")[0]
        csv_base_name = "from_" + first_wav + "_to_" + last_wav
        # Check if the saving directory exists, create it if it doesn't
        json_name = saving_path + "Sounds_" + csv_base_name + ".json"
        with open(json_name, "w") as json_file:
            json.dump(output, json_file, indent=4)

    return output


def file_origin_info(file, participant, gain, audio_info, origin):
    audio_info_json = {}

    if origin == "new_data":
        # Calculate mean Pleasantness and Eventfulness values
        P, E = calculate_P_E(audio_info)
        # Add basic info about audio to dictionary
        audio_info_json["info"] = {
            "file": file,
            "fold": int(6),
            "wav_gain": gain,
            "Leq_R_r": float(audio_info["info.Leq_R_r"].values[0].replace(",", ".")),
            "P_ground_truth": P,
            "E_ground_truth": E,
            "masker_bird": int(audio_info["info.masker_bird"].values[0]),
            "masker_construction": int(
                audio_info["info.masker_construction"].values[0]
            ),
            "masker_silence": int(audio_info["info.masker_silence"].values[0]),
            "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
            "masker_water": int(audio_info["info.masker_water"].values[0]),
            "masker_wind": int(audio_info["info.masker_wind"].values[0]),
        }
    elif origin == "ARAUS_original":
        audio_info_json["info"] = {
            "file": file,
            "participant": participant,
            "fold": int(audio_info["fold_r"].values[0]),
            "soundscape": audio_info["soundscape"].values[0],
            "masker": audio_info["masker"].values[0],
            "smr": int(audio_info["smr"].values[0]),
            "stimulus_index": int(audio_info["stimulus_index"].values[0]),
            "wav_gain": float(audio_info["wav_gain"].values[0]),
            "time_taken": audio_info["time_taken"].values[0],
            "is_attention": int(audio_info["is_attention"].values[0]),
            "pleasant": int(audio_info["pleasant"].values[0]),
            "eventful": int(audio_info["eventful"].values[0]),
            "chaotic": int(audio_info["chaotic"].values[0]),
            "vibrant": int(audio_info["vibrant"].values[0]),
            "uneventful": int(audio_info["uneventful"].values[0]),
            "calm": int(audio_info["calm"].values[0]),
            "annoying": int(audio_info["annoying"].values[0]),
            "monotonous": int(audio_info["monotonous"].values[0]),
            "appropriate": int(audio_info["appropriate"].values[0]),
            "P_ground_truth": audio_info["P_ground_truth"].values[0],
            "E_ground_truth": audio_info["E_ground_truth"].values[0],
            "Leq_R_r": audio_info["Leq_R_r"].values[0],
            "masker_bird": int(audio_info["masker_bird"].values[0]),
            "masker_construction": int(audio_info["masker_construction"].values[0]),
            "masker_silence": int(audio_info["masker_silence"].values[0]),
            "masker_traffic": int(audio_info["masker_traffic"].values[0]),
            "masker_water": int(audio_info["masker_water"].values[0]),
            "masker_wind": int(audio_info["masker_wind"].values[0]),
        }
    elif origin == "ARAUS_extended":
        # Add basic info about audio to dictionary
        audio_info_json["info"] = {
            "file": file,
            "participant": participant,
            "fold": int(audio_info["info.fold"].values[0]),
            "soundscape": audio_info["info.soundscape"].values[0],
            "masker": audio_info["info.masker"].values[0],
            "smr": int(audio_info["info.smr"].values[0]),
            "stimulus_index": int(audio_info["info.stimulus_index"].values[0]),
            "wav_gain": float(audio_info["info.wav_gain"].values[0]),
            "time_taken": audio_info["info.time_taken"].values[0],
            "is_attention": int(audio_info["info.is_attention"].values[0]),
            "pleasant": int(audio_info["info.pleasant"].values[0]),
            "eventful": int(audio_info["info.eventful"].values[0]),
            "chaotic": int(audio_info["info.chaotic"].values[0]),
            "vibrant": int(audio_info["info.vibrant"].values[0]),
            "uneventful": int(audio_info["info.uneventful"].values[0]),
            "calm": int(audio_info["info.calm"].values[0]),
            "annoying": int(audio_info["info.annoying"].values[0]),
            "monotonous": int(audio_info["info.monotonous"].values[0]),
            "appropriate": int(audio_info["info.appropriate"].values[0]),
            "P_ground_truth": audio_info["info.P_ground_truth"].values[0],
            "E_ground_truth": audio_info["info.E_ground_truth"].values[0],
            "Leq_R_r": audio_info["info.Leq_R_r"].values[0],
            "masker_bird": int(audio_info["info.masker_bird"].values[0]),
            "masker_construction": int(
                audio_info["info.masker_construction"].values[0]
            ),
            "masker_silence": int(audio_info["info.masker_silence"].values[0]),
            "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
            "masker_water": int(audio_info["info.masker_water"].values[0]),
            "masker_wind": int(audio_info["info.masker_wind"].values[0]),
        }
    return audio_info_json


def generate_features_general(
    audioFolderPath: str,
    csvPath: str,
    saving_path: str,
    type: list,
    origin: str,
    norm_gain: float = 1,
):
    """
    IN PROCESS
    Function to generate features from any input set of audios the desired
    acoustic and psychoacoustic features found in ARAUS dataset and the signal
    processing features extracted with Freesound Feature Extractor.

    Input:
    audioFolderPath: relative path to the folder that contains the
        ARAUS augmented soundscapes (.wav files)
    csvPath: absolute path to csv file that contains gain
    savingPath: save where output JSON is desired to be saved
    type: type of features to generate ["ARAUS", "Freesound"]

    Returns:
        output: version/subset of ARAUS-extended JSON file containing selected features of
            the corresponding audio files. It is directly saved in /data. It can be imported
            as a Pandas dataframe using import_json_to_dataframe() from dataset_functions.py
    """
    output = {}
    files_count = 0
    # CHANGE READ OUTSIDE !!!!!!
    csv_file = pd.read_csv(csvPath, delimiter=";")

    # Find the first and last WAV files for json name
    first_wav = None
    last_wav = None

    # Run only once
    if "embedding" in type:
        # Load the model
        print("------- code starts -----------")
        model = CLAP_Module(enable_fusion=True)
        print("------- clap module -----------")
        model.load_ckpt("data/models/630k-fusion-best.pt")
        print("------- model loaded -----------")

    # Go over each audio file
    files = sorted(os.listdir(audioFolderPath))
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav"):
            print("File ", file)

            # Find the first and last WAV files for json name
            if first_wav is None:
                first_wav = file
            last_wav = file

            # Check if this audio had already been processed by checking if json exists
            individual_json_path = saving_path + "individual_jsons/"
            csv_base_name = file.split(".")[0]
            json_name = str(individual_json_path + str(csv_base_name) + ".json")
            if os.path.exists(json_name):
                continue

            audio_path = audioFolderPath + file
            # Find the row in csv that the audio filename matches to get wav gain
            if origin == "new_data":
                audio_info = csv_file[csv_file["info.file"] == file]
                gain = float(audio_info["info.wav_gain"].values[0].replace(",", "."))
                participant = "mean_of_all"
            elif origin == "ARAUS_extended":
                file_split = file.split("_")
                file_fold = int(file_split[1])
                participant = "ARAUS_" + file_split[3]
                file_stimulus = int(file_split[5].split(".")[0])
                audio_info = csv_file[csv_file["info.fold"] == file_fold]
                audio_info = audio_info[
                    audio_info["info.stimulus_index"] == file_stimulus
                ]
                audio_info = audio_info[audio_info["info.participant"] == participant]
                gain = audio_info["info.wav_gain"].values[0]
            elif origin == "ARAUS_original":
                file_split = file.split("_")
                file_fold = int(file_split[1])
                participant = "ARAUS_" + file_split[3]
                file_stimulus = int(file_split[5].split(".")[0])
                audio_info = csv_file[csv_file["fold_r"] == file_fold]
                audio_info = audio_info[audio_info["stimulus_index"] == file_stimulus]
                audio_info = audio_info[audio_info["participant"] == participant]
                gain = audio_info["wav_gain"].values[0]

            audio_info = file_origin_info(file, participant, gain, audio_info, origin)
            """ # Calculate mean Pleasantness and Eventfulness values
            P, E = calculate_P_E(audio_info)

            # Add basic info about audio to dictionary
            audio_info_json = {}
            audio_info_json["info"] = {
                "file": file,
                "fold": int(6),
                "wav_gain": gain,
                "Leq_R_r": float(
                    audio_info["info.Leq_R_r"].values[0].replace(",", ".")
                ),
                "P_ground_truth": P,
                "E_ground_truth": E,
                "masker_bird": int(audio_info["info.masker_bird"].values[0]),
                "masker_construction": int(
                    audio_info["info.masker_construction"].values[0]
                ),
                "masker_silence": int(audio_info["info.masker_silence"].values[0]),
                "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
                "masker_water": int(audio_info["info.masker_water"].values[0]),
                "masker_wind": int(audio_info["info.masker_wind"].values[0]),
            } """

            audio_r, fs = load(audio_path, wav_calib=gain, ch=1)  # R
            audio_l, fs = load(audio_path, wav_calib=gain, ch=0)  # L

            if "Freesound" or "embedding" in type:
                # Normalisation gain to avoid a lot of clipping (because audio variables
                # are in Pascal peak measure, we need "digital version")
                adapted_audio_r = audio_r / norm_gain
                adapted_audio_l = audio_l / norm_gain
                adapted_signal = np.column_stack((adapted_audio_l, adapted_audio_r))
                max_gain = np.max(adapted_audio_r)
                min_gain = np.min(adapted_audio_r)
                # Clipping?
                if max_gain > 1 or min_gain < -1:
                    adapted_signal = np.clip(adapted_signal, -1, 1)
                # Save audio provisionally
                provisional_savingPath = saving_path + "provisional/"
                if not os.path.exists(provisional_savingPath):
                    os.makedirs(provisional_savingPath)
                provisional_saving_path_complete = provisional_savingPath + file
                save_wav(adapted_signal, fs, provisional_saving_path_complete)
                # This audio is used to generate Freesound or CLAP embedding group features

            ## PSYCHOACOUSTIC FEATURES EXTRACTION ########################################################
            if "ARAUS" in type:
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
                audio_acoustic_features = extract_ARAUS_features(audio_r, fs, list)
                # Add to dictionary
                audio_info["ARAUS"] = audio_acoustic_features
            ################################################################################################

            ## NON-PSYCHOACOUSTIC FEATURES EXTRACTION ######################################################
            if "Freesound" in type:
                # Extract features for signal
                audio_freesound_features = extract_Freesound_features(
                    provisional_saving_path_complete
                )
                # Add to dictionary
                audio_info["freesound"] = audio_freesound_features

            ################################################################################################

            ## EMBEDDING EXTRACTION ########################################################################
            if "embedding" in type:
                embedding = extract_CLAP_embeddings(
                    provisional_saving_path_complete, model
                )
                audio_info["CLAP"] = embedding
            ################################################################################################

            # Delete provisional audio
            if os.path.exists(provisional_savingPath):
                delete_wav(provisional_saving_path_complete)

            # Add this audio's dict to general dictionary
            output[int(files_count)] = audio_info

            # Save info in individual JSON for current audio
            if not os.path.exists(individual_json_path):
                os.makedirs(individual_json_path)
            with open(json_name, "w") as json_file:
                json.dump(audio_info, json_file, indent=4)

            print("Done audio ", files_count)
            files_count = files_count + 1
            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Save in json
    if first_wav != None:
        first_wav = first_wav.split(".")[0]
        last_wav = last_wav.split(".")[0]
        csv_base_name = "from_" + first_wav + "_to_" + last_wav
        # Check if the saving directory exists, create it if it doesn't
        json_name = saving_path + "Sounds_" + csv_base_name + ".json"
        with open(json_name, "w") as json_file:
            json.dump(output, json_file, indent=4)

    return output
