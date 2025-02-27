"""
This script contains a set of functions to create the new dataset 'ARAUS extension'. 

generate_features() is the main function, the other functions adapt the output of 
the first function for re-use or processing.
"""

import pandas as pd
import json
import os
import numpy as np
import time
from CLAP.src.laion_clap import CLAP_Module
from lib.dataset.auxiliary_functions import load
from lib.dataset.features import (
    extract_ARAUS_features,
    extract_Freesound_features,
    extract_CLAP_embeddings,
    calculate_P_E,
)
from lib.dataset.features_groups import (
    ARAUS_features,
    Freesound_features,
    masker_features,
    clap_features,
)
from lib.dataset.wav_files import save_wav, delete_wav


def import_json_to_dataframe(json_path: str, save: bool, saving_path: str):
    """
    Import data from a JSON file and transform it into a DataFrame.

    This function reads a JSON file containing a dataset generated using generate_features()
    from the specified path and converts its content into a pandas DataFrame. Optionally,
    the resulting DataFrame can be saved to a specified path as a CSV file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file to be imported.
    save : bool
        If True, the DataFrame will be saved to the specified saving path.
    saving_path : str
        Path where the DataFrame will be saved if the save parameter is True.

    Outputs
    -------
    dataframe : pandas.DataFrame
        A DataFrame containing the data imported from the JSON file.
    """

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

    if save:
        df.to_csv(saving_path, index=False)
    return df


def import_jsons_to_dataframe(jsons_path: list, save: bool, saving_path: str):
    """
    Import data from a list of JSON files and combine them into a single DataFrame.

    This function reads multiple JSON files (each containing the features generated with
    generate_features() for a single audio file) from the specified list of paths, all
    of which shares the same keys and format, and combines their content into a single
    pandas DataFrame. Optionally, the resulting DataFrame can be saved to a specified path
    as a CSV file.

    Parameters
    ----------
    jsons_path : list of str
        List of paths to the JSON files to be imported. Each JSON file should have the same
        structure and keys.
    save : bool
        If True, the combined DataFrame will be saved to the specified saving path.
    saving_path : str
        Path where the DataFrame will be saved if the save parameter is True.

    Outputs:
    -------
    df : pandas.DataFrame
        A DataFrame containing the combined data from all the JSON files.
    """
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

    if save:
        df.to_csv(saving_path, index=False)
    return df


def import_dataframe_to_json(df, save: bool, saving_path: str):
    """
    Convert a DataFrame to JSON format and optionally save it to a file.

    This function converts a pandas DataFrame containing the dataset of features into a
    JSON format. If specified, the JSON data can be saved to a file at the given path.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be converted to JSON format.
    save : bool
        If True, the JSON data will be saved to the specified saving path.
    saving_path : str
        Path where the JSON file will be saved if the save parameter is True.

    Outputs
    -------
    None
        This function does not return a value. The DataFrame is converted to JSON and
        optionally saved to a file.
    """

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
        with open(saving_path, "w") as file:
            json.dump(json_file, file, indent=4)


def import_jsons_to_json(jsons_path: list, save: bool, saving_path: str):
    """
    Combine multiple JSON files into a single JSON file.

    This function reads multiple JSON files (each containing the features generated with
    generate_features() for a single audio file) from the specified list of paths, each of which
    shares the same keys and format, and combines their content into a single JSON object.
    Optionally, the combined JSON data can be saved to a specified path.

    Parameters
    ----------
    jsons_path : list of str
        List of paths to the JSON files to be combined. Each JSON file should have the same
        structure and keys.
    save : bool
        If True, the combined JSON data will be saved to the specified saving path.
    saving_path : str
        Path where the combined JSON file will be saved if the save parameter is True.

    Returns
    -------
    single_json : dict
        A dictionary containing the combined data from all the JSON files.
    """

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
        with open(saving_path, "w") as file:
            json.dump(single_json, file, indent=4)
    return single_json


def file_origin_info(file, participant, gain, audio_info, origin):
    """
    Adapts metadata information from the dataset when generate_features() is called.

    We can work with three different origins of data to generate a dataset with generate_features():
    - "new_data": new audios not found in ARAUS dataset. CSV file of metadata only contains Leq,
        wav_gain, punctuation from listening tests and maskers information.
    - "ARAUS_original": original audios found in ARAUS dataset. CSV file of metadata contains the very
        same content as the original (constitutes responses_SoundLights.csv).
    - "ARAUS_extended": once a new dataset has been generated using generate_features(), any changes on this
        file, is done with this option.
    This function is called when generate_features() is called. It reads input metadata and adapts it depending
    on the data origin so that the resulting datastets, regardless of the origin, have the same metadata format.

    Parameters
    ----------
    file : str
        Name of file.
    participant : str
        Participant information that labelled current file.
    gain : float
        wav_gain information that transforms digital signal to peak-Pascals signal.
    audio_info: dict
        Dictionary with metadata of current file. Its keys depend on the origin.
    origin: str
        Possible origin of file (see explanation above).

    Returns
    -------
    audio_info_json : dict
        Adapted metadata dictionary.
    """

    audio_info_json = {}

    if origin == "new_data":
        # Calculate mean Pleasantness and Eventfulness values
        P, E = calculate_P_E(audio_info)
        # Add basic info about audio to dictionary
        audio_info_json["info"] = {
            "file": file,
            "fold": int(6),
            "wav_gain": float(gain),
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
            "wav_gain": float(gain),
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
            "wav_gain": float(gain),
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


def generate_features(
    audioFolderPath: str,
    csv_file: pd.DataFrame,
    saving_path: str,
    type: list,
    origin: str,
    norm_gain: float = 1,
    variation_gain: float = 1,
):
    """
    Function to generate features from any input set of audios. Specifically:
    - 'ARAUS' features: Statistics of the acoustic and psychoacoustic features used in
        ARAUS dataset.
    - 'Freesound' features: Statistics of a set of audio signal analysis features
        generated with FreesoundExtractor function from Essentia Library
    - 'CLAP' features: CLAP embeddings generated with LAION-AI's CLAP model.

    Parameters
    ----------
    audioFolderPath: str
        Relative path to the folder that contains the audios (.wav files)
    csv_file: pandas.Dataframe
        File that contains metadata information of the audios
    saving_path: str
        Saving path where output JSON or dataset is desired to be saved
    type: list
        Type of features to generate ["ARAUS", "Freesound", "embedding"]
    origin: str
        Origin of the generated features "new_data", "ARAUS_original", "ARAUS_extended"

    Outputs:
        output: JSON file / dictionary containing selected features of the corresponding audio files.
        It is saved in automatically in the specified path. Output can be imported as a Pandas dataframe
        using import_json_to_dataframe() function.
    """
    output = {}
    files_count = 0
    duration_time_ARAUS = 0
    duration_time_Freesound = 0
    duration_time_CLAP = 0

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

            # if variation_gain code= 9999 means that variation gain should be a random number
            if variation_gain == 9999:
                variation_gain = np.random.uniform(1, 10)

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

            audio_info = file_origin_info(
                file, participant, gain * variation_gain, audio_info, origin
            )

            audio_r, fs = load(audio_path, wav_calib=gain * variation_gain, ch=1)  # R
            audio_l, fs = load(audio_path, wav_calib=gain * variation_gain, ch=0)  # L

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
                start_time_ARAUS = time.time()
                audio_acoustic_features = extract_ARAUS_features(audio_r, fs, list)
                duration_time_ARAUS = time.time() - start_time_ARAUS
                # Add to dictionary
                audio_info["ARAUS"] = audio_acoustic_features
            ################################################################################################

            ## NON-PSYCHOACOUSTIC FEATURES EXTRACTION ######################################################
            if "Freesound" in type:
                # Extract features for signal
                start_time_Freesound = time.time()
                audio_freesound_features = extract_Freesound_features(
                    provisional_saving_path_complete
                )
                duration_time_Freesound = time.time() - start_time_Freesound
                # Add to dictionary
                audio_info["freesound"] = audio_freesound_features

            ################################################################################################

            ## EMBEDDING EXTRACTION ########################################################################
            if "embedding" in type:
                start_time_CLAP = time.time()
                embedding = extract_CLAP_embeddings(
                    provisional_saving_path_complete, model
                )
                duration_time_CLAP = time.time() - start_time_CLAP
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

    print("TIME STATISTICS ")
    print(f"ARAUS features {duration_time_ARAUS} seconds")
    print(f"Freesound features {duration_time_Freesound} seconds")
    print(f"CLAP features {duration_time_CLAP} seconds")

    return output


def expand_CLAP_features(df):
    """
    Expand the 'CLAP' column in the DataFrame into multiple columns.

    This function processes a DataFrame where one of the columns, 'CLAP', contains a vector
    of numbers. The vector is split into individual components, with each component being placed
    into a new column. This transformation allows for more manageable and accessible data for
    subsequent analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame or dataset. Each entry in the 'CLAP' column is a list or array-like structure
        with numerical values, generated using generate_features().

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the 'CLAP' column expanded into multiple columns. Each component of the
        original 'CLAP' vectors is now in its own separate column (column names determined in
        'clap_features' variable)
    """

    all_columns = (
        ARAUS_features
        + Freesound_features
        + masker_features
        + ["info.P_ground_truth", "info.E_ground_truth"]
        + clap_features
    )

    full_list = []
    for index, row in df.iterrows():
        string_list = row["CLAP"].split("[")[1].split("]")[0].split(",")
        clap_list = [float(item) for item in string_list]
        complete_new_row = (
            list(
                row[
                    ARAUS_features
                    + Freesound_features
                    + masker_features
                    + ["info.P_ground_truth", "info.E_ground_truth"]
                ].values
            )
            + clap_list
        )
        full_list.append(complete_new_row)
    df = pd.DataFrame(data=full_list, columns=all_columns)
    return df
