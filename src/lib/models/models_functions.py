"""
This script contains a collection of functions used for model related operations.

The purpose of this script is to provide a comprehensive set of utilities and methods that facilitate the entire machine learning workflow. It includes functions for:
- Finding and tuning hyperparameters to optimize model performance.
- Training models using different algorithms and configurations.
- Saving and loading model states and configurations for reproducibility.
- Testing and evaluating model performance on validation or test datasets.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
from joblib import dump, load
import copy
import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.dataset.features_groups import (
    general_info,
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
    clap_features,
)

def prepare_features_models(
    dataframe,
    features_evaluated,
    masker_transform: str = "None",
    maskers_gain: float = 1,
):

    # Maskers colum, increase values
    if masker_transform == "-1,1":
        dataframe["info.masker_bird"] = (
            dataframe["info.masker_bird"] * 2 - 1
        ) * maskers_gain
        dataframe["info.masker_construction"] = (
            dataframe["info.masker_construction"] * 2 - 1
        ) * maskers_gain
        dataframe["info.masker_traffic"] = (
            dataframe["info.masker_traffic"] * 2 - 1
        ) * maskers_gain
        dataframe["info.masker_silence"] = (
            dataframe["info.masker_silence"] * 2 - 1
        ) * maskers_gain
        dataframe["info.masker_water"] = (
            dataframe["info.masker_water"] * 2 - 1
        ) * maskers_gain
        dataframe["info.masker_wind"] = (
            dataframe["info.masker_wind"] * 2 - 1
        ) * maskers_gain
    else:
        dataframe["info.masker_bird"] = (dataframe["info.masker_bird"]) * maskers_gain
        dataframe["info.masker_construction"] = (
            dataframe["info.masker_construction"] * maskers_gain
        )
        dataframe["info.masker_traffic"] = (
            dataframe["info.masker_traffic"] * maskers_gain
        )
        dataframe["info.masker_silence"] = (
            dataframe["info.masker_silence"] * maskers_gain
        )
        dataframe["info.masker_water"] = dataframe["info.masker_water"] * maskers_gain
        dataframe["info.masker_wind"] = dataframe["info.masker_wind"] * maskers_gain

    # For fold 0, group data
    dataframe_fold0 = dataframe[dataframe["info.fold"] == 0]
    # Drop string columns
    dataframe_fold0 = dataframe_fold0.drop("info.file", axis=1)
    dataframe_fold0 = dataframe_fold0.drop("info.participant", axis=1)
    dataframe_fold0 = dataframe_fold0.groupby(
        ["info.soundscape", "info.masker", "info.smr"]
    ).mean()  # .reset_index()  # For the test set, the same 48 stimuli were shown to all participants so we take the mean of their ratings as the ground truth
    dataframe_filtered = dataframe[
        dataframe["info.fold"] != 0
    ]  # Filter rows where 'fold' column is not equal to 0
    dataframe = pd.concat(
        [dataframe_fold0, dataframe_filtered], ignore_index=True
    )  # Join together

    # Drop columns with all equal values or std=0
    std = np.std(dataframe[features_evaluated], axis=0)
    columns_to_mantain_arg = np.where(std >= 0.00001)[0]
    columns_to_drop_arg = np.where(std < 0.00001)[0]
    columns_to_mantain = [features_evaluated[i] for i in columns_to_mantain_arg]
    columns_to_drop = [features_evaluated[i] for i in columns_to_drop_arg]
    dataframe.drop(columns=columns_to_drop, inplace=True)

    return dataframe, columns_to_mantain


def prepare_dataframes_models(data_ARAUS_path, data_foldFs_path, saving_folder, feature_set:str):

    ############# PREPARE DATA #########################################################
    df = pd.read_csv(data_ARAUS_path)

    if feature_set=="ARAUS":
        # ARAUS features dataframe
        df_ARAUS = df[general_info + ARAUS_features]
    elif feature_set=="Freesound":
        # Freesound features dataframe
        df_Freesound = df[general_info + Freesound_features]
    elif feature_set=="CLAP":
        # CLAP embeddings dataframe
        df_clap = df[general_info + ["CLAP"]]
        all_columns = general_info + clap_features
        full_list = []
        for index, row in df_clap.iterrows():
            string_list = row["CLAP"].split("[")[2].split("]")[0].split(",")
            clap_list = [float(item) for item in string_list]
            complete_new_row = list(row[general_info].values) + clap_list
            full_list.append(complete_new_row)
        df_clap = pd.DataFrame(data=full_list, columns=all_columns)

    # Fold Fs preparation
    df_Fs = pd.read_csv(data_foldFs_path)
    df_foldFs = df_Fs[
        ARAUS_features
        + Freesound_features
        + masker_features
        + ["info.P_ground_truth", "info.E_ground_truth", "CLAP"]
    ]
    all_columns = (
        ARAUS_features
        + Freesound_features
        + masker_features
        + ["info.P_ground_truth", "info.E_ground_truth"]
        + clap_features
    )
    # Add and adapt CLAP features
    full_list = []
    for index, row in df_foldFs.iterrows():
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
    df_foldFs = pd.DataFrame(data=full_list, columns=all_columns)

    # Saving folder
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # returns
    if feature_set=="ARAUS":
        return df_ARAUS, ARAUS_features, df_foldFs 
    elif feature_set=="Freesound":
        return df_Freesound, Freesound_features, df_foldFs 
    elif feature_set=="CLAP":
        return df_clap, clap_features, df_foldFs 


def clip(x, x_min=-1, x_max=1):
    """
    Clips the input vector `x` to be within the range `[x_min, x_max]`.

    Args:
        x (numpy.ndarray): Input vector to be clipped.
        x_min (float, optional): Minimum value for clipping. Defaults to -1.
        x_max (float, optional): Maximum value for clipping. Defaults to 1.

    Returns:
        numpy.ndarray: Clipped vector.
    """
    clipped_x = np.where(x < x_min, x_min, x)
    clipped_x = np.where(clipped_x > x_max, x_max, clipped_x)
    return clipped_x


def normalize_columns(data):
    """
    Normalize each column of the input data separately using z-score normalization (standardization).

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The normalized data array where each column has been standardized.
        numpy.ndarray: The mean value of each column used for normalization.
        numpy.ndarray: The standard deviation of each column used for normalization.
    """

    # Calculate mean and standard deviation for each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - mean) / (std)

    return normalized_data, mean, std


def normalize_columns_minmax(data):
    """
    Normalize each column of the input data separately using min-max normalization.

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The normalized data array where each column has been scaled to the range [0, 1].
        numpy.ndarray: The minimum value of each column used for normalization.
        numpy.ndarray: The maximum value of each column used for normalization.
    """

    # Calculate minimum and maximum for each column
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - min) / (max - min)

    return normalized_data, min, max


def normalize_columns_log(data):
    """
    Perform log transformation on the input data.

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The log-transformed data array.
    """

    log_transformed_data = np.log(data + 1)

    return log_transformed_data


def test_model(model_path:str, config_file_path:str, df:pd.DataFrame):

    # Load data in dataframe
    #df = import_json_to_dataframe(data_path, False, "")

    # Load the JSON data
    with open(config_file_path, "r") as file:
        config_dict = json.load(file)
    features = config_dict["features"]
    maskers_active = config_dict["maskers_active"]
    masker_gain = config_dict["masker_gain"]
    masker_transform = config_dict["masker_transform"]
    std_mean_norm = config_dict["std_mean_norm"]
    min_max_norm = config_dict["min_max_norm"]
    predict = config_dict["predict"]
    min = config_dict["min"]
    max = config_dict["max"]
    mean = config_dict["mean"]
    std = config_dict["std"]

    # Prepare data for inputting model
    if maskers_active:
        """ features = features + [ # Already saved in "features" field
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ] """
        if masker_transform == "-1,1":
            df["info.masker_bird"] = (df["info.masker_bird"] * 2 - 1) * masker_gain
            df["info.masker_construction"] = (
                df["info.masker_construction"] * 2 - 1
            ) * masker_gain
            df["info.masker_traffic"] = (
                df["info.masker_traffic"] * 2 - 1
            ) * masker_gain
            df["info.masker_silence"] = (
                df["info.masker_silence"] * 2 - 1
            ) * masker_gain
            df["info.masker_water"] = (df["info.masker_water"] * 2 - 1) * masker_gain
            df["info.masker_wind"] = (df["info.masker_wind"] * 2 - 1) * masker_gain
        else:
            df["info.masker_bird"] = (df["info.masker_bird"]) * masker_gain
            df["info.masker_construction"] = (
                df["info.masker_construction"] * masker_gain
            )
            df["info.masker_traffic"] = df["info.masker_traffic"] * masker_gain
            df["info.masker_silence"] = df["info.masker_silence"] * masker_gain
            df["info.masker_water"] = df["info.masker_water"] * masker_gain
            df["info.masker_wind"] = df["info.masker_wind"] * masker_gain

    # Get X and Y arrays
    X_test = df[features].values
    if predict == "P":
        Y_test = df["info.P_ground_truth"].values
    elif predict == "E":
        Y_test = df["info.E_ground_truth"].values

    # If needed, apply normalization to data
    if std_mean_norm:
        X_test = (X_test - np.array(mean)) / (np.array(std))
    if min_max_norm:
        X_test = (X_test - np.array(min)) / (np.array(max) - np.array(min))

    # Load the model from the .joblib file
    model = load(model_path)

    # Do predictions
    Y_prediction = clip(model.predict(X_test))
    # Get MSEs
    MSE_test = np.mean((Y_prediction - Y_test) ** 2)
    MAE_test = np.mean(np.abs(Y_prediction - Y_test))

    # Print metrics
    print("|   MSE   |   MAE     |")
    print("|---------+-----------|")
    print(f"|  {MSE_test:.4f} |   {MAE_test:.4f}  |")


def run_variations_EN(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_features_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
    df_f6 = input_dict["df_foldFs"].copy()
    if input_dict["maskers_active"]:
        features_to_use = features_to_use + [
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]

    pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
    # Prepare data fold 6
    if masker_transform == "-1,1":
        df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_silence"] = (
            df_f6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_traffic"] = (
            df_f6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
        df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
    else:
        df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * masker_gain
        )
        df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
        df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
        df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
        df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    with open(input_dict["name"], "a") as f:
        f.write(
            "     |         Mean squared error        |             Mean  error            |"
        )
        f.write("\n")
        f.write(
            "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
        )
        f.write("\n")
        f.write(
            "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        l1_ratio = 0.5
        prev_mean = 9999
        for value in alpha:

            f.write(f"Estimator alpha {value}")
            f.write("\n")

            model = ElasticNet(alpha=value, l1_ratio=l1_ratio, selection="random")

            MSEs_train = []
            MSEs_val = []
            MSEs_test = []
            MSEs_foldFs = []
            MEs_train = []
            MEs_val = []
            MEs_test = []
            MEs_foldFs = []

            for val_fold in [1, 2, 3, 4, 5]:

                # Extract dataframes
                df_train = df_to_use[
                    (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
                ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
                df_val = df_to_use[df_to_use["info.fold"] == val_fold]
                df_test = df_to_use[df_to_use["info.fold"] == 0]

                # Get ground-truth labels
                if input_dict["predict"] == "P":
                    Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.P_ground_truth"].values
                    Y_test = df_test["info.P_ground_truth"].values
                    Y_foldFs = df_f6["info.P_ground_truth"].values
                elif input_dict["predict"] == "E":
                    Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.E_ground_truth"].values
                    Y_test = df_test["info.E_ground_truth"].values
                    Y_foldFs = df_f6["info.E_ground_truth"].values

                # Get feature matrices
                X_train = df_train[features_to_use].values  # [:,0:100]
                X_val = df_val[features_to_use].values  # [:,0:100]
                X_test = df_test[features_to_use].values  # [:,0:100]
                X_foldFs = df_f6[features_to_use].values  # [:,0:100]

                # Get features normalized_data = (data - mean) / (std)
                if input_dict["std_mean_norm"]:
                    X_train, mean, std = normalize_columns(X_train)
                    X_val = (X_val - mean) / (std)
                    X_test = (X_test - mean) / (std)
                    X_foldFs = (X_foldFs - mean) / (std)
                # Get features normalized_data = (data - min) / (max-min)
                if input_dict["min_max_norm"]:
                    X_train, min, max = normalize_columns_minmax(X_train)
                    X_val = (X_val - min) / (max - min)
                    X_test = (X_test - min) / (max - min)
                    X_foldFs = (X_foldFs - min) / (max - min)

                # Fit model
                model.fit(X_train, Y_train)

                # Get MSEs
                MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
                MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
                MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
                MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
                ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
                ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
                ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
                ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

                # Add metrics
                MSEs_train.append(MSE_train)
                MSEs_val.append(MSE_val)
                MSEs_test.append(MSE_test)
                MSEs_foldFs.append(MSE_foldFs)
                MEs_train.append(ME_train)
                MEs_val.append(ME_val)
                MEs_test.append(ME_test)
                MEs_foldFs.append(ME_foldFs)

            f.write(f"Parameters {value}")
            f.write("\n")
            f.write(
                f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            current_mean = (
                np.mean(MEs_val) + np.mean(MEs_test) + np.mean(MEs_foldFs)
            ) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                chosen = value

        f.write(f"Best parameter: {chosen}, giving a mean of {prev_mean}")
        f.write("\n")

        alpha = chosen
        l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        prev_mean = 9999
        for value in l1_ratio:

            f.write(f"Estimator l1_ratio {value}")
            f.write("\n")

            model = ElasticNet(alpha=alpha, l1_ratio=value, selection="random")

            MSEs_train = []
            MSEs_val = []
            MSEs_test = []
            MSEs_foldFs = []
            MEs_train = []
            MEs_val = []
            MEs_test = []
            MEs_foldFs = []

            for val_fold in [1, 2, 3, 4, 5]:

                # Extract dataframes
                df_train = df_to_use[
                    (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
                ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
                df_val = df_to_use[df_to_use["info.fold"] == val_fold]
                df_test = df_to_use[df_to_use["info.fold"] == 0]

                # Get ground-truth labels
                if input_dict["predict"] == "P":
                    Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.P_ground_truth"].values
                    Y_test = df_test["info.P_ground_truth"].values
                    Y_foldFs = df_f6["info.P_ground_truth"].values
                elif input_dict["predict"] == "E":
                    Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.E_ground_truth"].values
                    Y_test = df_test["info.E_ground_truth"].values
                    Y_foldFs = df_f6["info.E_ground_truth"].values

                # Get feature matrices
                X_train = df_train[features_to_use].values  # [:,0:100]
                X_val = df_val[features_to_use].values  # [:,0:100]
                X_test = df_test[features_to_use].values  # [:,0:100]
                X_foldFs = df_f6[features_to_use].values  # [:,0:100]

                # Get features normalized_data = (data - mean) / (std)
                if input_dict["std_mean_norm"]:
                    X_train, mean, std = normalize_columns(X_train)
                    X_val = (X_val - mean) / (std)
                    X_test = (X_test - mean) / (std)
                    X_foldFs = (X_foldFs - mean) / (std)
                # Get features normalized_data = (data - min) / (max-min)
                if input_dict["min_max_norm"]:
                    X_train, min, max = normalize_columns_minmax(X_train)
                    X_val = (X_val - min) / (max - min)
                    X_test = (X_test - min) / (max - min)
                    X_foldFs = (X_foldFs - min) / (max - min)

                # Fit model
                model.fit(X_train, Y_train)
                print(".")

                # Get MSEs
                MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
                MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
                MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
                MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
                ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
                ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
                ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
                ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

                # Add metrics
                MSEs_train.append(MSE_train)
                MSEs_val.append(MSE_val)
                MSEs_test.append(MSE_test)
                MSEs_foldFs.append(MSE_foldFs)
                MEs_train.append(ME_train)
                MEs_val.append(ME_val)
                MEs_test.append(ME_test)
                MEs_foldFs.append(ME_foldFs)

            f.write(f"Parameters {value}")
            f.write("\n")
            f.write(
                f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            current_mean = (
                np.mean(MEs_val) + np.mean(MEs_test) + np.mean(MEs_foldFs)
            ) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                chosen = value

        f.write(f"Best parameter: {chosen}, giving a mean of {prev_mean}")
        f.write("\n")


def run_variations_RFR(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_features_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
    df_f6 = input_dict["df_foldFs"].copy()
    if input_dict["maskers_active"]:
        features_to_use = features_to_use + [
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]

    pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
    # Prepare data fold 6
    if masker_transform == "-1,1":
        df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_silence"] = (
            df_f6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_traffic"] = (
            df_f6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
        df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
    else:
        df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * masker_gain
        )
        df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
        df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
        df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
        df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.ensemble import RandomForestRegressor

    with open(input_dict["name"], "a") as f:
        f.write(
            "     |         Mean squared error        |             Mean  error            |"
        )
        f.write("\n")
        f.write(
            "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
        )
        f.write("\n")
        f.write(
            "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        n_estimators = [10, 20, 50, 100, 150, 180, 200, 250, 300, 350, 400, 500]

        prev_mean = 9999
        for value in n_estimators:

            f.write(f"Estimator {value}")
            f.write("\n")

            model = RandomForestRegressor(n_estimators=value, random_state=0)

            MSEs_train = []
            MSEs_val = []
            MSEs_test = []
            MSEs_foldFs = []
            MEs_train = []
            MEs_val = []
            MEs_test = []
            MEs_foldFs = []

            for val_fold in [1, 2, 3, 4, 5]:

                # Extract dataframes
                df_train = df_to_use[
                    (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
                ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
                df_val = df_to_use[df_to_use["info.fold"] == val_fold]
                df_test = df_to_use[df_to_use["info.fold"] == 0]

                # Get ground-truth labels
                if input_dict["predict"] == "P":
                    Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.P_ground_truth"].values
                    Y_test = df_test["info.P_ground_truth"].values
                    Y_foldFs = df_f6["info.P_ground_truth"].values
                elif input_dict["predict"] == "E":
                    Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.E_ground_truth"].values
                    Y_test = df_test["info.E_ground_truth"].values
                    Y_foldFs = df_f6["info.E_ground_truth"].values

                # Get feature matrices
                X_train = df_train[features_to_use].values  # [:,0:100]
                X_val = df_val[features_to_use].values  # [:,0:100]
                X_test = df_test[features_to_use].values  # [:,0:100]
                X_foldFs = df_f6[features_to_use].values  # [:,0:100]

                # Get features normalized_data = (data - mean) / (std)
                if input_dict["std_mean_norm"]:
                    X_train, mean, std = normalize_columns(X_train)
                    X_val = (X_val - mean) / (std)
                    X_test = (X_test - mean) / (std)
                    X_foldFs = (X_foldFs - mean) / (std)
                # Get features normalized_data = (data - min) / (max-min)
                if input_dict["min_max_norm"]:
                    X_train, min, max = normalize_columns_minmax(X_train)
                    X_val = (X_val - min) / (max - min)
                    X_test = (X_test - min) / (max - min)
                    X_foldFs = (X_foldFs - min) / (max - min)

                # Fit model
                model.fit(X_train, Y_train)
                print(".")

                # Get MSEs
                MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
                MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
                MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
                MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
                ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
                ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
                ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
                ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

                # Add metrics
                MSEs_train.append(MSE_train)
                MSEs_val.append(MSE_val)
                MSEs_test.append(MSE_test)
                MSEs_foldFs.append(MSE_foldFs)
                MEs_train.append(ME_train)
                MEs_val.append(ME_val)
                MEs_test.append(ME_test)
                MEs_foldFs.append(ME_foldFs)

            f.write(f"Parameters {value}")
            f.write("\n")
            f.write(
                f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            current_mean = (
                np.mean(MEs_val) + np.mean(MEs_test) + np.mean(MEs_foldFs)
            ) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                chosen = value

        f.write(f"Best parameter: {chosen}, giving a mean of {prev_mean}")
        f.write("\n")


def run_variations_KNN(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_features_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
    df_f6 = input_dict["df_foldFs"].copy()
    if input_dict["maskers_active"]:
        features_to_use = features_to_use + [
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]

    pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
    # Prepare data fold 6
    if masker_transform == "-1,1":
        df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_silence"] = (
            df_f6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_traffic"] = (
            df_f6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
        df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
    else:
        df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * masker_gain
        )
        df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
        df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
        df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
        df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.ensemble import RandomForestRegressor

    with open(input_dict["name"], "a") as f:
        f.write(
            "     |         Mean squared error        |             Mean  error            |"
        )
        f.write("\n")
        f.write(
            "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
        )
        f.write("\n")
        f.write(
            "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        n_neighbors = [10, 20, 50, 100, 150, 180, 200, 250, 300, 350, 400, 500]

        prev_mean = 9999
        for value in n_neighbors:

            model = KNeighborsRegressor(n_neighbors=value)  # , weights="distance"
            # print(f'Investigating performance of {model} model...')

            MSEs_train = []
            MSEs_val = []
            MSEs_test = []
            MSEs_foldFs = []
            MEs_train = []
            MEs_val = []
            MEs_test = []
            MEs_foldFs = []

            for val_fold in [1, 2, 3, 4, 5]:

                # Extract dataframes
                df_train = df_to_use[
                    (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
                ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
                df_val = df_to_use[df_to_use["info.fold"] == val_fold]
                df_test = df_to_use[df_to_use["info.fold"] == 0]

                # Get ground-truth labels
                if input_dict["predict"] == "P":
                    Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.P_ground_truth"].values
                    Y_test = df_test["info.P_ground_truth"].values
                    Y_foldFs = df_f6["info.P_ground_truth"].values
                elif input_dict["predict"] == "E":
                    Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                    Y_val = df_val["info.E_ground_truth"].values
                    Y_test = df_test["info.E_ground_truth"].values
                    Y_foldFs = df_f6["info.E_ground_truth"].values

                # Get feature matrices
                X_train = df_train[features_to_use].values  # [:,0:100]
                X_val = df_val[features_to_use].values  # [:,0:100]
                X_test = df_test[features_to_use].values  # [:,0:100]
                X_foldFs = df_f6[features_to_use].values  # [:,0:100]

                # Get features normalized_data = (data - mean) / (std)
                if input_dict["std_mean_norm"]:
                    X_train, mean, std = normalize_columns(X_train)
                    X_val = (X_val - mean) / (std)
                    X_test = (X_test - mean) / (std)
                    X_foldFs = (X_foldFs - mean) / (std)
                # Get features normalized_data = (data - min) / (max-min)
                if input_dict["min_max_norm"]:
                    X_train, min, max = normalize_columns_minmax(X_train)
                    X_val = (X_val - min) / (max - min)
                    X_test = (X_test - min) / (max - min)
                    X_foldFs = (X_foldFs - min) / (max - min)

                # Fit model
                model.fit(X_train, Y_train)
                print(".")
                # print("iterations ", X_LR.n_iter_, X_LR.n_features_in_)

                # Get MSEs
                MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
                MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
                MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
                MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
                ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
                ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
                ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
                ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

                # Add metrics
                MSEs_train.append(MSE_train)
                MSEs_val.append(MSE_val)
                MSEs_test.append(MSE_test)
                MSEs_foldFs.append(MSE_foldFs)
                MEs_train.append(ME_train)
                MEs_val.append(ME_val)
                MEs_test.append(ME_test)
                MEs_foldFs.append(ME_foldFs)

                # print(f'{val_fold:4d} | {MSE_train:.4f} | {MSE_val:.4f} | {MSE_test:.4f} | {ME_train:.4f} | {ME_val:.4f} | {ME_test:.4f} | {X_LR.intercept_:7.4f} | {X_train.shape[0]:5d} | {X_val.shape[0]:5d} | {X_test.shape[0]:^4d} | {X_train.shape[1]:^5d} | {np.sum(np.abs(X_LR.coef_) > 0):^5d} |')
            f.write(f"Parameters {value}")
            f.write("\n")
            f.write(
                f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            current_mean = (
                np.mean(MEs_val) + np.mean(MEs_test) + np.mean(MEs_foldFs)
            ) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                chosen = value

        f.write(f"Best parameter: {chosen}, giving a mean of {prev_mean}")
        f.write("\n")


def train_EN(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_features_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
    df_f6 = input_dict["df_foldFs"].copy()
    if input_dict["maskers_active"]:
        features_to_use = features_to_use + [
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]

    pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
    # Prepare data fold 6
    if masker_transform == "-1,1":
        df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_silence"] = (
            df_f6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_traffic"] = (
            df_f6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
        df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
    else:
        df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * masker_gain
        )
        df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
        df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
        df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
        df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


    # Store input data in output dictionary
    output_dict = {
        "maskers_active": input_dict["maskers_active"],
        "masker_gain": input_dict["masker_gain"],
        "masker_transform": input_dict["masker_transform"],
        "std_mean_norm": input_dict["std_mean_norm"],
        "min_max_norm": input_dict["min_max_norm"],
        "features": features_to_use,
        "predict": input_dict["predict"],
        "params": input_dict["params"],
    }

    print(
        "     |         Mean squared error        |             Mean  error            |"
    )
    print(
        "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
    )
    print(
        "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
    )
    print(
        "-----+--------+--------+--------+--------+--------+--------+--------+----------"
    )
    # Get parametres
    alpha = input_dict["params"][0]
    l1_ratio = input_dict["params"][1]

    print(f"Parameters {alpha, l1_ratio}")

    # Auxiliary variables to save once best model is chosen
    prev_mean = 9999
    val_fold_chosen = 0
    min_chosen = 0
    max_chosen = 0
    mean_chosen = 0
    std_chosen = 0

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, selection="random")

    MSEs_train = []
    MSEs_val = []
    MSEs_test = []
    MSEs_foldFs = []
    MEs_train = []
    MEs_val = []
    MEs_test = []
    MEs_foldFs = []

    for val_fold in [1, 2, 3, 4, 5]:

        # Extract dataframes
        df_train = df_to_use[
            (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
        ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
        df_val = df_to_use[df_to_use["info.fold"] == val_fold]
        df_test = df_to_use[df_to_use["info.fold"] == 0]

        # Get ground-truth labels
        if input_dict["predict"] == "P":
            Y_train = df_train["info.P_ground_truth"].values  # [0:10]
            Y_val = df_val["info.P_ground_truth"].values
            Y_test = df_test["info.P_ground_truth"].values
            Y_foldFs = df_f6["info.P_ground_truth"].values
        elif input_dict["predict"] == "E":
            Y_train = df_train["info.E_ground_truth"].values  # [0:10]
            Y_val = df_val["info.E_ground_truth"].values
            Y_test = df_test["info.E_ground_truth"].values
            Y_foldFs = df_f6["info.E_ground_truth"].values

        # Get feature matrices
        X_train = df_train[features_to_use].values  # [:,0:100]
        X_val = df_val[features_to_use].values  # [:,0:100]
        X_test = df_test[features_to_use].values  # [:,0:100]
        X_foldFs = df_f6[features_to_use].values  # [:,0:100]

        # Get features normalized_data = (data - mean) / (std)
        if input_dict["std_mean_norm"]:
            X_train, mean, std = normalize_columns(X_train)
            X_val = (X_val - mean) / (std)
            X_test = (X_test - mean) / (std)
            X_foldFs = (X_foldFs - mean) / (std)
        # Get features normalized_data = (data - min) / (max-min)
        if input_dict["min_max_norm"]:
            X_train, min, max = normalize_columns_minmax(X_train)
            X_val = (X_val - min) / (max - min)
            X_test = (X_test - min) / (max - min)
            X_foldFs = (X_foldFs - min) / (max - min)

        # Fit model
        model.fit(X_train, Y_train)
        print(".")

        # Get MSEs
        MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
        MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
        MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
        MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
        ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
        ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
        ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
        ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

        # Add metrics
        MSEs_train.append(MSE_train)
        MSEs_val.append(MSE_val)
        MSEs_test.append(MSE_test)
        MSEs_foldFs.append(MSE_foldFs)
        MEs_train.append(ME_train)
        MEs_val.append(ME_val)
        MEs_test.append(ME_test)
        MEs_foldFs.append(ME_foldFs)

        print(
            f"fold{val_fold} | {(MSE_train):.4f} | {(MSE_val):.4f} | {(MSE_test):.4f} | {(MSE_foldFs):.4f} | {(ME_train):.4f} | {(ME_val):.4f} | {(ME_test):.4f} | {(ME_foldFs):.4f} |"
        )
        print(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )

        # Check if validation fold provide the best results
        current_mean = (ME_val + ME_test + ME_foldFs) / 3
        if current_mean < prev_mean:
            prev_mean = current_mean
            model_chosen = copy.deepcopy(model)
            val_fold_chosen = val_fold
            if input_dict["std_mean_norm"]:
                std_chosen = std
                mean_chosen = mean
            if input_dict["min_max_norm"]:
                min_chosen = min
                max_chosen = max

    print(
        f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
    )
    print(
        "-----+--------+--------+--------+--------+--------+--------+--------+----------"
    )
    print(f"Parameters {alpha, l1_ratio}, best validation fold {val_fold_chosen}")

    # Save data to given path
    if type(min_chosen) == np.ndarray:
        output_dict["min"] = min_chosen.tolist()
    else:
        output_dict["min"] = min_chosen
    if type(max_chosen) == np.ndarray:
        output_dict["max"] = max_chosen.tolist()
    else:
        output_dict["max"] = max_chosen
    if type(mean_chosen) == np.ndarray:
        output_dict["mean"] = mean_chosen.tolist()
    else:
        output_dict["mean"] = mean_chosen
    if type(mean_chosen) == np.ndarray:
        output_dict["std"] = mean_chosen.tolist()
    else:
        output_dict["std"] = std_chosen
    output_dict["val_fold"] = val_fold_chosen
    json_path_name = (
        input_dict["folder_path"] + input_dict["model_name"] + "_config.json"
    )
    with open(json_path_name, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    # Save model to given path
    model_path_name = input_dict["folder_path"] + input_dict["model_name"] + ".joblib"
    dump(model_chosen, model_path_name)


def train_KNN(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_features_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
    df_f6 = input_dict["df_foldFs"].copy()
    if input_dict["maskers_active"]:
        features_to_use = features_to_use + [
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]

    pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
    # Prepare data fold 6
    if masker_transform == "-1,1":
        df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_silence"] = (
            df_f6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_traffic"] = (
            df_f6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
        df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
    else:
        df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
        df_f6["info.masker_construction"] = (
            df_f6["info.masker_construction"] * masker_gain
        )
        df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
        df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
        df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
        df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Store input data in output dictionary
    output_dict = {
        "maskers_active": input_dict["maskers_active"],
        "masker_gain": input_dict["masker_gain"],
        "masker_transform": input_dict["masker_transform"],
        "std_mean_norm": input_dict["std_mean_norm"],
        "min_max_norm": input_dict["min_max_norm"],
        "features": features_to_use,
        "predict": input_dict["predict"],
        "params": input_dict["params"],
    }

    print(
        "     |         Mean squared error        |             Mean  error            |"
    )
    print(
        "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
    )
    print(
        "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
    )
    print(
        "-----+--------+--------+--------+--------+--------+--------+--------+----------"
    )
    # Get parameter
    n_neighbors = input_dict["params"][0]

    print(f"Number of neighbors {n_neighbors}")

    # Auxiliary variables to save once best model is chosen
    prev_mean = 9999
    val_fold_chosen = 0
    min_chosen = 0
    max_chosen = 0
    mean_chosen = 0
    std_chosen = 0

    model = KNeighborsRegressor(n_neighbors=n_neighbors)

    MSEs_train = []
    MSEs_val = []
    MSEs_test = []
    MSEs_foldFs = []
    MEs_train = []
    MEs_val = []
    MEs_test = []
    MEs_foldFs = []

    for val_fold in [1, 2, 3, 4, 5]:

        # Extract dataframes
        df_train = df_to_use[
            (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
        ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
        df_val = df_to_use[df_to_use["info.fold"] == val_fold]
        df_test = df_to_use[df_to_use["info.fold"] == 0]

        # Get ground-truth labels
        if input_dict["predict"] == "P":
            Y_train = df_train["info.P_ground_truth"].values  # [0:10]
            Y_val = df_val["info.P_ground_truth"].values
            Y_test = df_test["info.P_ground_truth"].values
            Y_foldFs = df_f6["info.P_ground_truth"].values
        elif input_dict["predict"] == "E":
            Y_train = df_train["info.E_ground_truth"].values  # [0:10]
            Y_val = df_val["info.E_ground_truth"].values
            Y_test = df_test["info.E_ground_truth"].values
            Y_foldFs = df_f6["info.E_ground_truth"].values

        # Get feature matrices
        X_train = df_train[features_to_use].values  # [:,0:100]
        X_val = df_val[features_to_use].values  # [:,0:100]
        X_test = df_test[features_to_use].values  # [:,0:100]
        X_foldFs = df_f6[features_to_use].values  # [:,0:100]

        # Get features normalized_data = (data - mean) / (std)
        if input_dict["std_mean_norm"]:
            X_train, mean, std = normalize_columns(X_train)
            X_val = (X_val - mean) / (std)
            X_test = (X_test - mean) / (std)
            X_foldFs = (X_foldFs - mean) / (std)
        # Get features normalized_data = (data - min) / (max-min)
        if input_dict["min_max_norm"]:
            X_train, min, max = normalize_columns_minmax(X_train)
            X_val = (X_val - min) / (max - min)
            X_test = (X_test - min) / (max - min)
            X_foldFs = (X_foldFs - min) / (max - min)

        # Fit model
        model.fit(X_train, Y_train)
        print(".")

        # Get MSEs
        MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
        MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
        MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
        MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
        ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
        ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
        ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
        ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

        # Add metrics
        MSEs_train.append(MSE_train)
        MSEs_val.append(MSE_val)
        MSEs_test.append(MSE_test)
        MSEs_foldFs.append(MSE_foldFs)
        MEs_train.append(ME_train)
        MEs_val.append(ME_val)
        MEs_test.append(ME_test)
        MEs_foldFs.append(ME_foldFs)

        print(
            f"fold{val_fold} | {(MSE_train):.4f} | {(MSE_val):.4f} | {(MSE_test):.4f} | {(MSE_foldFs):.4f} | {(ME_train):.4f} | {(ME_val):.4f} | {(ME_test):.4f} | {(ME_foldFs):.4f} |"
        )
        print(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )

        # Check if validation fold provide the best results
        current_mean = (ME_val + ME_test + ME_foldFs) / 3
        if current_mean < prev_mean:
            prev_mean = current_mean
            model_chosen = copy.deepcopy(model)
            val_fold_chosen = val_fold
            if input_dict["std_mean_norm"]:
                std_chosen = std
                mean_chosen = mean
            if input_dict["min_max_norm"]:
                min_chosen = min
                max_chosen = max

    print(
        f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
    )
    print(
        "-----+--------+--------+--------+--------+--------+--------+--------+----------"
    )
    print(f"N_neighbors {n_neighbors}, best validation fold {val_fold_chosen}")

    # Save data to given path
    if type(min_chosen) == np.ndarray:
        output_dict["min"] = min_chosen.tolist()
    else:
        output_dict["min"] = min_chosen
    if type(max_chosen) == np.ndarray:
        output_dict["max"] = max_chosen.tolist()
    else:
        output_dict["max"] = max_chosen
    if type(mean_chosen) == np.ndarray:
        output_dict["mean"] = mean_chosen.tolist()
    else:
        output_dict["mean"] = mean_chosen
    if type(mean_chosen) == np.ndarray:
        output_dict["std"] = mean_chosen.tolist()
    else:
        output_dict["std"] = std_chosen
    output_dict["val_fold"] = val_fold_chosen
    json_path_name = (
        input_dict["folder_path"] + input_dict["model_name"] + "_config.json"
    )
    with open(json_path_name, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    # Save model to given path
    model_path_name = input_dict["folder_path"] + input_dict["model_name"] + ".joblib"
    dump(model_chosen, model_path_name)


def train_RFR(input_dict):

    txt_name = input_dict["folder_path"] + input_dict["model_name"]+".txt"
    with open(txt_name, "a") as f:
        masker_transform = input_dict["masker_transform"]
        masker_gain = input_dict["masker_gain"]
        df_to_use, features_to_use = prepare_features_models(
            input_dict["dataframe"].copy(),
            input_dict["features"],
            masker_transform,
            masker_gain,
        )
        df_f6 = input_dict["df_foldFs"].copy()
        if input_dict["maskers_active"]:
            features_to_use = features_to_use + [
                "info.masker_bird",
                "info.masker_construction",
                "info.masker_silence",
                "info.masker_traffic",
                "info.masker_water",
                "info.masker_wind",
            ]

        pd.options.mode.chained_assignment = None  # Ignore warning, default='warn'
        # Prepare data fold 6
        if masker_transform == "-1,1":
            df_f6["info.masker_bird"] = (df_f6["info.masker_bird"] * 2 - 1) * masker_gain
            df_f6["info.masker_construction"] = (
                df_f6["info.masker_construction"] * 2 - 1
            ) * masker_gain
            df_f6["info.masker_silence"] = (
                df_f6["info.masker_silence"] * 2 - 1
            ) * masker_gain
            df_f6["info.masker_traffic"] = (
                df_f6["info.masker_traffic"] * 2 - 1
            ) * masker_gain
            df_f6["info.masker_water"] = (df_f6["info.masker_water"] * 2 - 1) * masker_gain
            df_f6["info.masker_wind"] = (df_f6["info.masker_wind"] * 2 - 1) * masker_gain
        else:
            df_f6["info.masker_bird"] = df_f6["info.masker_bird"] * masker_gain
            df_f6["info.masker_construction"] = (
                df_f6["info.masker_construction"] * masker_gain
            )
            df_f6["info.masker_silence"] = df_f6["info.masker_silence"] * masker_gain
            df_f6["info.masker_traffic"] = df_f6["info.masker_traffic"] * masker_gain
            df_f6["info.masker_water"] = df_f6["info.masker_water"] * masker_gain
            df_f6["info.masker_wind"] = df_f6["info.masker_wind"] * masker_gain

        # Suppress ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Store input data in output dictionary
        output_dict = {
            "maskers_active": input_dict["maskers_active"],
            "masker_gain": input_dict["masker_gain"],
            "masker_transform": input_dict["masker_transform"],
            "std_mean_norm": input_dict["std_mean_norm"],
            "min_max_norm": input_dict["min_max_norm"],
            "features": features_to_use,
            "predict": input_dict["predict"],
            "params": input_dict["params"],
        }

        f.write(
            "     |         Mean squared error        |             Mean  error            |"
        )
        f.write("\n")
        f.write(
            "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
        )
        f.write("\n")
        f.write(
            "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        # Get parameter
        n_estimators = input_dict["params"][0]

        f.write(f"Number of estimators {n_estimators}")
        f.write("\n")

        # Auxiliary variables to save once best model is chosen
        prev_mean = 9999
        val_fold_chosen = 0
        min_chosen = 0
        max_chosen = 0
        mean_chosen = 0
        std_chosen = 0

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

        MSEs_train = []
        MSEs_val = []
        MSEs_test = []
        MSEs_foldFs = []
        MEs_train = []
        MEs_val = []
        MEs_test = []
        MEs_foldFs = []

        for val_fold in [1, 2, 3, 4, 5]:

            # Extract dataframes
            df_train = df_to_use[
                (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
            ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
            df_val = df_to_use[df_to_use["info.fold"] == val_fold]
            df_test = df_to_use[df_to_use["info.fold"] == 0]

            # Get ground-truth labels
            if input_dict["predict"] == "P":
                Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                Y_val = df_val["info.P_ground_truth"].values
                Y_test = df_test["info.P_ground_truth"].values
                Y_foldFs = df_f6["info.P_ground_truth"].values
            elif input_dict["predict"] == "E":
                Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                Y_val = df_val["info.E_ground_truth"].values
                Y_test = df_test["info.E_ground_truth"].values
                Y_foldFs = df_f6["info.E_ground_truth"].values

            # Get feature matrices
            X_train = df_train[features_to_use].values  # [:,0:100]
            X_val = df_val[features_to_use].values  # [:,0:100]
            X_test = df_test[features_to_use].values  # [:,0:100]
            X_foldFs = df_f6[features_to_use].values  # [:,0:100]

            # Get features normalized_data = (data - mean) / (std)
            if input_dict["std_mean_norm"]:
                X_train, mean, std = normalize_columns(X_train)
                X_val = (X_val - mean) / (std)
                X_test = (X_test - mean) / (std)
                X_foldFs = (X_foldFs - mean) / (std)
            # Get features normalized_data = (data - min) / (max-min)
            if input_dict["min_max_norm"]:
                X_train, min, max = normalize_columns_minmax(X_train)
                X_val = (X_val - min) / (max - min)
                X_test = (X_test - min) / (max - min)
                X_foldFs = (X_foldFs - min) / (max - min)

            # Fit model
            model.fit(X_train, Y_train)
            print(".")

            # Get MSEs
            MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
            MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
            MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
            MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
            ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
            ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
            ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
            ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

            # Add metrics
            MSEs_train.append(MSE_train)
            MSEs_val.append(MSE_val)
            MSEs_test.append(MSE_test)
            MSEs_foldFs.append(MSE_foldFs)
            MEs_train.append(ME_train)
            MEs_val.append(ME_val)
            MEs_test.append(ME_test)
            MEs_foldFs.append(ME_foldFs)

            f.write(
                f"fold{val_fold} | {(MSE_train):.4f} | {(MSE_val):.4f} | {(MSE_test):.4f} | {(MSE_foldFs):.4f} | {(ME_train):.4f} | {(ME_val):.4f} | {(ME_test):.4f} | {(ME_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            # Check if validation fold provide the best results
            current_mean = (ME_val + ME_test + ME_foldFs) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                model_chosen = copy.deepcopy(model)
                val_fold_chosen = val_fold
                if input_dict["std_mean_norm"]:
                    std_chosen = std
                    mean_chosen = mean
                if input_dict["min_max_norm"]:
                    min_chosen = min
                    max_chosen = max

        f.write(
            f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        f.write(f"N_estimators {n_estimators}, best validation fold {val_fold_chosen}")
        f.write("\n")

        # Save data to given path
        if type(min_chosen) == np.ndarray:
            output_dict["min"] = min_chosen.tolist()
        else:
            output_dict["min"] = min_chosen
        if type(max_chosen) == np.ndarray:
            output_dict["max"] = max_chosen.tolist()
        else:
            output_dict["max"] = max_chosen
        if type(mean_chosen) == np.ndarray:
            output_dict["mean"] = mean_chosen.tolist()
        else:
            output_dict["mean"] = mean_chosen
        if type(mean_chosen) == np.ndarray:
            output_dict["std"] = mean_chosen.tolist()
        else:
            output_dict["std"] = std_chosen
        output_dict["val_fold"] = val_fold_chosen
        json_path_name = (
            input_dict["folder_path"] + input_dict["model_name"] + "_config.json"
        )
        with open(json_path_name, "w") as json_file:
            json.dump(output_dict, json_file, indent=4)

        # Save model to given path
        model_path_name = input_dict["folder_path"] + input_dict["model_name"] + ".joblib"
        dump(model_chosen, model_path_name)
