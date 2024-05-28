import sklearn.linear_model
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from SoundLights.dataset.features_groups import (
    general_info,
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
    clap_features,
)
from SoundLights.models.models_functions import (
    clip,
    normalize_columns,
    normalize_columns_minmax,
)

# INPUT #############################################################################
df = pd.read_csv("data/main_files/SoundLights_complete.csv")
saving_folder = "data/output_files_parallelized/"


#####################################################################################
#
#
#
#
#
############# FUNCTIONS ###########################################################
def prepare_data_models(
    dataframe,
    features_evaluated,
    masker_transform: str = "None",
    maskers_gain: float = 1,
):

    # Drop string columns
    """dataframe = dataframe.drop("info.file", axis=1)
    dataframe = dataframe.drop("info.participant", axis=1)"""

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
    print("\n dataframe fold 0 before anything", dataframe_fold0.info())
    print(" ----------------------------- ")
    dataframe_fold0 = dataframe_fold0.drop("info.file", axis=1)
    dataframe_fold0 = dataframe_fold0.drop("info.participant", axis=1)
    dataframe_fold0 = dataframe_fold0.groupby(
        ["info.soundscape", "info.masker", "info.smr"]
    ).mean()  # .reset_index()  # For the test set, the same 48 stimuli were shown to all participants so we take the mean of their ratings as the ground truth
    print("\n dataframe fold 0 after drop and groupby", dataframe_fold0.info())
    print(" ----------------------------- ")
    # print("\n dataframe fold 0 has infoo.soundscape????", dataframe_fold0["info.soundscape"])
    dataframe_filtered = dataframe[
        dataframe["info.fold"] != 0
    ]  # Filter rows where 'fold' column is not equal to 0
    print("\n dataframe fildered info", dataframe_filtered.info())
    print(" ----------------------------- ")
    dataframe = pd.concat(
        [dataframe_fold0, dataframe_filtered], ignore_index=True
    )  # Join together

    print("\n dataframe concat", dataframe.columns)
    print(" ----------------------------- ")

    # Drop columns with all equal values or std=0
    std = np.std(dataframe[features_evaluated], axis=0)
    columns_to_mantain_arg = np.where(std >= 0.00001)[0]
    columns_to_drop_arg = np.where(std < 0.00001)[0]
    columns_to_mantain = [features_evaluated[i] for i in columns_to_mantain_arg]
    columns_to_drop = [features_evaluated[i] for i in columns_to_drop_arg]
    print("columns to drop ", columns_to_drop)
    print(" ----------------------------- ")
    # print(features_evaluated[np.where(std == 0)[0]])
    dataframe.drop(columns=columns_to_drop, inplace=True)

    return dataframe, columns_to_mantain


def run_variations_model(input_dict):
    masker_transform = input_dict["masker_transform"]
    masker_gain = input_dict["masker_gain"]
    df_to_use, features_to_use = prepare_data_models(
        input_dict["dataframe"].copy(),
        input_dict["features"],
        masker_transform,
        masker_gain,
    )
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
        df_fold6["info.masker_bird"] = (
            df_fold6["info.masker_bird"] * 2 - 1
        ) * masker_gain
        df_fold6["info.masker_construction"] = (
            df_fold6["info.masker_construction"] * 2 - 1
        ) * masker_gain
        df_fold6["info.masker_silence"] = (
            df_fold6["info.masker_silence"] * 2 - 1
        ) * masker_gain
        df_fold6["info.masker_traffic"] = (
            df_fold6["info.masker_traffic"] * 2 - 1
        ) * masker_gain
        df_fold6["info.masker_water"] = (
            df_fold6["info.masker_water"] * 2 - 1
        ) * masker_gain
        df_fold6["info.masker_wind"] = (
            df_fold6["info.masker_wind"] * 2 - 1
        ) * masker_gain
    else:
        df_fold6["info.masker_bird"] = df_fold6["info.masker_bird"] * masker_gain
        df_fold6["info.masker_construction"] = (
            df_fold6["info.masker_construction"] * masker_gain
        )
        df_fold6["info.masker_silence"] = df_fold6["info.masker_silence"] * masker_gain
        df_fold6["info.masker_traffic"] = df_fold6["info.masker_traffic"] * masker_gain
        df_fold6["info.masker_water"] = df_fold6["info.masker_water"] * masker_gain
        df_fold6["info.masker_wind"] = df_fold6["info.masker_wind"] * masker_gain

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
            MSEs_fold6 = []
            MEs_train = []
            MEs_val = []
            MEs_test = []
            MEs_fold6 = []

            for val_fold in [1, 2, 3, 4, 5]:

                # Extract dataframes
                df_train = df_to_use[
                    (df_to_use["info.fold"] != val_fold) & (df_to_use["info.fold"] > 0)
                ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
                df_val = df_to_use[df_to_use["info.fold"] == val_fold]
                df_test = df_to_use[df_to_use["info.fold"] == 0]

                # Get ground-truth labels
                Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                Y_val = df_val["info.P_ground_truth"].values
                Y_test = df_test["info.P_ground_truth"].values
                Y_fold6 = df_fold6["info.P_ground_truth"].values

                # Get feature matrices
                X_train = df_train[features_to_use].values  # [:,0:100]
                X_val = df_val[features_to_use].values  # [:,0:100]
                X_test = df_test[features_to_use].values  # [:,0:100]
                X_fold6 = df_fold6[features_to_use].values  # [:,0:100]

                # Get features normalized_data = (data - mean) / (std)
                if input_dict["std_mean_norm"]:
                    X_train, mean, std = normalize_columns(X_train)
                    X_val = (X_val - mean) / (std)
                    X_test = (X_test - mean) / (std)
                    X_fold6 = (X_fold6 - mean) / (std)
                # Get features normalized_data = (data - min) / (max-min)
                if input_dict["min_max_norm"]:
                    X_train, min, max = normalize_columns_minmax(X_train)
                    X_val = (X_val - min) / (max - min)
                    X_test = (X_test - min) / (max - min)
                    X_fold6 = (X_fold6 - min) / (max - min)

                # Fit model
                X_LR = model.fit(X_train, Y_train)

                # Get MSEs
                MSE_train = np.mean((clip(X_LR.predict(X_train)) - Y_train) ** 2)
                MSE_val = np.mean((clip(X_LR.predict(X_val)) - Y_val) ** 2)
                MSE_test = np.mean((clip(X_LR.predict(X_test)) - Y_test) ** 2)
                MSE_fold6 = np.mean((clip(X_LR.predict(X_fold6)) - Y_fold6) ** 2)
                ME_train = np.mean(np.abs(clip(X_LR.predict(X_train)) - Y_train))
                ME_val = np.mean(np.abs(clip(X_LR.predict(X_val)) - Y_val))
                ME_test = np.mean(np.abs(clip(X_LR.predict(X_test)) - Y_test))
                ME_fold6 = np.mean(np.abs(clip(X_LR.predict(X_fold6)) - Y_fold6))

                # Add metrics
                MSEs_train.append(MSE_train)
                MSEs_val.append(MSE_val)
                MSEs_test.append(MSE_test)
                MSEs_fold6.append(MSE_fold6)
                MEs_train.append(ME_train)
                MEs_val.append(ME_val)
                MEs_test.append(ME_test)
                MEs_fold6.append(ME_fold6)

            f.write(f"Parameters {value}")
            f.write("\n")
            f.write(
                f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_fold6):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_fold6):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")

            current_mean = (
                np.mean(MEs_val) + np.mean(MEs_test) + np.mean(MEs_fold6)
            ) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                chosen = value

        f.write(f"Best parameter: {chosen}, giving a mean of {prev_mean}")
        f.write("\n")


#####################################################################################
#
#
#
#
#
############# PREPARE DATA #########################################################
# ARAUS features dataframe
df_ARAUS = df[general_info + ARAUS_features]
# Freesound features dataframe
df_Freesound = df[general_info + Freesound_features]
# CLAP embeddings dataframe
df_clap = df[general_info + ["CLAP"]]
# print(df_clap["CLAP"].values[1])
# print(df_clap["info.P_ground_truth"].values[1])
all_columns = general_info + clap_features
full_list = []
for index, row in df_clap.iterrows():
    string_list = row["CLAP"].split("[")[2].split("]")[0].split(",")
    clap_list = [float(item) for item in string_list]
    complete_new_row = list(row[general_info].values) + clap_list
    full_list.append(complete_new_row)
df_clap = pd.DataFrame(data=full_list, columns=all_columns)

df_real = pd.read_csv("data/main_files/SoundLights_fold6.csv")
# Adapt CLAP features
df_fold6 = df_real[
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
full_list = []
for index, row in df_fold6.iterrows():
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
df_fold6 = pd.DataFrame(data=full_list, columns=all_columns)

# Saving folder
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)
#####################################################################################
#
#
#
#
#
############# RUN ###################################################################
# ARAUS

input_dicts = [
{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_noM_noNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_noM_stdMeanNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_noM_minMaxnNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M1_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 5,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M5_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M10_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 20,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M20_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M10_stdMeanNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_ARAUS,
    "features": ARAUS_features,
    "name": saving_folder + "RFR_ARAUS_M10_minMaxNorm.txt",
},{  # Freesound
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_noM_noNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_noM_stdMeanNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_noM_minMaxnNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M1_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 5,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M5_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M10_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 20,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M20_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M10_stdMeanNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_Freesound,
    "features": Freesound_features,
    "name": saving_folder + "RFR_Freesound_M10_minMaxNorm.txt",
},{  # CLAP
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_noM_noNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_noM_stdMeanNorm.txt",
},{
    "maskers_active": False,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_noM_minMaxnNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 1,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M1_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 5,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M5_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M10_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 20,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M20_noNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": True,
    "min_max_norm": False,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M10_stdMeanNorm.txt",
},{
    "maskers_active": True,
    "masker_gain": 10,
    "masker_transform": "None",
    "std_mean_norm": False,
    "min_max_norm": True,
    "dataframe": df_clap,
    "features": clap_features,
    "name": saving_folder + "RFR_clap_M10_minMaxNorm.txt",
}
]

#for input_dict in input_dicts:
#    run_variations_model(input_dict)


# To use pymtg, you need to install the package like this:
# pip install git+https://github.com/MTG/pymtg

from pymtg.processing import WorkParallelizer
wp = WorkParallelizer()
for input_dict in input_dicts:
    wp.add_task(run_variations_model, input_dict)

wp.run(num_workers=14)
if wp.num_tasks_failed > 0:
    wp.show_errors()