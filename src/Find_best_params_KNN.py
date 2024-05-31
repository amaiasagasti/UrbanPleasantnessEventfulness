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
    run_variations_KNN,
)

# INPUT #############################################################################
df = pd.read_csv("data/main_files/SoundLights_complete.csv")
saving_folder = "data/training_KNN/"
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


input_dicts = [
    # ARAUS
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_ARAUS_M20_minMaxNorm.txt",
    },
    {  # Freesound
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "df_fold6": df_fold6,
        "predict": "E",
        "features": Freesound_features,
        "name": saving_folder + "E_KNN_Freesound_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_Freesound_M20_minMaxNorm.txt",
    },
    {  # CLAP
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_clap,
        "features": clap_features,
        "df_fold6": df_fold6,
        "predict": "E",
        "name": saving_folder + "E_KNN_clap_M20_minMaxNorm.txt",
    },
]


for input_dict in input_dicts:
    run_variations_KNN(input_dict)


# To use pymtg, you need to install the package like this:
# pip install git+https://github.com/MTG/pymtg

""" from pymtg.processing import WorkParallelizer

wp = WorkParallelizer()
for input_dict in input_dicts:
    wp.add_task(run_variations_model, input_dict)

wp.run(num_workers=14)
if wp.num_tasks_failed > 0:
    wp.show_errors() """
