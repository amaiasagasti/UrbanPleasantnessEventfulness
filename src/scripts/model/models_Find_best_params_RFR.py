import pandas as pd
import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from SoundLights.dataset.features_groups import (
    general_info,
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
    clap_features,
)
from SoundLights.models.models_functions import run_variations_RFR

# INPUT #############################################################################
data_path = "data/main_files/SoundLights_complete.csv"
data_foldFs_path = "data/main_files/SoundLights_fold6.csv"
saving_folder = "data/training_RFR_delete/"
#####################################################################################
#
#
#
#
#
############# PREPARE DATA #########################################################
df = pd.read_csv(data_path)
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

df_real = pd.read_csv(data_foldFs_path)
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
        "name": saving_folder + "E_RFR_ARAUS_noM_noNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_noM_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_noM_minMaxnNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M1_noNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M5_noNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M10_noNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M20_noNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M20_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_ARAUS_M20_minMaxNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_noM_noNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_noM_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_noM_minMaxnNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M1_noNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M5_noNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M10_noNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M20_noNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M20_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_Freesound_M20_minMaxNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_noM_noNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_noM_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_noM_minMaxnNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M1_noNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M5_noNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M10_noNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M20_noNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M20_stdMeanNorm.txt",
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
        "name": saving_folder + "E_RFR_clap_M20_minMaxNorm.txt",
    },
]


for input_dict in input_dicts:
    run_variations_RFR(input_dict)
