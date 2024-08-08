"""
This script runs different configurations  of parameters for Elastic Net models in
order to find the parameters that retrieve the smallest MAE error. 

run_variations_EN() is the main function, it tests different Elastic Net parameters,
alpha and l1_ratio, with the here specified input configurations. Training, validating
and testing MAE values are stored in txt files in the specified saving folder. Then,
manually, these text files were analysed to find the best working model configuration.
The best performance options are the configurations trained and saved in the script 
named models_train.py.
"""

import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.models.models_functions import (
    run_variations_EN,
)
from lib.models.models_functions import prepare_dataframes_models

# INPUT #############################################################################
data_path = "data/ARAUS_extended.csv"
data_foldFs_path = "data/fold_Fs.csv"
saving_folder = "data/training_EN/"
#####################################################################################
#
#
#
#
#
############# PREPARE DATA #########################################################
df_ARAUS, ARAUS_features, df_foldFs = prepare_dataframes_models(
    data_path, data_foldFs_path, saving_folder, "ARAUS"
)
df_Freesound, Freesound_features, df_foldFs = prepare_dataframes_models(
    data_path, data_foldFs_path, saving_folder, "Freesound"
)
df_clap, clap_features, df_foldFs = prepare_dataframes_models(
    data_path, data_foldFs_path, saving_folder, "CLAP"
)
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
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_ARAUS_M20_minMaxNorm.txt",
    },
    {  # Freesound
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "features": Freesound_features,
        "name": saving_folder + "P_EN_Freesound_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_Freesound_M20_minMaxNorm.txt",
    },
    {  # CLAP
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_noM_noNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_noM_stdMeanNorm.txt",
    },
    {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_noM_minMaxnNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M1_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M5_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 10,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M10_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M20_noNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M20_stdMeanNorm.txt",
    },
    {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": True,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "name": saving_folder + "P_EN_clap_M20_minMaxNorm.txt",
    },
]


for input_dict in input_dicts:
    run_variations_EN(input_dict)
