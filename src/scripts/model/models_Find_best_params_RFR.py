"""
This script runs different configurations  of parameters for Random Forest Regressor
models in order to find the parameters that retrieve the smallest MAE error. 

run_variations_RFR() is the main function, it tests different Random Forest Regressor
parameters, number of estimators, with the here specified input configurations. Training, 
validating and testing MAE values are stored in txt files in the specified saving folder. 
Then, manually, these text files were analysed to find the best working model configuration.
The best performance options are the configurations trained and saved in the script 
named models_train.py.
"""

import os
import sys
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.models.models_functions import run_variations_RFR
from lib.models.models_functions import prepare_dataframes_models


def main(data_path, data_foldFs_path, saving_folder):
    ############# PREPARE DATA #########################################################
    """df_ARAUS, ARAUS_features, df_foldFs = prepare_dataframes_models(
        data_path, data_foldFs_path, saving_folder, "ARAUS"
    )
    df_Freesound, Freesound_features, df_foldFs = prepare_dataframes_models(
        data_path, data_foldFs_path, saving_folder, "Freesound"
    )"""
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
    """
    # region ARAUS
        {
            "maskers_active": False,
            "masker_gain": 1,
            "masker_transform": "None",
            "std_mean_norm": False,
            "min_max_norm": False,
            "dataframe": df_ARAUS,
            "features": ARAUS_features,
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
            "predict": "E",
            "name": saving_folder + "E_RFR_ARAUS_M20_minMaxNorm.txt",
        },
        # endregion
        # region Freesound
        {
            "maskers_active": False,
            "masker_gain": 1,
            "masker_transform": "None",
            "std_mean_norm": False,
            "min_max_norm": False,
            "dataframe": df_Freesound,
            "features": Freesound_features,
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
            "predict": "E",
            "name": saving_folder + "E_RFR_Freesound_M20_minMaxNorm.txt",
        },
        # endregion
        
    """
    input_dicts = [
        # region CLAP
        {
            "maskers_active": False,
            "masker_gain": 1,
            "masker_transform": "None",
            "std_mean_norm": False,
            "min_max_norm": False,
            "dataframe": df_clap,
            "features": clap_features,
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
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
            "df_foldFs": df_foldFs,
            "predict": "E",
            "name": saving_folder + "E_RFR_clap_M20_minMaxNorm.txt",
        },
        # endregion
    ]

    for input_dict in input_dicts:
        run_variations_RFR(input_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Try different parameter combinations for a Random Forest Regression model for the prediction of Pleasantness and Eventfulness. Saves data in txt file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file --> ARAUS_extended_norm_1_5.csv .",
    )  # FIXME remove file name
    parser.add_argument(
        "--data_foldFs_path",
        type=str,
        required=True,
        help="Path to the fold_Fs file --> fold_Fs_norm_1_5_all.csv",
    )  # FIXME remove file name
    parser.add_argument(
        "--saving_folder",
        type=str,
        required=True,
        help="Path to the folder where results will be saved --> training_RFR_norm_1_5/",
    )  # FIXME remove file name

    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    data_foldFs_path = args.data_foldFs_path
    saving_folder = args.saving_folder

    # Call main function
    main(data_path, data_foldFs_path, saving_folder)

# Example of command line:
# python src/scripts/model/models_Find_best_params_RFR.py --data_path data/ARAUS_extended.csv --data_foldFs_path data/fold_Fs.csv --saving_folder data/training_RFR/
