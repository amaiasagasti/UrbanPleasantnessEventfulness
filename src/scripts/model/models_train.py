"""
This script trains and saves the best performing models for each feature set. 

train_RFR() and train_EN() are the main function, each for one algorithm type.
Best performing models are saved when you run code.
Uncomment code at the bottom in order to train and save models according to the 
specified input configuration.
"""

import os
import sys
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.models.models_functions import train_EN, train_RFR, prepare_dataframes_models

# INPUT #############################################################################
""" data_path = "data/ARAUS_extended_norm_1_5.csv"
data_foldFs_path = "fold_Fs_norm_1_5_all.csv"
saving_folder = "data/models/trained_norm_1_5/"
 """


#####################################################################################
#
#
#
#
#
def main(data_path, data_foldFs_path, saving_folder):
    ############# PREPARE DATA #########################################################
    df_clap, clap_features, df_foldFs = prepare_dataframes_models(
        data_path, data_foldFs_path, saving_folder, "CLAP"
    )

    ############# RUN ###################################################################
    # BEST PERFORMING MODEL FOR PLEASANTNESS PREDICTION
    input_dict = {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [250],
        "folder_path": saving_folder,
        "model_name": "model_pleasantness_norm_1_5",
    }
    train_RFR(input_dict)
    # BEST PERFORMING MODEL FOR EVENTFULNESS PREDICTION
    input_dict = {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [500],
        "folder_path": saving_folder,
        "model_name": "model_eventfulness_norm_1_5",
    }
    train_RFR(input_dict)
    #####################################################################################
    #
    #
    #
    #
    #
    # region ############ OTHERS ###############################################################
    # print("RANDOM FOREST REGRESSOR ")

    ### RFR - ARAUS - Pleasantness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [100],
        "folder_path": saving_folder,
        "model_name": "RFR_ARAUS_P",
    }
    train_RFR(input_dict) """

    ### RFR - ARAUS - Eventfulness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [400],
        "folder_path": saving_folder,
        "model_name": "RFR_ARAUS_E",
    }
    train_RFR(input_dict) """

    ### RFR - Freesound - Pleasantness
    """ input_dict = {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [250],
        "folder_path": saving_folder,
        "model_name": "RFR_Freesound_P",
    }
    train_RFR(input_dict) """

    ### RFR - Freesound - Eventfulness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [250],
        "folder_path": saving_folder,
        "model_name": "RFR_Freesound_E",
    }
    train_RFR(input_dict) """

    ### RFR - CLAP - Pleasantness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [100],
        "folder_path": saving_folder,
        "model_name": "RFR_CLAP_P",
    }
    train_RFR(input_dict) """

    ### RFR - CLAP - Eventfulness
    """ input_dict = {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [500],
        "folder_path": saving_folder,
        "model_name": "RFR_CLAP_E",
    }
    train_RFR(input_dict) """

    ### RFR - CLAP - Pleasantness (ADDITIONAL)
    """ input_dict = {
        "maskers_active": False,
        "masker_gain": 1,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [250],
        "folder_path": saving_folder,
        "model_name": "RFR_CLAP_P_raw",
    }
    train_RFR(input_dict) """

    # print("\n")
    # print("\n")
    # print("##########################################################################")
    # print("ELASTIC NET ")

    ### Elastic Net - ARAUS - Pleasantness
    """
    input_dict = {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [0.6, 0.5],
        "folder_path": saving_folder,
        "model_name": "EN_ARAUS_P",
    }

    train_EN(input_dict)"""

    ### Elastic Net - ARAUS - Eventfulness
    """
    input_dict = {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_ARAUS,
        "features": ARAUS_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [0.1, 0.6],
        "folder_path": saving_folder,
        "model_name": "EN_ARAUS_E",
    }

    train_EN(input_dict)"""

    ### Elastic Net - Freesound - Pleasantness
    """
    input_dict = {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [0.3, 0.5],
        "folder_path": saving_folder,
        "model_name": "EN_Freesound_P",
    }

    train_EN(input_dict)"""

    ### Elastic Net - Freesound - Pleasantness
    """
    input_dict = {
        "maskers_active": True,
        "masker_gain": 5,
        "masker_transform": "None",
        "std_mean_norm": False,
        "min_max_norm": False,
        "dataframe": df_Freesound,
        "features": Freesound_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [0.2, 0.6],
        "folder_path": saving_folder,
        "model_name": "EN_Freesound_E",
    }

    train_EN(input_dict) """

    ### Elastic Net - CLAP - Pleasantness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "P",
        "params": [0.5, 0.5],
        "folder_path": saving_folder,
        "model_name": "EN_CLAP_P",
    }

    train_EN(input_dict) """

    ### Elastic Net - CLAP - Eventfulness
    """ input_dict = {
        "maskers_active": True,
        "masker_gain": 20,
        "masker_transform": "None",
        "std_mean_norm": True,
        "min_max_norm": False,
        "dataframe": df_clap,
        "features": clap_features,
        "df_foldFs": df_foldFs,
        "predict": "E",
        "params": [0.5, 0.5],
        "folder_path": saving_folder,
        "model_name": "EN_CLAP_E",
    }

    train_EN(input_dict) """

    # To use pymtg, you need to install the package like this:
    # pip install git+https://github.com/MTG/pymtg

    """ from pymtg.processing import WorkParallelizer

    wp = WorkParallelizer()
    for input_dict in input_dicts:
        wp.add_task(run_variations_model, input_dict)

    wp.run(num_workers=14)
    if wp.num_tasks_failed > 0:
        wp.show_errors() """
    # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest Regression models for the prediction of Pleasantness and Eventfulness with determined configuration. Saves models."
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
        help="Path to the folder where results will be saved --> data/models/trained_norm_1_5/",
    )  # FIXME remove file name

    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    data_foldFs_path = args.data_foldFs_path
    saving_folder = args.saving_folder

    # Call main function
    main(data_path, data_foldFs_path, saving_folder)

# Example of command line:
# python src/scripts/model/models_train.py --data_path data/ARAUS_extended.csv --data_foldFs_path data/fold_Fs.csv --saving_folder data/models_trained_RFR_norm_1_5/
