"""
This script tests models' robustess to changes in calibration (variations of fold_0). 

test_model() is the main function. It imports the fold_0 variations subdatasets and
applies the saved models to make predictions. Predicted values are compared to the
ground-truth. Resulting MAE values are printed through terminal
"""

import os
import sys
import pandas as pd

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.dataset.features_groups import (
    general_info,
    ARAUS_features,
    Freesound_features,
    clap_features,
)
from lib.models.models_functions import test_model
from lib.dataset.dataset_functions import (
    expand_CLAP_features,
)
from lib.models.models_functions import prepare_features_models

"""
NOTE:
We are importing .csv files of fold 0 variations. If you only have the .json versions, use function
import_json_to_dataframe() to save them first and then proceed.
"""
# INPUT #############################################################################
data_path = "data/ARAUS_extended.csv"
data_0_5_path = "data/variations_fold0/variations_fold0_0_5.csv"
data_2_path = "data/variations_fold0/variations_fold0_2.csv"
data_4_path = "data/variations_fold0/variations_fold0_4.csv"
data_6_path = "data/variations_fold0/variations_fold0_6.csv"
data_random_path = "data/variations_fold0/variations_fold0_random.csv"
#####################################################################################
#
#
#
#
#
############# PREPARE DATA #########################################################
# Data of fold 0 to know "ground truth"
df_fold0 = pd.read_csv(data_path)
full_list = []
all_columns = general_info + ARAUS_features + Freesound_features + clap_features
for index, row in df_fold0.iterrows():
    string_list = row["CLAP"].split("[")[2].split("]")[0].split(",")
    clap_list = [float(item) for item in string_list]
    complete_new_row = (
        list(row[general_info + ARAUS_features + Freesound_features].values) + clap_list
    )
    full_list.append(complete_new_row)
df_fold0 = pd.DataFrame(data=full_list, columns=all_columns)
all_features = ARAUS_features + Freesound_features + clap_features
df_fold0, features = prepare_features_models(df_fold0, all_features)
df_fold0 = df_fold0[df_fold0["info.fold"] == 0]

# Dataframes of fold 0 variations
df_0_5 = expand_CLAP_features(pd.read_csv(data_0_5_path))
df_2 = expand_CLAP_features(pd.read_csv(data_2_path))
df_4 = expand_CLAP_features(pd.read_csv(data_4_path))
df_6 = expand_CLAP_features(pd.read_csv(data_6_path))
df_random = expand_CLAP_features(pd.read_csv(data_random_path))
#####################################################################################
#
#
#
#
#
############# RUN ###################################################################
files_dicts = [
    {
        "title": "ELASTIC NET - ARAUS - PLEASANTNESS",
        "model_path": "data/models/trained/EN_ARAUS_P.joblib",
        "config_file_path": "data/models/trained/EN_ARAUS_P_config.json",
    },
    {
        "title": "RFR - ARAUS - PLEASANTNESS",
        "model_path": "data/models/trained/RFR_ARAUS_P.joblib",
        "config_file_path": "data/models/trained/RFR_ARAUS_P_config.json",
    },
    {
        "title": "ELASTIC NET - Freesound - PLEASANTNESS",
        "model_path": "data/models/trained/EN_Freesound_P.joblib",
        "config_file_path": "data/models/trained/EN_Freesound_P_config.json",
    },
    {
        "title": "RFR - Freesound - PLEASANTNESS",
        "model_path": "data/models/trained/RFR_Freesound_P.joblib",
        "config_file_path": "data/models/trained/RFR_Freesound_P_config.json",
    },
    {
        "title": "ELASTIC NET - CLAP - PLEASANTNESS",
        "model_path": "data/models/trained/EN_CLAP_P.joblib",
        "config_file_path": "data/models/trained/EN_CLAP_P_config.json",
    },
    {
        "title": "RFR - CLAP - PLEASANTNESS",
        "model_path": "data/models/trained/RFR_CLAP_P.joblib",
        "config_file_path": "data/models/trained/RFR_CLAP_P_config.json",
    },
    {
        "title": "RFR RAW  CLAP - PLEASANTNESS",
        "model_path": "data/models/trained/RFR_CLAP_P_raw.joblib",
        "config_file_path": "data/models/trained/RFR_CLAP_P_raw_config.json",
    },
    {
        "title": "ELASTIC NET - ARAUS - EVENTFULNESS",
        "model_path": "data/models/trained/EN_ARAUS_E.joblib",
        "config_file_path": "data/models/trained/EN_ARAUS_E_config.json",
    },
    {
        "title": "RFR - ARAUS - EVENTFULNESS",
        "model_path": "data/models/trained/RFR_ARAUS_E.joblib",
        "config_file_path": "data/models/trained/RFR_ARAUS_E_config.json",
    },
    {
        "title": "ELASTIC NET - Freesound - EVENTFULNESS",
        "model_path": "data/models/trained/EN_Freesound_E.joblib",
        "config_file_path": "data/models/trained/EN_Freesound_E_config.json",
    },
    {
        "title": "RFR - Freesound - EVENTFULNESS",
        "model_path": "data/models/trained/RFR_Freesound_E.joblib",
        "config_file_path": "data/models/trained/RFR_Freesound_E_config.json",
    },
    {
        "title": "ELASTIC NET - CLAP - EVENTFULNESS",
        "model_path": "data/models/trained/EN_CLAP_E.joblib",
        "config_file_path": "data/models/trained/EN_CLAP_E_config.json",
    },
    {
        "title": "RFR - CLAP - EVENTFULNESS",
        "model_path": "data/models/trained/RFR_CLAP_E.joblib",
        "config_file_path": "data/models/trained/RFR_CLAP_E_config.json",
    },
]

for file_dict in files_dicts:
    title = file_dict["title"]
    model_path = file_dict["model_path"]
    config_file_path = file_dict["config_file_path"]
    print("\n")
    print("################################################################")
    print(title)
    print("fold0 ground truth")
    test_model(model_path, config_file_path, df_fold0.copy())
    print("fold0 variation 0_5")
    test_model(model_path, config_file_path, df_0_5.copy())
    print("fold0 variation 2")
    test_model(model_path, config_file_path, df_2.copy())
    print("fold0 variation 4")
    test_model(model_path, config_file_path, df_4.copy())
    print("fold0 variation 6")
    test_model(model_path, config_file_path, df_6.copy())
    print("fold0 variation random")
    test_model(model_path, config_file_path, df_random.copy())
    print("################################################################")
