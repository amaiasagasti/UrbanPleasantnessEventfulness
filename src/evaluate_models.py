import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from joblib import load

from SoundLights.dataset.features_groups import (
    general_info,
    ARAUS_features,
    Freesound_features,
    clap_features,
    mix_features,
    masker_features,
)
from SoundLights.models.models_functions import test_model
from SoundLights.dataset.dataset_functions import (
    import_json_to_dataframe,
    expand_CLAP_features,
)
from SoundLights.models.models_functions import prepare_data_models

"""
WARNING:
We are importing .csv files of fold 0 variations. If you only have the .json versions, use function
import_json_to_dataframe() to save them first and then proceed.
"""

# Load data
# Data of fold 0 to know "ground truth"
data_path = "data/main_files/SoundLights_complete.csv"
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
df_fold0, features = prepare_data_models(df_fold0, all_features)
df_fold0 = df_fold0[df_fold0["info.fold"] == 0]
# Data of fold 0 variations
data_path = "data/main_files/variations_fold0/variations_fold0_0_5.csv"
df_0_5 = expand_CLAP_features(pd.read_csv(data_path))
data_path = "data/main_files/variations_fold0/variations_fold0_2.csv"
df_2 = expand_CLAP_features(pd.read_csv(data_path))
data_path = "data/main_files/variations_fold0/variations_fold0_4.csv"
df_4 = expand_CLAP_features(pd.read_csv(data_path))
data_path = "data/main_files/variations_fold0/variations_fold0_6.csv"
df_6 = expand_CLAP_features(pd.read_csv(data_path))
data_path = "data/main_files/variations_fold0/variations_fold0_random.csv"
df_random = expand_CLAP_features(pd.read_csv(data_path))


files_dicts = [
    {
        "title": "ELASTIC NET - ARAUS - PLEASANTNESS",
        "model_path": "data/models/trained/EN_ARAUS_P.joblib",
        "config_file_path": "data/models/trained/EN_ARAUS_P_config.json",
    },
    {
        "title": "KNN - ARAUS - PLEASANTNESS",
        "model_path": "data/models/trained/KNN_ARAUS_P.joblib",
        "config_file_path": "data/models/trained/KNN_ARAUS_P_config.json",
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
        "title": "KNN - Freesound - PLEASANTNESS",
        "model_path": "data/models/trained/KNN_Freesound_P.joblib",
        "config_file_path": "data/models/trained/KNN_Freesound_P_config.json",
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
        "title": "KNN - CLAP - PLEASANTNESS",
        "model_path": "data/models/trained/KNN_CLAP_P.joblib",
        "config_file_path": "data/models/trained/KNN_CLAP_P_config.json",
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
        "title": "KNN - ARAUS - EVENTFULNESS",
        "model_path": "data/models/trained/KNN_ARAUS_E.joblib",
        "config_file_path": "data/models/trained/KNN_ARAUS_E_config.json",
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
        "title": "KNN - Freesound - EVENTFULNESS",
        "model_path": "data/models/trained/KNN_Freesound_E.joblib",
        "config_file_path": "data/models/trained/KNN_Freesound_E_config.json",
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
        "title": "KNN - CLAP - EVENTFULNESS",
        "model_path": "data/models/trained/KNN_CLAP_E.joblib",
        "config_file_path": "data/models/trained/KNN_CLAP_E_config.json",
    },
    {
        "title": "RFR - CLAP - EVENTFULNESS",
        "model_path": "data/models/trained/RFR_CLAP_E.joblib",
        "config_file_path": "data/models/trained/RFR_CLAP_E_config.json",
    },
]

""" """

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
