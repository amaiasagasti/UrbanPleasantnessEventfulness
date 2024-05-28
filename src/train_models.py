import pandas as pd

from SoundLights.dataset.features_groups import (
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
    clap_features,
    general_info,
)
from SoundLights.models.models_functions import clip, train_elastic_net, train_RFR


# Import data
df = pd.read_csv("data/main_files/SoundLights_complete.csv")
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
    # clap_list=clap_list[0:101] ##############################!!!!!!!!!!!!
    complete_new_row = list(row[general_info].values) + clap_list
    full_list.append(complete_new_row)
df_clap = pd.DataFrame(data=full_list, columns=all_columns)

# MODEL TO PREDICT PLEASANTNESS USING CLAP FEATURES
n_estimators = 200
features = clap_features
train_RFR(
    df_clap,
    features,
    n_estimators,
    1,
    "P",
    "data/models/RFR_CLAP_P2.joblib",
)


# MODEL TO PREDICT PLEASANTNESS USING ARAUS + MASKERS FEATURES
""" alpha = 0.2
l1_ratio = 0.5
features = ARAUS_features + masker_features
train_elastic_net(
    responses_ARAUS,
    features,
    alpha,
    l1_ratio,
    1,
    "P",
    "data/models/ElasticNet_ARAUS_P.joblib",
) """

# MODEL TO PREDICT EVENTFULNESS USING ARAUS + MASKERS FEATURES
""" alpha = 0.2
l1_ratio = 0.5
features = ARAUS_features + masker_features
train_elastic_net(
    responses_ARAUS,
    features,
    alpha,
    l1_ratio,
    1,
    "E",
    "data/models/ElasticNet_ARAUS_E.joblib",
) """
