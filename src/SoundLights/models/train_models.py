import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from joblib import dump

sys.path.append(os.getcwd())
from src.SoundLights.features_groups import (
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
)
from src.SoundLights.models.models_functions import clip, train_elastic_net


# Import data
responses_ARAUS = pd.read_csv(os.path.join("data/csv_files", "SoundLights_ARAUS.csv"))
responses_ARAUS = responses_ARAUS.drop("info.file", axis=1)
responses_ARAUS = responses_ARAUS.drop("info.participant", axis=1)

# Maskers colum, increase values
responses_ARAUS["info.masker_bird"] = responses_ARAUS["info.masker_bird"] * 5
responses_ARAUS["info.masker_construction"] = (
    responses_ARAUS["info.masker_construction"] * 5
)
responses_ARAUS["info.masker_traffic"] = responses_ARAUS["info.masker_traffic"] * 5
responses_ARAUS["info.masker_water"] = responses_ARAUS["info.masker_water"] * 5
responses_ARAUS["info.masker_wind"] = responses_ARAUS["info.masker_wind"] * 5


# MODEL TO PREDICT PLEASANTNESS USING ARAUS + MASKERS FEATURES
alpha = 0.2
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
)

# MODEL TO PREDICT EVENTFULNESS USING ARAUS + MASKERS FEATURES
alpha = 0.2
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
)
