import joblib

import sklearn.linear_model
import numpy as np
import pandas as pd
import os

# from SoundLights.features_groups import ARAUS_features, Freesound_features, mix_features


# Import data
responses_ARAUS = pd.read_csv(
    os.path.join("..", "data/csv_files", "SoundLights_ARAUS.csv"),
    dtype={"info.participant": str},
)
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

print(responses_ARAUS)
