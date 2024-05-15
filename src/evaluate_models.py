import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from joblib import load

from SoundLights.dataset.features_groups import (
    ARAUS_features,
    Freesound_features,
    mix_features,
    masker_features,
)
from SoundLights.models.models_functions import test_model
from SoundLights.dataset.dataset_functions import (
    import_json_to_dataframe,
    prepare_data_models,
)

# Import data
responses_ARAUS_0_25 = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_0_25.json")
    )
)
responses_ARAUS_0_5 = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_0_5.json")
    )
)
responses_ARAUS_2 = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_2.json")
    )
)
responses_ARAUS_4 = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_4.json")
    )
)
responses_ARAUS_equal = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_equal.json")
    )
)
responses_ARAUS_random = prepare_data_models(
    import_json_to_dataframe(
        os.path.join("data/fold0_variations", "fold0_ARAUS_random.json")
    )
)


model = load("data/models/ElasticNet_ARAUS_P.joblib")
features = ARAUS_features + masker_features
print("WAV GAIN = Original X 0.25")
test_model(model, responses_ARAUS_0_25, features, "P")
print("WAV GAIN = Original X 0.5")
test_model(model, responses_ARAUS_0_5, features, "P")
print("WAV GAIN = Original X 2")
test_model(model, responses_ARAUS_2, features, "P")
print("WAV GAIN = Original X 4")
test_model(model, responses_ARAUS_4, features, "P")
print("WAV GAIN = 5 (all equal)")
test_model(model, responses_ARAUS_equal, features, "P")
print("WAV GAIN = Random")
test_model(model, responses_ARAUS_random, features, "P")

model = load("data/models/ElasticNet_ARAUS_E.joblib")
features = ARAUS_features + masker_features
print("WAV GAIN = Original X 0.25")
test_model(model, responses_ARAUS_0_25, features, "E")
print("WAV GAIN = Original X 0.5")
test_model(model, responses_ARAUS_0_5, features, "E")
print("WAV GAIN = Original X 2")
test_model(model, responses_ARAUS_2, features, "E")
print("WAV GAIN = Original X 4")
test_model(model, responses_ARAUS_4, features, "E")
print("WAV GAIN = 5 (all equal)")
test_model(model, responses_ARAUS_equal, features, "E")
print("WAV GAIN = Random")
test_model(model, responses_ARAUS_random, features, "E")
