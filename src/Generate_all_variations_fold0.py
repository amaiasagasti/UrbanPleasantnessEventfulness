import pandas as pd
from SoundLights.dataset.dataset_functions import (
    generate_features,
)


csvPath = "data/main_files/SoundLights_complete.csv"
df = pd.read_csv(csvPath)
audios_path = "data/soundscapes_augmented/ARAUS_fold0_01/"


variations = ["x0_5", "x2", "x4", "x6", "random"]
for variation in variations:
    print(variation)
    saving_path = "data/output/variation_" + variation + "/"
    print("saving path ", saving_path)

    if variation == "x0_5":
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            0.5,
        )
    elif variation == "x2":
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            2,
        )
    elif variation == "x4":
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            4,
        )
    elif variation == "x6":
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            6,
        )
    elif variation == "random":
        generate_features(
            audios_path,
            df,
            saving_path,
            ["ARAUS", "Freesound", "embedding"],
            "ARAUS_extended",
            6.44,
            9999,
        )
