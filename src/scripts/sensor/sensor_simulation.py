import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from CLAP.src.laion_clap import CLAP_Module
from lib.sensor.sensor_simulate import sensor_simulation
from lib.dataset.auxiliary_sources import sources_USM


sensor_simulation(
    audio_file_path="data/simulation_audios/audio_simulation_sources.wav",
    seconds_segment=1,
    maintain_time=6,
    seconds_analysis=3,
    saving_file="data/simulation_predictions.txt",
    P_model_dir="data/models/trained/model_pleasantness.joblib",
    E_model_dir="data/models/trained/model_eventfulness.joblib",
    sources=sources_USM,
    sources_models_dir="data/models/sources",
)
