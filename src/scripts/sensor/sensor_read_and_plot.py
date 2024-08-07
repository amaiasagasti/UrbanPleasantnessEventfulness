import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from CLAP.src.laion_clap import CLAP_Module
from lib.sensor.sensor_predictions_plotting import (
    plot_predictions_all,
    plot_predictions_PE,
)
from lib.dataset.auxiliary_sources import sources_USM

# Pleasantness and Eventfulness prediction
""" plot_predictions_PE(
    predictions_file_path="data/simulation_predictions.txt",
    palette={"q1": "#FC694D", "q2": "#0DB2AC", "q3": "#FABA32", "q4": "#84B66F"},
) """

# Sound Sources, Pleasantness and Eventfulness prediction
""" plot_predictions_all(
    predictions_file_path="data/simulation_predictions.txt",
    sound_classes=sources_USM,
    palette_PE={"q1": "#FC694D", "q2": "#0DB2AC", "q3": "#FABA32", "q4": "#84B66F"},
) """
