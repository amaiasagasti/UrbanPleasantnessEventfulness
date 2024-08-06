import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import os

# Set the global font properties
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.dataset.auxiliary_sources import sources_USM


def read_last_line(predictions_file_path, sources):
    """Function to read the last row of the file"""
    with open(predictions_file_path, "r") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].split(";")
            timestamp = last_line[-1]
            sources_predictions = last_line[:-3]
            PE_predictions = last_line[-3:-1]
            if len(sources) == len(sources_predictions):
                return (
                    sources_predictions,
                    PE_predictions,
                    timestamp,
                )
            else:
                raise ValueError(
                    "Length of input list of sources does not match the number of predictions imported"
                )
    return None, None


def format_timestamp(timestamp):
    """Function to format the timestamp"""

    date, time = timestamp.split("T")
    date = date.replace("-", "/")
    time = time.split(".")[0]
    return f"Measurement:\nDate {date}\nTime {time}"


def plot_predictions(predictions_file_path, sound_classes, palette_PE):

    # region --- Wait until the file has some lines
    while True:
        print("Waiting for data...")
        with open(predictions_file_path, "r") as file:
            lines = file.readlines()
            if lines:
                break
        time.sleep(1)
    # endregion

    # region --- Initialize the plot
    # Initialize the plots
    plt.ion()  # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # Two columns, equal width
    # General title for the entire figure
    plt.suptitle("URBAN SOUNDSCAPES", fontsize=30)

    # Plot 1: Horizontal Bar Plot
    # region --- Plot the bars horizontally
    ax1.barh(sound_classes, np.zeros(len(sound_classes)))
    ax1.set_xlim(0, 1)
    ax1.set_title("SOUNDS", fontsize=20)
    # endregion

    # Plot 2: Scatter Plot with Quadrants
    # Adjust the plot area to leave space for the text box
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color="#BBBBBB", linestyle="--")  # Add horizontal dashed line at y=0
    ax2.axvline(0, color="#BBBBBB", linestyle="--")  # Add vertical dashed line at x=0
    ax2.set_xlabel("PLEASANTNESS", fontsize=14)
    ax2.set_ylabel("EVENTFULNESS", fontsize=14)
    ax2.set_title("PERCEPTION", fontsize=20)
    # Force the second plot to have a square aspect ratio
    ax2.set_aspect("equal", "box", anchor="N")

    rects = [
        Rectangle(
            (-1, 0), 1, 1, color=palette_PE["q1"], alpha=0.2
        ),  # Top-left quadrant
        Rectangle(
            (0, 0), 1, 1, color=palette_PE["q2"], alpha=0.2
        ),  # Top-right quadrant
        Rectangle(
            (-1, -1), 1, 1, color=palette_PE["q3"], alpha=0.2
        ),  # Bottom-left quadrant
        Rectangle(
            (0, -1), 1, 1, color=palette_PE["q4"], alpha=0.2
        ),  # Bottom-right quadrant
    ]
    for rect in rects:
        ax2.add_patch(rect)
    # endregion

    # region --- Start reading and plotting data
    while True:
        (
            sources_predictions,
            PE_predictions,
            timestamp,
        ) = read_last_line(predictions_file_path, sound_classes)

        p = float(PE_predictions[0])
        e = float(PE_predictions[1])
        if p >= 0:
            if e >= 0:
                color = palette_PE["q2"]
            else:
                color = palette_PE["q4"]
        else:
            if e >= 0:
                color = palette_PE["q1"]
            else:
                color = palette_PE["q3"]

        ######## PLOT 1 ########################################
        ax1.clear()  # Clear the previous plot
        values = np.array(sources_predictions, dtype=float)
        # Redraw
        # Convert palette colors with transparency
        rgba_colors = []
        for value in values:
            print(color, value)
            rgba_color = mcolors.to_rgba(color, alpha=value)
            rgba_colors.append(rgba_color)

        bars = ax1.barh(
            sound_classes, np.ones(len(sources_predictions)), color=rgba_colors
        )
        ax1.set_xlim(0, 1)
        ax1.set_title("SOUNDS", fontsize=20)
        ######## PLOT 2 ##########################################
        ax2.clear()  # Clear the previous plot
        # Redraw
        for rect in rects:
            ax2.add_patch(rect)
        ax2.scatter(x=[p], y=[e], s=150, color=color)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        plt.axhline(0, color="#BBBBBB", linestyle="--")
        plt.axvline(0, color="#BBBBBB", linestyle="--")
        ax2.set_xlabel("PLEASANTNESS", fontsize=14)
        ax2.set_ylabel("EVENTFULNESS", fontsize=14)
        ax2.set_title("PERCEPTION", fontsize=20)
        ######### Add text box with formatted timestamp ###########
        info_text = format_timestamp(timestamp)
        ax2.text(
            0.4,
            -0.2,
            info_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        ############################################################
        plt.draw()  # Update the plot
        plt.pause(0.5)  # Pause for a second
        time.sleep(0.5)  # Read file every half second
        # endregion


# File path
file_path = "data/output/segments/output_file.txt"
palette_PE = {"q1": "#FC694D", "q2": "#0DB2AC", "q3": "#FABA32", "q4": "#84B66F"}

plot_predictions(
    predictions_file_path="data/simulation_predictions.txt",
    sound_classes=sources_USM,
    palette_PE=palette_PE,
)
