import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import os
import seaborn as sns

# Set the global font properties
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
#


def format_timestamp(timestamp):
    """Function to format the timestamp"""

    date, time = timestamp.split("T")
    date = date.replace("-", "/")
    time = time.split(".")[0]
    return f"Measurement:\nDate {date}\nTime {time}"


def read_last_line_PE(predictions_file_path):
    """Function to read the last row of the file when predicting only
    pleasantness and eventfulness simultaneously"""
    with open(predictions_file_path, "r") as file:
        lines = file.readlines()
        if len(lines) > 1:
            last_line = lines[-1].split(";")
            x = float(last_line[0])
            y = float(last_line[1])
            timestamp = last_line[2]
            return x, y, timestamp
    return None, None, None


def plot_predictions_PE(predictions_file_path, palette):
    """Function to plot predictions of only
    Pleasantness and Eventfulness in iterative interface."""

    # Wait until the file has some lines
    while True:
        print("Waiting for data...")
        with open(predictions_file_path, "r") as file:
            lines = file.readlines()
            if len(lines) > 1:
                break
        time.sleep(1)

    # region - Initialize the plot
    plt.ion()  # Interactive mode on
    print("Hello")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Adjust the plot area to leave space for the text box
    plt.subplots_adjust(right=0.75)
    plt.xlim(-1, 1)  # Adjust according to your expected data range
    plt.ylim(-1, 1)  # Adjust according to your expected data range
    plt.axhline(0, color="#BBBBBB", linestyle="--")  # Add horizontal dashed line at y=0
    plt.axvline(0, color="#BBBBBB", linestyle="--")  # Add vertical dashed line at x=0
    ax.set_xlabel("PLEASANTNESS", fontsize=14)
    ax.set_ylabel("EVENTFULNESS", fontsize=14)
    rects = [
        Rectangle((-1, 0), 1, 1, color=palette["q1"], alpha=0.2),  # Top-left quadrant
        Rectangle((0, 0), 1, 1, color=palette["q2"], alpha=0.2),  # Top-right quadrant
        Rectangle(
            (-1, -1), 1, 1, color=palette["q3"], alpha=0.2
        ),  # Bottom-left quadrant
        Rectangle(
            (0, -1), 1, 1, color=palette["q4"], alpha=0.2
        ),  # Bottom-right quadrant
    ]
    for rect in rects:
        ax.add_patch(rect)

    # endregion

    # region - Reading and plotting data
    while True:
        x, y, timestamp = read_last_line_PE(predictions_file_path)
        print(" Pleasantness ", x, " and Eventfulness ", y, " Time stamp", timestamp)
        if x is not None and y is not None:
            if x >= 0:
                if y >= 0:
                    color = palette["q2"]
                else:
                    color = palette["q4"]
            else:
                if y >= 0:
                    color = palette["q1"]
                else:
                    color = palette["q3"]

            ax.clear()  # Clear the previous plot
            # Redraw
            for rect in rects:
                ax.add_patch(rect)
            sns.scatterplot(x=[x], y=[y], ax=ax, s=50, color=color)
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.axhline(0, color="#BBBBBB", linestyle="--")
            plt.axvline(0, color="#BBBBBB", linestyle="--")
            ax.set_xlabel("PLEASANTNESS", fontsize=14)
            ax.set_ylabel("EVENTFULNESS", fontsize=14)
            ax.set_title("URBAN SOUNDSCAPE MONITORING", fontsize=20)

            # Add text box with formatted timestamp
            info_text = format_timestamp(timestamp)
            plt.text(
                1.05,
                0.5,
                info_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            plt.draw()  # Update the plot
            plt.pause(0.3)  # Pause for a second

            sns.scatterplot(x=[x], y=[y], ax=ax, s=100, color=color)
            plt.draw()  # Update the plot
            plt.pause(0.3)  # Pause for a second
            sns.scatterplot(x=[x], y=[y], ax=ax, s=150, color=color)
            plt.draw()  # Update the plot
            plt.pause(0.4)  # Pause for a second

        time.sleep(1)  # Read file every second
    # endregion
