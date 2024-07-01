import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set the global font properties
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"


# Function to read the last row of the file
# Function to read the last row of the file
def read_last_line(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip().split(";")
            x = float(last_line[0].strip("[]"))
            y = float(last_line[1].strip("[]"))
            timestamp = last_line[2]
            return x, y, timestamp
    return None, None, None


# Function to format the timestamp
def format_timestamp(timestamp):
    date, time = timestamp.split("T")
    date = date.replace("-", "/")
    time = time.split(".")[0]
    return f"Measurement:\nDate {date}\nTime {time}"


# File path
file_path = "segments/output_file.txt"

palette = {"q1": "#FC694D", "q2": "#0DB2AC", "q3": "#FABA32", "q4": "#84B66F"}

# Wait until the file has some lines
print("Waiting for data...")
while True:
    with open(file_path, "r") as file:
        lines = file.readlines()
        if lines:
            break
    time.sleep(1)

# Initialize the plot
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
    Rectangle((-1, -1), 1, 1, color=palette["q3"], alpha=0.2),  # Bottom-left quadrant
    Rectangle((0, -1), 1, 1, color=palette["q4"], alpha=0.2),  # Bottom-right quadrant
]
for rect in rects:
    ax.add_patch(rect)

# Initialize empty lists for data
x_data = []
y_data = []


# Start reading and plotting data
while True:
    x, y, timestamp = read_last_line(file_path)
    print(x, y, timestamp)
    if x is not None and y is not None:
        # x_data.append(x)
        # y_data.append(y)

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
