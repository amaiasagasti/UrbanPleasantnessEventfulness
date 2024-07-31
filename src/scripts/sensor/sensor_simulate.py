import pyaudio
import wave
import threading
import time
import datetime
import json
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import csv
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from CLAP.src.laion_clap import CLAP_Module

matplotlib.use("Agg")  # Use the 'Agg' backend which does not require a GUI


# Function to play audio in real-time and save segments
def play_audio(filename, seconds, maintain_time, saving_file):

    # region PREPARATION #########################
    # Frames will store the chunks to be stored
    frames = []
    # Will be overwritten, but it is required to be declared as a time object from now already
    prev_time = time.time()
    i = 0
    prev_i = 0
    # Create a condition variable
    condition = threading.Condition()
    # Check if the file exists and create it with headers if it doesn't
    file_exists = os.path.exists(saving_file)
    # Open the file in append mode
    with open(saving_file, "a") as file:
        # endregion PREPARATION ######################

        # region MODEL LOADING #######################
        # Load the trained model
        model_P = joblib.load("data/models/trained/RFR_CLAP_P_raw.joblib")
        model_E = joblib.load("data/models/trained/RFR_CLAP_E.joblib")
        # Load the CLAP model to generate features
        print("------- code starts -----------")
        model_CLAP = CLAP_Module(enable_fusion=True)
        print("------- clap module -----------")
        model_CLAP.load_ckpt("data/models/630k-fusion-best.pt")
        print("------- model loaded -----------")
        # endregion MODEL LOADING ####################

        # region SAVE SEGMENTS ########################
        # Function to save segments, it will be a thread that will be active all the time
        def save_segment():
            # It works with the nonlocal variables prev_time(to check when to save) and frames (data to save)
            nonlocal prev_time, frames, seconds, i, prev_i
            while True:
                with condition:
                    # Check when to save --> every 3 seconds
                    # if (time.time() - prev_time) >= seconds:
                    if i != prev_i:
                        print("im in")
                        # Prepare files names with current date-time data
                        date_time = datetime.datetime.now()
                        time_str = date_time.strftime("%Y%m%d_%H%M%S")
                        file_name = f"segment_{time_str}"

                        # Convert frames (list of byte strings) to a single numpy array of integers
                        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

                        # Save WAV file
                        """ wav_file_path = os.path.join(save_directory, f"{file_name}.wav")
                        wf_segment = wave.open(wav_file_path, "wb")
                        wf_segment.setnchannels(wf.getnchannels())
                        wf_segment.setsampwidth(wf.getsampwidth())
                        wf_segment.setframerate(wf.getframerate())
                        wf_segment.writeframes(b"".join(frames))
                        wf_segment.close() """

                        # Prepare JSON info and save it
                        dB = 0  # Placeholder for Leq calculation
                        calib = 1.0  # Placeholder for mic calibration factor
                        json_info = {
                            "file": file_name,
                            "Leq": dB,
                            "mic_calib": calib,
                            "fs": wf.getframerate(),
                        }
                        json_file_path = os.path.join(
                            save_directory, f"{file_name}.json"
                        )
                        with open(json_file_path, "w") as f:
                            json.dump(json_info, f, indent=4)

                        # Save audio and JSON info in a pickle file
                        pickle_file_path = os.path.join(
                            save_directory, f"{file_name}.pkl"
                        )
                        with open(pickle_file_path, "wb") as f:
                            pickle.dump(
                                {"audio": audio_data, "info": json_info}, f
                            )  # frames

                        frames = []
                        prev_i = i
                        condition.notify()

        # Start the thread to save audio segments every 3 seconds
        threading.Thread(target=save_segment, daemon=True).start()
        # endregion SAVE SEGMENTS ######################

        # region DELETE OLD SEGMENTS ##################
        # Function to delete old segments, it will be a thread that will be active all the time
        def cleanup_old_files(folder_path, maintain_time):
            """Cleanup old files from a specified folder based on their age.

            Parameters:
            - folder_path (str): Path to the folder containing the files.
            - maintain_time (int): Maximum age (in seconds) beyond which files are considered old and eligible for deletion.
            """

            nonlocal prev_time, seconds, i, prev_i
            while True:
                # if (time.time() - prev_time) >= seconds:
                if i != prev_i:
                    file_pattern = "segment_*.pkl"
                    files = glob.glob(os.path.join(folder_path, file_pattern))

                    current_time = datetime.datetime.now()

                    for file_path in files:
                        # Extract timestamp from the file name
                        file_name = os.path.basename(file_path)
                        file_date_time = file_name.split("segment_")[1].split(".pkl")[0]
                        file_ts = datetime.datetime.strptime(
                            file_date_time, "%Y%m%d_%H%M%S"
                        )

                        # Calculate time difference in seconds
                        time_difference = (current_time - file_ts).total_seconds()

                        # Check if the file is older than maintain_time seconds
                        if time_difference > maintain_time:
                            try:
                                # Remove .pkl file
                                os.remove(file_path)

                                # Remove corresponding .json file
                                json_file_path = os.path.join(
                                    folder_path, file_name.split(".pkl")[0] + ".json"
                                )
                                if os.path.exists(json_file_path):
                                    os.remove(json_file_path)

                                # print(f"Deleted old files: {file_path}, {json_file_path}")
                            except Exception as e:
                                print(
                                    f"Error deleting files: {file_path}, {json_file_path}, Error: {str(e)}"
                                )

        threading.Thread(
            target=cleanup_old_files, daemon=True, args=(["segments/", maintain_time])
        ).start()
        # endregion DELETE OLD SEGMENTS ################

        # region JOIN SEGMENTS TO PROCESS##############
        # Function to join recent segments, it will be a thread that will be active all the time
        def join_and_save_audio(folder_path, output_wav_path):
            """
            Join audio segments from .pkl files in a specified folder, save them as a single WAV file, and plot the audio data.

            Parameters:
            - folder_path (str): Path to the folder containing .pkl files.
            - output_wav_path (str): Path to save the joined WAV file.
            """

            nonlocal prev_time, seconds, i, prev_i, model_CLAP, model_P, model_E, file
            while True:
                # if (time.time() - prev_time) >= seconds:
                # if i != prev_i:
                with condition:
                    condition.wait()
                    # time.sleep(1)
                    joined_audio = np.array(
                        [], dtype=np.int16
                    )  # Initialize empty numpy array for joined audio data
                    metadata = []  # Initialize list for metadata from each file

                    # Find all .pkl files in the folder
                    file_pattern = "segment_*.pkl"
                    files = glob.glob(os.path.join(folder_path, file_pattern))

                    # Sort files by timestamp in the filename
                    files.sort()

                    # Leave only 30 seconds of data (10 files), if there are less files, then less
                    number_files = len(files)
                    if number_files >= 10:
                        files = files[(number_files - 10) : number_files]

                    # Iterate over each .pkl file
                    for file_path in files:
                        try:
                            # Load data from .pkl file
                            with open(file_path, "rb") as f:
                                data = pickle.load(f)

                            # Append audio data to joined_audio
                            audio_segment = (data["audio"]) / (2**15 - 1)  # 16bit
                            if isinstance(audio_segment, np.ndarray):
                                joined_audio = np.concatenate(
                                    (joined_audio, audio_segment)
                                )
                            else:
                                print(f"Invalid audio data format in {file_path}")

                            # Append metadata to list
                            metadata.append(data["info"])

                        except Exception as e:
                            print(f"Error loading {file_path}: {str(e)}")

                    if len(metadata) == 0:
                        print("No audio segments found.")
                    else:
                        # Get Pleasantness and Eventfulness values
                        start_time = time.time()
                        # Extract features
                        features = model_CLAP.get_audio_embedding_from_data(
                            [joined_audio], use_tensor=False
                        )
                        # Predict using the trained model
                        prediction_P = model_P.predict(features)
                        prediction_E = model_E.predict(features)
                        finish_time = time.time()
                        # sum_predictions_P = sum_predictions_P + prediction_P
                        # sum_predictions_E = sum_predictions_E + prediction_E
                        print(
                            "PLEASANTNESS ",
                            prediction_P,
                            " EVENTFULNESS ",
                            prediction_E,
                            " and TIME ",
                            finish_time - start_time,
                        )
                        # Create a new log entry
                        file.write(
                            f"{prediction_P};{prediction_E};{datetime.datetime.now().isoformat()}"
                        )
                        file.write("\n")
                        file.flush()

                        # Save the joined audio to a WAV file
                        """ sample_width = 2  # 16-bit audio
                        channels = 1  # Mono audio
                        framerate = metadata[0].get("fs", 44100)

                        with wave.open(output_wav_path, "wb") as wf:
                            wf.setnchannels(channels)
                            wf.setsampwidth(sample_width)
                            wf.setframerate(framerate)
                            wf.writeframes(joined_audio.tobytes())

                        print(f"Joined audio saved to {output_wav_path}") """

                        # Play the joined audio
                        """ play_audio(joined_audio, framerate) """

                        # Plot the joined audio data
                        """ plt.figure(figsize=(10, 4))
                        time_axis = np.arange(len(joined_audio)) * (1 / 48000)
                        plt.plot(time_axis, joined_audio)
                        plt.ylim([-1, 1])
                        plt.title("Joined Audio Waveform")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Amplitude")
                        title = "segments/joined_audio_plot_" + str(i) + ".png"
                        print(title)
                        plt.savefig(title)  # Save plot to file instead of showing
                        plt.close() """

        threading.Thread(
            target=join_and_save_audio, daemon=True, args=(["segments/", 0])
        ).start()
        # endregion JOIN SEGMENTS TO PROCESS############

        # region READ AUDIO FILE IN REAL TIME #########
        # Read audio file, import it
        wf = wave.open(filename, "rb")
        fs = wf.getframerate()
        ch = wf.getnchannels()
        sample_width = wf.getsampwidth()
        print(fs, ch, sample_width)
        # Prepare audio player and stream
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=ch,
            rate=fs,
            output=True,
        )
        # Prepare analysis
        chunk_size = seconds * fs  # 3 because i want 3 seconds slots
        # Saving directory
        save_directory = "segments"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        #########################
        # Save in data first chunk of audio
        data = wf.readframes(chunk_size)
        # Play audio (does not work properly) and collect frames
        i = 0
        while len(data) > 0:
            prev_time = time.time()
            print(i)
            # Play chunk
            stream.write(data)
            # Save data to frames (to save)
            frames.append(data)
            # Read next chunk
            data = wf.readframes(chunk_size)
            # prepare for next it
            i = i + 1
        #########################
        # Stop stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        # endregion READ AUDIO FILE IN REAL TIME ######


# if __name__ == "__main__":
play_audio(
    "data/simulation_audios/audio_simulation_long3.wav",  # "data/listening_test_audios_32bit_simulation/freesound_23063_mono_16b.wav"
    seconds=3,
    maintain_time=30,
    saving_file="segments/output_file.txt",
)
