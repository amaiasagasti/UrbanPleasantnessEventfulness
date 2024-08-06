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
from lib.dataset.auxiliary_sources import sources_USM


matplotlib.use("Agg")  # Use the 'Agg' backend which does not require a GUI


# Function to play audio in real-time and save segments
def play_audio(
    audio_file_path,
    seconds_segment,
    maintain_time,
    seconds_analysis,
    saving_file,
    sources,
    sources_models_dir,
    P_model_dir,
    E_model_dir,
):

    # region PREPARATION #########################
    # Frames will store the chunks to be stored
    frames = []
    # Will be overwritten, but it is required to be declared as a time object from now already
    prev_time = time.time()
    i = 0
    prev_i = 0
    # Create a condition variable
    condition = threading.Condition()
    # Open the file in append mode
    with open(saving_file, "a") as file:
        # endregion PREPARATION ######################

        # region MODEL LOADING #######################
        # Load model for source predictions and store in the dictionary
        models_sources = {}
        for source in sources:
            if " " in source:
                # Replace spaces with underscores
                source_path = source.replace(" ", "_")
            else:
                source_path = source
            model_path = os.path.join(sources_models_dir, f"{source_path}.joblib")
            models_sources[source] = joblib.load(model_path)

        # Load the trained P and E models
        model_P = joblib.load(P_model_dir)
        model_E = joblib.load(E_model_dir)

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
            nonlocal prev_time, frames, seconds_segment, i, prev_i
            while True:
                with condition:
                    # Check when to save --> every 3 seconds
                    # if (time.time() - prev_time) >= seconds:
                    if i != prev_i:
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

            nonlocal prev_time, seconds_segment, i, prev_i
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
            nonlocal prev_time, seconds_segment, seconds_analysis, i, prev_i, model_CLAP, models_sources, file

            while True:
                # if (time.time() - prev_time) >= seconds:
                # if i != prev_i:
                with condition:
                    condition.wait()
                    joined_audio = np.array(
                        [], dtype=np.int16
                    )  # Initialize empty numpy array for joined audio data
                    metadata = []  # Initialize list for metadata from each file

                    # Find all .pkl files in the folder
                    file_pattern = "segment_*.pkl"
                    files = glob.glob(os.path.join(folder_path, file_pattern))

                    # Sort files by timestamp in the filename
                    files.sort()

                    # Take only the neccessary segments for the analysis
                    desired_analysis_files = int(seconds_analysis / seconds_segment)
                    number_files = len(files)
                    if number_files >= desired_analysis_files:
                        files = files[
                            (number_files - desired_analysis_files) : number_files
                        ]

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
                        # Get sources presence predictions
                        # start_time = time.time()
                        # Extract features
                        features = model_CLAP.get_audio_embedding_from_data(
                            [joined_audio], use_tensor=False
                        )
                        # finish_time = time.time()

                        # Calculate probabilities for each source model
                        predictions = []
                        for source in sources:
                            print("SOURCEEEEEE ", source)
                            prediction = models_sources[source].predict_proba(features)[
                                0
                            ][1]
                            predictions.append(prediction)

                        # Calculate prediction values for each P/E model
                        prediction_P = model_P.predict(features)[0]
                        prediction_E = model_E.predict(features)[0]
                        print(prediction_P, "holaaa")
                        predictions.append(prediction_P)
                        predictions.append(prediction_E)

                        # Format the predictions into a string
                        prediction_str = ";".join(
                            [f"{pred:.2f}" for pred in predictions]
                        )

                        # Add the timestamp
                        timestamp = datetime.datetime.now().isoformat()
                        output_line = f"{prediction_str};{timestamp}\n"

                        # file.write("\n")
                        file.write(output_line)
                        file.flush()

        threading.Thread(
            target=join_and_save_audio, daemon=True, args=(["segments/", 0])
        ).start()
        # endregion JOIN SEGMENTS TO PROCESS############

        # region READ AUDIO FILE IN REAL TIME #########
        # Read audio file, import it
        wf = wave.open(audio_file_path, "rb")
        fs = wf.getframerate()
        ch = wf.getnchannels()
        sample_width = wf.getsampwidth()
        # Prepare audio player and stream
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=ch,
            rate=fs,
            output=True,
        )
        # Prepare analysis
        chunk_size = seconds_segment * fs  # 3 because i want 3 seconds slots
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


play_audio(
    audio_file_path="data/simulation_audios/audio_simulation_sources.wav",
    seconds_segment=1,
    maintain_time=6,
    seconds_analysis=3,
    saving_file="data/simulation_predictions.txt",
    sources=sources_USM,
    sources_models_dir="data/models/sources",
    P_model_dir="data/models/trained/model_pleasantness.joblib",
    E_model_dir="data/models/trained/model_eventfulness.joblib",
)
