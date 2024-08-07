"""
This script contains the function that simulates the workflow of a sensor connected to a microphone by progressively reading an audio file, 
similar to how a real sensor would record data in real-time.

The script performs the following tasks:
- Simulates audio recording by reading an audio file in fragments, emulating how a sensor would capture and process audio data in chunks.
- In a separate thread, it reassembles these data fragments to reconstruct the complete audio signal.
- The reconstructed audio is then fed into a model to predict values of P (pleasantness), E (eventfulness) or sound sources (specified)
- The predicted values are saved into a text file, with each prediction on a new line.
- Simultaneously, in a separate thread, old data fragments are deleted.

"""

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
import warnings
from contextlib import redirect_stdout, redirect_stderr
from transformers import logging
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)


# Imports from this project
from CLAP.src.laion_clap import CLAP_Module
from lib.dataset.auxiliary_sources import sources_USM


matplotlib.use("Agg")  # Use the 'Agg' backend which does not require a GUI


def sensor_simulation(
    audio_file_path: str,
    seconds_segment: float,
    maintain_time: float,
    seconds_analysis: float,
    saving_file: str,
    sources: list = None,
    sources_models_dir: str = None,
    P_model_dir: str = None,
    E_model_dir: str = None,
):
    """
    Simulates real-time audio recording and processing for predicting specified qualities such as sound sources,
    pleasantness, and eventfulness.

    Parameters:
    ----------
    audio_file_path : str
        Path to the audio file that will be simulated for real-time recording.

    seconds_segment : float
        Duration of each audio chunk to be stored, in seconds.

    maintain_time : float
        Total time, in seconds, for which audio chunks are stored. Chunks older than this time limit will be deleted.
        Must be a multiple of `seconds_segment`.

    seconds_analysis : float
        Time, in seconds, of audio chunks to be considered for processing.
        Must be an integer multiple of `seconds_segment` and smaller than `maintain_time`.

    saving_file : str
        Path to the text file where the results of the predictions will be stored.
        A new line will be added for each new prediction.

    sources : list, optional
        List of sound sources to be predicted. Each string in the list corresponds to the model name (e.g., 'source.joblib').

    sources_models_dir : str, optional
        Path to the directory containing the models for sound sources. The names of the models in this folder
        must match the names in the `sources` list (e.g., 'source.joblib').

    P_model_dir : str, optional
        Path to the model for predicting pleasantness.

    E_model_dir : str, optional
        Path to the model for predicting eventfulness.


    Notes:
    -----
    This function simulates real-time operating sensor by progressively reading audio. While the audio is being read, it is stored in chunks. Old audio
    chunks are deleted as specified. Simulating real-time, the saved chunks are processed to make predictions about the presence of sound sources, pleasantness,
    and eventfulness, saving these predictions to a specified text file.

    ValueError raises:
    If none of the optional parameters (`P_model_dir`, `E_model_dir`, `sources` and `sources_models_dir`)
    are provided or if `sources` is provided without `sources_models_dir` or vice versa.
    If `maintain_time` or `seconds_analysis` are not multiples of `seconds_segment`.
    If `seconds_analysis` is not smaller than `maintain_time`.
    """

    # region CHECK INPUTS ########################
    if (
        P_model_dir is None
        and E_model_dir is None
        and (sources is None or sources_models_dir is None)
    ):
        raise ValueError(
            "Cannot make any predictions. "
            "At least one of the following conditions must be met: "
            "1) P_model_dir is not None, "
            "2) E_model_dir is not None, "
            "3) Both sources and sources_models_dir are not None."
        )

    if (sources is not None and sources_models_dir is None) or (
        sources is None and sources_models_dir is not None
    ):
        raise ValueError(
            "Both sources and sources_models_dir must be provided together."
        )

    # Check if maintain_time and seconds_analysis are multiples of seconds_segment
    if maintain_time % seconds_segment != 0:
        raise ValueError("maintain_time must be a multiple of seconds_segment.")

    if seconds_analysis % seconds_segment != 0:
        raise ValueError("seconds_analysis must be a multiple of seconds_segment.")

    # Check if seconds_analysis is smaller than maintain_time
    if seconds_analysis >= maintain_time:
        raise ValueError("seconds_analysis must be smaller than maintain_time.")

    # Your function implementation

    # endregion CHECK INPUTS #####################

    # region PREPARATION #########################
    # Frames will store the chunks to be stored
    frames = []
    # Will be overwritten, but it is required to be declared as a time object from now already
    prev_time = time.time()
    i = 0
    prev_i = 0
    # Create a condition variable
    condition = threading.Condition()
    # Open the file in write mode to clear its contents
    with open(saving_file, "w") as file:
        file.write("")
    # Open the file in append mode
    with open(saving_file, "a") as file:
        # Write first line with headers
        first_line = []
        if sources is not None:
            for source in sources:
                first_line.append(source)
        if P_model_dir is not None:
            first_line.append("P")
        if E_model_dir is not None:
            first_line.append("E")
        first_line.append("datetime\n")
        first_line_str = ";".join([header for header in first_line])
        file.write(first_line_str)
        file.flush()
        # endregion PREPARATION ######################

        # region MODEL LOADING #######################
        # Load model for source predictions and store in the dictionary
        if sources is not None:
            models_sources = {}
            for source in sources:
                if " " in source:
                    # Replace spaces with underscores
                    source_path = source.replace(" ", "_")
                else:
                    source_path = source
                model_path = os.path.join(sources_models_dir, f"{source_path}.joblib")
                models_sources[source] = joblib.load(model_path)
            print("------- sources models loaded -----------")

        # Load the trained P and E models
        if P_model_dir is not None:
            model_P = joblib.load(P_model_dir)
            print("------- pleasantness model loaded -----------")
        if E_model_dir is not None:
            model_E = joblib.load(E_model_dir)
            print("------- eventfulness model loaded -----------")

        # Load the CLAP model to generate features
        # Suppress specific warnings
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="torch.meshgrid: in an upcoming release",
        )
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                model_CLAP = CLAP_Module(enable_fusion=True)
                model_CLAP.load_ckpt("data/models/630k-fusion-best.pt")
                print("------- clap model loaded -----------")
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
                        print(date_time, "- new segment")
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
                        if sources is not None:
                            for source in sources:
                                prediction = models_sources[source].predict_proba(
                                    features
                                )[0][1]
                                predictions.append(prediction)

                        # Calculate prediction values for each P/E model
                        if P_model_dir is not None:
                            prediction_P = model_P.predict(features)[0]
                            predictions.append(prediction_P)
                        if E_model_dir is not None:
                            prediction_E = model_E.predict(features)[0]
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
