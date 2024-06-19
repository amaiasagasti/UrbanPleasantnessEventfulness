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


# Function to play audio in real-time and save segments
def play_audio(filename, seconds, maintain_time):
    # Frames will store the chunks to be stored
    frames = []
    # Will be overwritten, but it is required to be declared as a time object from now already
    prev_time = time.time()
    i = 0
    prev_i = 0

    #########################
    # Function to save segments, it will be a thread that will be active all the time
    def save_segment():
        # It works with the nonlocal variables prev_time(to check when to save) and frames (data to save)
        nonlocal prev_time, frames, seconds, i, prev_i
        while True:
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
                json_file_path = os.path.join(save_directory, f"{file_name}.json")
                with open(json_file_path, "w") as f:
                    json.dump(json_info, f, indent=4)

                # Save audio and JSON info in a pickle file
                pickle_file_path = os.path.join(save_directory, f"{file_name}.pkl")
                with open(pickle_file_path, "wb") as f:
                    pickle.dump({"audio": audio_data, "info": json_info}, f)  # frames

                frames = []
                prev_i = i

    # Start the thread to save audio segments every 3 seconds
    threading.Thread(target=save_segment, daemon=True).start()
    #########################

    #########################
    # Function to delete old segments, it will be a thread that will be active all the time
    def cleanup_old_files(folder_path, maintain_time):
        """Cleanup old files from a specified folder based on their age.

        Parameters:
        - folder_path (str): Path to the folder containing the files.
        - maintain_time (int): Maximum age (in seconds) beyond which files are considered old and eligible for deletion.
        """

        nonlocal prev_time, seconds
        while True:
            if (time.time() - prev_time) >= seconds:
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

                            print(f"Deleted old files: {file_path}, {json_file_path}")
                        except Exception as e:
                            print(
                                f"Error deleting files: {file_path}, {json_file_path}, Error: {str(e)}"
                            )

    threading.Thread(
        target=cleanup_old_files, daemon=True, args=(["segments/", maintain_time])
    ).start()
    #########################

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


# if __name__ == "__main__":
play_audio(
    "data/listening_test_audios_32bit_simulation/freesound_23063_16bit.wav",
    seconds=3,
    maintain_time=35,
)
