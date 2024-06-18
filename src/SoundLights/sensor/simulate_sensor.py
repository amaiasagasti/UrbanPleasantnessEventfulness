import pyaudio
import wave
import threading
import time
import datetime
import json
import pickle
import os


# Function to play audio in real-time and save segments
def play_audio(filename):
    # Frames will store the chunks to be stored
    frames = []
    # Will be overwritten, but it is required to be declared as a time object from now already
    prev_time = time.time()

    #########################
    # Function to save segments, it will be a thread that will be active all the time
    def save_segment():
        # It works with the nonlocal variables prev_time(to check when to save) and frames (data to save)
        nonlocal prev_time, frames
        while True:
            # Check when to save --> every 3 seconds
            if (time.time() - prev_time) >= 3:
                # Prepare files names with current date-time data
                date_time = datetime.datetime.now()
                time_str = date_time.strftime("%Y%m%d_%H%M%S")
                file_name = f"segment_{time_str}"

                # Save WAV file
                wav_file_path = os.path.join(save_directory, f"{file_name}.wav")
                wf_segment = wave.open(wav_file_path, "wb")
                wf_segment.setnchannels(wf.getnchannels())
                wf_segment.setsampwidth(wf.getsampwidth())
                wf_segment.setframerate(wf.getframerate())
                wf_segment.writeframes(b"".join(frames))
                wf_segment.close()

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
                    pickle.dump({"audio": frames, "info": json_info}, f)

                frames = []

    # Start the thread to save audio segments every 3 seconds
    threading.Thread(target=save_segment, daemon=True).start()
    #########################

    # Read audio file, import it
    wf = wave.open(filename, "rb")
    fs = wf.getframerate()
    # Prepare audio player and stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=fs,
        output=True,
    )
    # Prepare analysis
    chunk_size = 3 * fs  # 3 because i want 3 seconds slots
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
        print(i)
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


# if __name__ == "__main__":
play_audio("data/listening_test_audios/freesound_23063.wav")
