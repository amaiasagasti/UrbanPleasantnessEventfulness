from scipy.io import wavfile
import numpy as np
import os


def save_wav(signal, fs, filepath):
    """
    Save the signal to a WAV file.

    Parameters:
    - signal: NumPy array representing the audio signal
    - fs: Sampling frequency of the signal
    - filepath: Path to save the WAV file
    """
    # Check if the signal needs to be converted to int16
    if signal.dtype != np.int16:
        # Scale the signal to the range [-32768, 32767] (16-bit signed integer)
        scaled_signal = np.int16(signal * 32767)
    else:
        scaled_signal = signal

    # Save the WAV file
    wavfile.write(filepath, fs, scaled_signal)


def delete_wav(filepath):
    """
    Delete a WAV file.

    Parameters:
    - filepath: Path to the WAV file to be deleted
    """
    try:
        os.remove(filepath)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    except Exception as e:
        print(f"Error deleting file '{filepath}': {e}")
