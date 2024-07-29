"""
This script contains a set of auxiliary functions, adapted from other libraries 
for personal use. Specifically:

- Two functions are adapted from the Mosqito library.
- The rest are obtained from a GitHub project developed by Andrea Castiella Aguirrezabala.

These adaptations have been made to suit our specific needs while retaining the core 
functionality of the original implementations. 

Full credit is given to the original authors.
"""

from scipy.signal import resample
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import numpy as np
import warnings
from scipy import signal
from mosqito.sq_metrics import sharpness_din_from_loudness


# FUNCTIONS FROM MOSQITO LIBRARY ################################################


def load(file, wav_calib=None, ch=None):
    """
    Extract the peak-pressure signal of chosen channel from .wav
    or .uff file and resample the signal to 48 kHz.

    Parameters
    ----------
    file : string
        Path to the signal file
    wav_calib : float, optional
        Wav file calibration factor [Pa/FS]. Level of the signal in Pa_peak
        corresponding to the full scale of the .wav file. If None, a
        calibration factor of 1 is considered. Default to None.
    ch : int, optional for mono files
        Channel chosen

    Outputs
    -------
    signal : numpy.array
        time signal values
    fs : integer
        sampling frequency
    """

    # Suppress WavFileWarning
    warnings.filterwarnings("ignore", category=WavFileWarning)

    # load the .wav file content
    if file[-3:] == "wav" or file[-3:] == "WAV":
        fs, signal = wavfile.read(file)

        # manage multichannel files
        if signal.ndim > 1:
            signal = signal[
                :, ch
            ]  # MODIFICATION: instead of taking channel-0 directly, choose

        # calibration factor for the signal to be in Pa
        if wav_calib is None:
            wav_calib = 1
            print("[Info] A calibration of 1 Pa/FS is considered")
        if isinstance(signal[0], np.int16):
            signal = wav_calib * signal / (2**15 - 1)
        elif isinstance(signal[0], np.int32):
            signal = wav_calib * signal / (2**31 - 1)
        elif isinstance(signal[0], np.float):
            signal = wav_calib * signal

    else:
        raise ValueError("""ERROR: only .wav .mat or .uff files are supported""")

    # resample to 48kHz to allow calculation
    if fs != 48000:
        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000

    return signal, fs


def sharpness_din(N, N_spec, time_axis, skip):
    """
    Calculate the sharpness of a signal based on its loudness using
    the DIN 45692 standard. This function computes the sharpness
    from the given loudness and spectral data, and then removes the
    transient effect by skipping a specified initial period.

    Parameters:
    ----------
    N : array-like
        Loudness data of the signal.
    N_spec : array-like
        Spectral data of the signal.
    time_axis : array-like
        Time axis corresponding to the data.
    skip : float
        Time in seconds to skip for removing the transient effect.

    Outputs:
    -------
    S_cut : array-like
        Sharpness values
    time_axis_cut : array-like
        Time axis
    """

    # Get sharpness from loudness using mosqito's function
    S = sharpness_din_from_loudness(N, N_spec, "din", 0)

    # Cut transient effect
    cut_index = np.argmin(np.abs(time_axis - skip))

    return S[cut_index:], time_axis[cut_index:]


#################################################################################


# FUNCTIONS FROM PsychoacousticParametersMeasurer by Andrea Castiella ###########
# https://github.com/AndreaCastiella/PsychoacousticParametersMeasurer


def acousticRoughness(specificLoudness, fmod):
    """
    Calculate the acoustic roughness of a signal. This function computes the
    acoustic roughness based on the specific loudness and modulation frequency of
    the signal. The roughness is calculated by finding the absolute difference
    in specific loudness between consecutive samples and applying a scaling factor
    based on the modulation frequency. Approximation of Roughness model proposed
    by Fastl H. and Zwicker E. in "Psychoacoustics. Facts and Models".

    Parameters
    ----------
    specificLoudness : array-like
        An array containing the specific loudness values of the signal.
    fmod : float
        The modulation frequency in Hz.

    Outputs
    -------
    R : float
        The calculated acoustic roughness value.
    """
    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
    R = 0.3 * (fmod / 1000) * sum(0.1 * specificLoudnessdiff)
    return R


def acousticFluctuation(specificLoudness, fmod):
    """
    Calculate the acoustic fluctuation of a signal. This function computes the
    acoustic fluctuation based on the specific loudness and modulation frequency of
    the signal. The fluctuation is calculated by finding the absolute difference in
    specific loudness between consecutive samples and applying a scaling factor
    based on the modulation frequency. Approximation of Fluctuation model proposed
    by Fastl H. and Zwicker E. in "Psychoacoustics. Facts and Models".

    Parameters
    ----------
    specificLoudness : array-like
        An array containing the specific loudness values of the signal.
    fmod : float
        The modulation frequency in Hz.

    Outputs
    -------
    F : float
        The calculated acoustic fluctuation value.
    """

    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
    F = (0.008 * sum(0.1 * specificLoudnessdiff)) / ((fmod / 4) + (4 / fmod))
    return F


def fmoddetection(specificLoudness, fmin=0.2, fmax=64):
    """
    Detect the modulation frequency of a signal based on its specific loudness.
    This function estimates the modulation frequency by transforming the specific
    loudness into the phon scale, reshaping the data into overlapping bands,
    applying a high-pass filter, and performing a Fast Fourier Transform (FFT).
    It then identifies the modulation frequency by finding the peak in the FFT spectrum.

    Parameters
    ----------
    specificLoudness : array-like
        An array containing the specific loudness values of the signal.
    fmin : float, optional
        The minimum detection frequency in Hz. Default is 0.2 Hz.
    fmax : float, optional
        The maximum detection frequency in Hz. Default is 64 Hz.

    Outputs
    -------
    cf : float
        The detected modulation frequency in Hz.
    """

    if len(specificLoudness.shape) == 1:
        specificLoudness = np.reshape(specificLoudness, (1, len(specificLoudness)))
    else:
        specificLoudness = np.transpose(specificLoudness)
    phon = np.zeros((specificLoudness.shape[0], specificLoudness.shape[1], 1))
    for i, ms in enumerate(specificLoudness):
        for j, bark in enumerate(ms):
            phon[i, j, 0] = sone2phon(bark)
    FSpec = 1 / 0.002

    # Reagrupar saluda en bandas de 24 o 47 Bark
    phon1024a = np.reshape(phon, newshape=(24, phon.shape[0], 10))
    # Overlap
    phon1024b = np.reshape(
        phon[:, 5:235], newshape=(23, phon.shape[0], 10)
    )  # Deja fuera los primeros 5 y los últimos 5
    h = np.hamming(10)
    phonB = np.zeros((phon.shape[0], 47))
    phonBtempa = np.sum((phon1024a * h), 2)
    phonB[:, 0:47:2] = np.transpose(phonBtempa)
    phonBtempb = np.sum((phon1024b * h), 2)
    phonB[:, 1:46:2] = np.transpose(phonBtempb)

    phonBm = phonB - (np.mean(phonB, 0))  # Eliminar media
    phonBm = np.maximum(0, phonBm)  # Eliminar parte negativa
    pbfstd = 5 * (np.std(phonB, 0, ddof=1))  # Recortar amplitudes extremas
    phonBm = np.minimum(phonBm, pbfstd)
    phonBf = hpfilt(phonBm)  # Eliminar bajas frecuencias
    pbfstd2 = np.std(phonBf, 0, ddof=1)  # Recortar amplitudes extremas
    phonBf = np.minimum(phonBf, pbfstd2)
    phonBf = np.maximum(phonBf, -pbfstd2)
    NP = np.maximum(8192, 2 ** (next_power_of_2(specificLoudness.shape[0])))
    if not NP <= np.maximum(8192, 2 * specificLoudness.shape[0]):
        print("Error")
        return None
    fmin = int(np.floor(NP * fmin / FSpec))  # Frecuncia de detección mínima
    fmax = int(np.ceil(NP * fmax / FSpec))  # Frecuencia de detección máxima
    X = np.sum(np.abs(np.fft.fft(phonBf, int(NP), axis=0)), 1)
    if not X.shape[0] <= NP:
        print("Error")
        return None
    nf = fmax - fmin + 1
    t = np.arange(fmin, fmax + 1)
    b = np.matmul(
        np.linalg.pinv(np.stack((np.ones(nf), t), axis=1)),
        np.log(np.maximum(X[fmin - 1 : fmax], np.finfo(float).tiny)),
    )  # Evitar log(0)
    XR = np.exp(b[1] * t + b[0])
    idx = np.argmax(X[fmin - 1 : fmax] - XR)
    idx = idx + fmin - 1
    cf = (idx) * FSpec / (NP - 1)
    if idx > 4:
        cf = refineCF(cf, X[idx - 3 : idx + 2], NP, FSpec)
    return cf


# others (required for above)


def sone2phon(Loudness):
    if Loudness >= 1:
        LN = 40 + 33.22 * np.log10(Loudness)
    else:
        LN = 40 * np.power(Loudness + 0.0005, 0.35)
    return LN


def hpfilt(x):
    sos = np.array(
        [
            [
                0.998195566288485491845960950741,
                -1.996391132576970983691921901482,
                0.998195566288485491845960950741,
                1,
                -1.994897079594937228108619819977,
                0.994904703139997681482498137484,
            ],
            [
                0.998195566288485491845960950741,
                -1.996391132576970983691921901482,
                0.998195566288485491845960950741,
                1,
                -1.997878669564554510174048118643,
                0.997886304503829646428414434922,
            ],
        ]
    )
    fn = 10**4
    xL = np.vstack((x, np.zeros((fn, x.shape[1]))))
    xLf = np.flipud(signal.sosfilt(sos, np.flipud(signal.sosfilt(sos, xL))))
    return xLf[0 : x.shape[0], :]


def next_power_of_2(x):
    return 1 if x == 0 else np.log2(2 ** (x - 1).bit_length())


def refineCF(cf, p, NP, FSpec):
    rd_denom = p[0] + p[4] - 16 * (p[1] + p[3]) + 30 * p[2]
    if not rd_denom == 0:
        rd = (p[0] - p[4] + 8 * (p[3] - p[1])) / rd_denom
        if np.abs(rd) < 0.5:
            cf = cf + rd * FSpec / (NP - 1)
        else:
            p = p[1:4]
            if p[0] - 2 * p[1] + p[2] < 0:
                rd_denom = 2 * (p[0] + p[2] - 2 * p[1])
                if not rd_denom == 0:
                    rd = 3 * p[0] - 4 * p[1] + p[2]
                    if np.abs(rd) < 0.5:
                        cf = cf + rd * FSpec / (NP - 1)
    return cf


#################################################################################
