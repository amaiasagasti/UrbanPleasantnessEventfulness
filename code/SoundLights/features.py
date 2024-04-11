import sys
import numpy as np
from scipy.signal import lfilter
from scipy.signal.filter_design import bilinear
from essentia.standard import FreesoundExtractor
from numpy import pi, convolve
from maad.spl import pressure2leq
from maad.util import mean_dB, dB2power, power2dB
from mosqito.sq_metrics import loudness_zwtv
from Mosqito.Sharpness import sharpness_din
from PsychoacousticParametersMeasurerAndreaCastiella.Roughness import acousticRoughness
from PsychoacousticParametersMeasurerAndreaCastiella.FluctuationStrength import (
    acousticFluctuation,
    fmoddetection,
)
from PsychoacousticParametersMeasurerAndreaCastiella.loudness_ISO532 import (
    loudness_ISO532_time,
)
from PsychoacousticParametersMeasurerAndreaCastiella.ThirdOctaveFilters import (
    ThirdOctaveLevelTime,
)
from SoundLights.freesound_auxiliary_info import (
    barkbands,
    erbbands,
    gfcc,
    melbands,
    melbands96,
    mfcc,
    spectral_contrast_coeffs,
    spectral_contrast_valleys,
    beats_loudness_band_ratio,
    tristimulus,
    hpcp,
)


sys.path.append("..")


auxiliars = {
    "barkbands": barkbands,
    "erbbands": erbbands,
    "gfcc": gfcc,
    "melbands": melbands,
    "melbands96": melbands96,
    "mfcc": mfcc,
    "spectral_contrast_coeffs": spectral_contrast_coeffs,
    "spectral_contrast_valleys": spectral_contrast_valleys,
    "beats_loudness_band_ratio": beats_loudness_band_ratio,
    "tristimulus": tristimulus,
    "hpcp": hpcp,
}
center_freq = [
    5.0,
    6.3,
    8.0,
    10.0,
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
]


def calculate_roughness(specLoudness):
    fmodR = fmoddetection(specLoudness, fmin=40, fmax=150)
    R = []
    for i in range(specLoudness.shape[1]):
        R.append(acousticRoughness(specLoudness[:, i], fmodR))
    R = [round(num, 4) for num in R]
    return R


def calculate_fluctuation(specLoudness):
    fmodFS = fmoddetection(specLoudness, fmin=0.2, fmax=64)
    FS = []
    for i in range(specLoudness.shape[1]):
        FS.append(acousticFluctuation(specLoudness[:, i], fmodFS))
    FS = [round(num, 4) for num in FS]
    return FS


def calculate_M(signal, fs):
    time_step = 1 / fs
    time_vector = np.arange(0, len(signal) * time_step, time_step)
    M = fft_hann(time_vector, signal)
    return M


def freq_limits(f_c):
    f_l = f_c / 2 ** (1 / 6)
    f_h = f_c * 2 ** (1 / 6)

    return (f_l, f_h)


def filter_band(magnitude_spectrum, frequency_axis, f_low, f_high):
    # Find indices where frequency is greater than f_low and less than f_high
    indices = np.where((frequency_axis > f_low) & (frequency_axis < f_high))[0]

    # Cut the band from both arrays using the indices
    magnitude_spectrum_cut = magnitude_spectrum[indices]
    frequency_axis_cut = frequency_axis[indices]

    return magnitude_spectrum_cut, frequency_axis_cut


def fft_hann(t, pt):
    # Signal length
    N = len(pt)
    # Window length (amount of data pts/win)
    n = 8192
    # Overlap
    overlap_ratio = 0.5
    overlap = n * overlap_ratio
    num_windows = int((N / n) / (1 - overlap_ratio)) + 1

    # prepare array where each row represents each window and each column the sum of power per band
    results = np.zeros((num_windows, len(center_freq)))
    # Frequencies
    freq = np.fft.fftfreq(n, t[1] - t[0])[0 : int(n / 2)]
    # Loop over windows
    for i in range(0, num_windows):
        # Select window of complete audio
        begin = int(i * overlap)
        end = int(begin + n)
        signal_cut = pt[begin:end]
        # Fill in with zeros last window if needed
        if signal_cut.size < n:
            add = n - signal_cut.size
            signal_cut = np.pad(signal_cut, (0, add), mode="constant")
        # Obtain fft
        spectra = np.abs(np.fft.fft(np.hanning(n) * signal_cut))
        spectra = (1 / n) * spectra
        spectra = spectra[0:4096]  # only positive half
        # Filter band and obtain the power for each band
        for index, f_c in enumerate(center_freq):
            # Calculate frequency band limits
            (f_l, f_h) = freq_limits(f_c)
            # Filter the band
            spectra_cut, freq_cut = filter_band(spectra, freq, f_l, f_h)
            # Save in results the sum of all the power of the band
            if spectra_cut.shape[0] == 0:
                results[i, index] = 0.00002  # 0dB
            else:
                results[i, index] = np.sum(spectra_cut)
    return 20 * np.log10(np.mean(results, axis=0) / 0.00002)


def A_weighting(Fs):
    """Design of an A-weighting filter.

    B, A = A_weighting(Fs) designs a digital A-weighting filter for
    sampling frequency Fs. Usage: y = lfilter(B, A, x).
    Warning: Fs should normally be higher than 20 kHz. For example,
    Fs = 48000 yields a class 1-compliant filter.

    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.

    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.

    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = convolve(
        [1, +4 * pi * f4, (2 * pi * f4) ** 2],
        [1, +4 * pi * f1, (2 * pi * f1) ** 2],
        mode="full",
    )
    DENs = convolve(
        convolve(DENs, [1, 2 * pi * f3], mode="full"), [1, 2 * pi * f2], mode="full"
    )

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, Fs)


def C_weighting(fs):
    "https://gist.github.com/endolith/148112/cbf8fea907ed3998d8469b776f7245adae862282"

    f1 = 20.598997
    f4 = 12194.217
    C1000 = 0.0619

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (C1000 / 20.0)), 0, 0]
    DENs = np.polymul(
        [1, 4 * pi * f4, (2 * pi * f4) ** 2.0], [1, 4 * pi * f1, (2 * pi * f1) ** 2]
    )
    DENs = convolve(
        [1, +4 * pi * f4, (2 * pi * f4) ** 2],
        [1, +4 * pi * f1, (2 * pi * f1) ** 2],
        mode="full",
    )

    # Use the bilinear transformation to get the digital filter.
    return bilinear(NUMs, DENs, fs)


def var_dB(vector, axis):

    # dB to energy as sum has to be done with energy
    e = dB2power(vector)
    e_var = np.var(e, axis)

    # energy (power) => dB
    e_var = power2dB(e_var)

    return e_var


def calculate_stats(vector: np.array, stats: list, descriptor=None):
    output = {}
    for i, stat in enumerate(stats):
        if stat == "avg":
            # Calculate statistic (axis 0 corresponds to time)
            avg = np.round(np.mean(vector, axis=0), 4)
            if type(avg) == np.ndarray:
                # We have bands/frequencies/coefficients --> save in separate keys
                output["avg"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], avg)
                }
            else:
                # Just a value
                output["avg"] = float(avg)
        if stat == "median":
            median = np.round(np.median(vector, axis=0), 4)
            if type(median) == np.ndarray:
                output["median"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], median)
                }
            else:
                output["median"] = float(median)
        if stat == "var":
            var = np.round(np.var(vector, axis=0), 4)
            if type(var) == np.ndarray:
                output["var"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], var)
                }
            else:
                output["var"] = float(var)
        if stat == "max":
            max = np.round(np.max(vector, axis=0), 4)
            if type(max) == np.ndarray:
                output["max"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], max)
                }
            else:
                output["max"] = float(max)
        if stat == "min":
            min = np.round(np.min(vector, axis=0), 4)
            if type(min) == np.ndarray:
                output["min"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], min)
                }
            else:
                output["min"] = float(min)
        if stat == "p05":
            p05 = np.round(np.percentile(vector, 95, axis=0), 4)
            if type(p05) == np.ndarray:
                output["p05"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p05)
                }
            else:
                output["p05"] = float(p05)
        if stat == "p10":
            p10 = np.round(np.percentile(vector, 90, axis=0), 4)
            if type(p10) == np.ndarray:
                output["p10"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p10)
                }
            else:
                output["p10"] = float(p10)
        if stat == "p20":
            p20 = np.round(np.percentile(vector, 80, axis=0), 4)
            if type(p20) == np.ndarray:
                output["p20"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p20)
                }
            else:
                output["p20"] = float(p20)
        if stat == "p30":
            p30 = np.round(np.percentile(vector, 70, axis=0), 4)
            if type(p30) == np.ndarray:
                output["p30"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p30)
                }
            else:
                output["p30"] = float(p30)
        if stat == "p40":
            p40 = np.round(np.percentile(vector, 60, axis=0), 4)
            if type(p40) == np.ndarray:
                output["p40"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p40)
                }
            else:
                output["p40"] = float(p40)
        if stat == "p50":
            p50 = np.round(np.percentile(vector, 50, axis=0), 4)
            if type(p50) == np.ndarray:
                output["p50"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p50)
                }
            else:
                output["p50"] = float(p50)
        if stat == "p60":
            p60 = np.round(np.percentile(vector, 40, axis=0), 4)
            if type(p60) == np.ndarray:
                output["p60"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p60)
                }
            else:
                output["p60"] = float(p60)
        if stat == "p70":
            p70 = np.round(np.percentile(vector, 30, axis=0), 4)
            if type(p70) == np.ndarray:
                output["p70"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p70)
                }
            else:
                output["p70"] = float(p70)
        if stat == "p80":
            p80 = np.round(np.percentile(vector, 20, axis=0), 4)
            if type(p80) == np.ndarray:
                output["p80"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p80)
                }
            else:
                output["p80"] = float(p80)
        if stat == "p90":
            p90 = np.round(np.percentile(vector, 10, axis=0), 4)
            if type(p90) == np.ndarray:
                output["p90"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p90)
                }
            else:
                output["p90"] = float(p90)
        if stat == "p95":
            p95 = np.round(np.percentile(vector, 5, axis=0), 4)
            if type(p95) == np.ndarray:
                output["p95"] = {
                    str(key): float(value)
                    for key, value in zip(auxiliars[descriptor], p95)
                }
            else:
                output["p95"] = float(p95)
    return output


def extract_ARAUS_features(signal: np.array, fs: float, feature_list: list):

    # Check if feature list is empty
    if len(feature_list) == 0:
        raise ValueError("List of features is empty, please provide.")

    # Prepare output
    output = {}

    # Initialize to zero data(loudnes value, and A-weigth filter) that is
    #  re-used in several feature calculations
    N = None
    A_A = None
    B_A = None

    # Stats to calculate
    stats = [
        "avg",
        "min",
        "max",
        "median",
        "var",
        "p05",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p95",
    ]
    stats_dB = [
        "avgdB",
        "vardB",
        "min",
        "max",
        "median",
        "p05",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p95",
    ]

    # Go over list of desired features
    for i, feature in enumerate(feature_list):
        if feature == "loudness":
            print("Calculating loudness")
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                signal, fs, field_type="free"
            )
            loudness_data = calculate_stats(N, stats)
            output["loudness"] = loudness_data

        if feature == "sharpness":
            print("Calculating sharpness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            S, S_time_axis = sharpness_din(N, N_spec, time_axis, 0.5)
            sharpness_data = calculate_stats(S, stats)
            output["sharpness"] = sharpness_data

        if feature == "LA":
            print("Calculating LA")
            [B_A, A_A] = A_weighting(fs)
            signal_A = lfilter(B_A, A_A, signal)
            LAeq = pressure2leq(signal_A, fs, 0.125)
            LA_data = calculate_stats(LAeq, stats)
            output["LA"] = LA_data

        if feature == "LC":
            print("Calculating LC")
            [B_C, A_C] = C_weighting(fs)
            signal_C = lfilter(B_C, A_C, signal)
            LCeq = pressure2leq(signal_C, fs, 0.125)
            LC_data = calculate_stats(LCeq, stats)
            output["LC"] = LC_data

        if feature == "roughness":
            print("Calculating roughness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            R = calculate_roughness(N_spec)
            roughness_data = calculate_stats(R, stats)
            output["roughness"] = roughness_data

        if feature == "fluctuation":
            print("Calculating fluctuation strength")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            FS = calculate_fluctuation(N_spec)
            fluctuation_data = calculate_stats(FS, stats)
            output["fluctuation"] = fluctuation_data

        if feature == "frequency":
            print("Calculating frequency features")
            if B_A is None:
                [B_A, A_A] = A_weighting(fs)
                signal_A = lfilter(B_A, A_A, signal)
            M_values = calculate_M(signal_A, fs)
            output["energy_frequency"] = {
                "00005_0": np.round(M_values[0], 4),
                "00006_3": np.round(M_values[1], 4),
                "00008_0": np.round(M_values[2], 4),
                "00010_0": np.round(M_values[3], 4),
                "00012_5": np.round(M_values[4], 4),
                "00016_0": np.round(M_values[5], 4),
                "00020_0": np.round(M_values[6], 4),
                "00025_0": np.round(M_values[7], 4),
                "00031_5": np.round(M_values[8], 4),
                "00040_0": np.round(M_values[9], 4),
                "00050_0": np.round(M_values[10], 4),
                "00063_0": np.round(M_values[11], 4),
                "00080_0": np.round(M_values[12], 4),
                "00100_0": np.round(M_values[13], 4),
                "00125_0": np.round(M_values[14], 4),
                "00160_0": np.round(M_values[15], 4),
                "00200_0": np.round(M_values[16], 4),
                "00250_0": np.round(M_values[17], 4),
                "00315_0": np.round(M_values[18], 4),
                "00400_0": np.round(M_values[19], 4),
                "00500_0": np.round(M_values[20], 4),
                "00630_0": np.round(M_values[21], 4),
                "00800_0": np.round(M_values[22], 4),
                "01000_0": np.round(M_values[23], 4),
                "01250_0": np.round(M_values[24], 4),
                "01600_0": np.round(M_values[25], 4),
                "02000_0": np.round(M_values[26], 4),
                "02500_0": np.round(M_values[27], 4),
                "03150_0": np.round(M_values[28], 4),
                "04000_0": np.round(M_values[29], 4),
                "05000_0": np.round(M_values[30], 4),
                "06300_0": np.round(M_values[31], 4),
                "08000_0": np.round(M_values[32], 4),
                "10000_0": np.round(M_values[33], 4),
                "12500_0": np.round(M_values[34], 4),
                "16000_0": np.round(M_values[35], 4),
                "20000_0": np.round(M_values[36], 4),
            }

    return output


def run_freesound_extractor(audiofile):
    """Runs Essentia standard FreesoundExtractor
    audiofile: absolute path to the audio file to analyze
    """

    parameters = {
        "analysisSampleRate": 48000,
        "startTime": 0,
        "endTime": 30,
        "lowlevelFrameSize": 2048,
        "lowlevelHopSize": 1024,
        "lowlevelSilentFrames": "noise",
        "lowlevelStats": ["mean"],
        "lowlevelWindowType": "hann",
        "lowlevelZeroPadding": 0,
        "mfccStats": ["mean"],
        "gfccStats": ["mean"],
        "rhythmMaxTempo": 208,
        "rhythmMethod": "degara",
        "rhythmMinTempo": 40,
        "rhythmStats": ["mean"],
        "tonalFrameSize": 2048,
        "tonalHopSize": 1024,
        "tonalSilentFrames": "noise",
        "tonalStats": ["mean"],
        "tonalWindowType": "hann",
        "tonalZeroPadding": 0,
    }

    try:
        result, resultFrames = FreesoundExtractor(**parameters)(audiofile)
    except RuntimeError as e:
        raise e
    return result, resultFrames


def extract_Freesound_features(input):
    # Run the essentia standard FreesoundExtractor
    _, frames = run_freesound_extractor(input)

    # Calculate statistics and save all available features in a dictionary
    features = {}
    for descriptor in frames.descriptorNames():
        d = descriptor.split(".")
        # Calculate stats for arrays
        if type(frames[descriptor]) == np.ndarray:
            stats = [
                "avg",
                "median",
                "var",
                "min",
                "max",
                "p05",
                "p10",
                "p20",
                "p30",
                "p40",
                "p50",
                "p60",
                "p70",
                "p80",
                "p90",
                "p95",
            ]
            stats_result = calculate_stats(frames[descriptor], stats, d[1])
            # Check if d[0] key exists
            if d[0] not in features:
                features[d[0]] = {}
            # Now you can safely assign value to nested dictionary
            features[str(d[0])][str(d[1])] = stats_result
        # Otherwise just save in dictionary
        else:
            # Check if d[0] key exists
            if d[0] not in features:
                features[d[0]] = {}
            features[str(d[0])][str(d[1])] = frames[descriptor]

    # Metadata is returned, unwanted
    del features["metadata"]

    return features
