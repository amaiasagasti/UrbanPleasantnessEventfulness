import sys
import numpy as np
from scipy.signal import lfilter
from scipy.signal.filter_design import bilinear
from essentia.standard import FreesoundExtractor
from numpy import pi, convolve
from maad.spl import pressure2leq
from maad.util import mean_dB
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


sys.path.append("..")


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


def calculate_stats(vector: np.array, stats: list, descriptor_name):
    output = {}
    for i, stat in enumerate(stats):
        if stat == "avg":
            output[descriptor_name + "_avg"] = np.round(np.mean(vector, axis=0), 4)
        if stat == "median":
            output[descriptor_name + "_median"] = np.round(np.median(vector, axis=0), 4)
        if stat == "var":
            output[descriptor_name + "_var"] = np.round(np.var(vector, axis=0), 4)
        if stat == "avgdB":
            output[descriptor_name + "_avg"] = np.round(mean_dB(vector), 4)
        if stat == "max":
            output[descriptor_name + "_max"] = np.round(np.max(vector, axis=0), 4)
        if stat == "min":
            output[descriptor_name + "_min"] = np.round(np.min(vector, axis=0), 4)
        if stat == "p05":
            output[descriptor_name + "_p05"] = np.round(
                np.percentile(vector, 95, axis=0), 4
            )
        if stat == "p10":
            output[descriptor_name + "_p10"] = np.round(
                np.percentile(vector, 90, axis=0), 4
            )
        if stat == "p20":
            output[descriptor_name + "_p20"] = np.round(
                np.percentile(vector, 80, axis=0), 4
            )
        if stat == "p30":
            output[descriptor_name + "_p30"] = np.round(
                np.percentile(vector, 70, axis=0), 4
            )
        if stat == "p40":
            output[descriptor_name + "_p40"] = np.round(
                np.percentile(vector, 60, axis=0), 4
            )
        if stat == "p50":
            output[descriptor_name + "_p50"] = np.round(
                np.percentile(vector, 50, axis=0), 4
            )
        if stat == "p60":
            output[descriptor_name + "_p60"] = np.round(
                np.percentile(vector, 40, axis=0), 4
            )
        if stat == "p70":
            output[descriptor_name + "_p70"] = np.round(
                np.percentile(vector, 30, axis=0), 4
            )
        if stat == "p80":
            output[descriptor_name + "_p80"] = np.round(
                np.percentile(vector, 20, axis=0), 4
            )
        if stat == "p90":
            output[descriptor_name + "_p90"] = np.round(
                np.percentile(vector, 10, axis=0), 4
            )
        if stat == "p95":
            output[descriptor_name + "_p95"] = np.round(
                np.percentile(vector, 5, axis=0), 4
            )
    return output


def extract_features(signal: np.array, fs: float, feature_list: list):

    # Check if feature list is empty
    if len(feature_list) == 0:
        raise ValueError("List of features is empty, please provide.")

    # Prepare output
    """ output = {
        "Savg": 0,  # Sharpness
        "Smax": 0,
        "S05": 0,
        "S10": 0,
        "S20": 0,
        "S30": 0,
        "S40": 0,
        "S50": 0,
        "S60": 0,
        "S70": 0,
        "S80": 0,
        "S90": 0,
        "S95": 0,
        "Navg": 0,  # Loudness
        "Nrmc": 0,
        "Nmax": 0,
        "N05": 0,
        "N10": 0,
        "N20": 0,
        "N30": 0,
        "N40": 0,
        "N50": 0,
        "N60": 0,
        "N70": 0,
        "N80": 0,
        "N90": 0,
        "N95": 0,
        "Favg": 0,  # Fluctuation Strength
        "Fmax": 0,
        "F05": 0,
        "F10": 0,
        "F20": 0,
        "F30": 0,
        "F40": 0,
        "F50": 0,
        "F60": 0,
        "F70": 0,
        "F80": 0,
        "F90": 0,
        "F95": 0,
        "LAavg": 0,  # LA
        "LAmin": 0,
        "LAmax": 0,
        "LA05": 0,
        "LA10": 0,
        "LA20": 0,
        "LA30": 0,
        "LA40": 0,
        "LA50": 0,
        "LA60": 0,
        "LA70": 0,
        "LA80": 0,
        "LA90": 0,
        "LA95": 0,
        "LCavg": 0,  # LC
        "LCmin": 0,
        "LCmax": 0,
        "LC05": 0,
        "LC10": 0,
        "LC20": 0,
        "LC30": 0,
        "LC40": 0,
        "LC50": 0,
        "LC60": 0,
        "LC70": 0,
        "LC80": 0,
        "LC90": 0,
        "LC95": 0,
        "Ravg": 0,  # Roughness
        "Rmax": 0,
        "R05": 0,
        "R10": 0,
        "R20": 0,
        "R30": 0,
        "R40": 0,
        "R50": 0,
        "R60": 0,
        "R70": 0,
        "R80": 0,
        "R90": 0,
        "R95": 0,
        "Tgavg": 0,  # Tonality
        "Tavg": 0,
        "Tmax": 0,
        "T05": 0,
        "T10": 0,
        "T20": 0,
        "T30": 0,
        "T40": 0,
        "T50": 0,
        "T60": 0,
        "T70": 0,
        "T80": 0,
        "T90": 0,
        "T95": 0,
        "M00005_0": 0,  # Frequency
        "M00006_3": 0,
        "M00008_0": 0,
        "M00010_0": 0,
        "M00012_5": 0,
        "M00016_0": 0,
        "M00020_0": 0,
        "M00025_0": 0,
        "M00031_5": 0,
        "M00040_0": 0,
        "M00050_0": 0,
        "M00063_0": 0,
        "M00080_0": 0,
        "M00100_0": 0,
        "M00125_0": 0,
        "M00160_0": 0,
        "M00200_0": 0,
        "M00250_0": 0,
        "M00315_0": 0,
        "M00400_0": 0,
        "M00500_0": 0,
        "M00630_0": 0,
        "M00800_0": 0,
        "M01000_0": 0,
        "M01250_0": 0,
        "M01600_0": 0,
        "M02000_0": 0,
        "M02500_0": 0,
        "M03150_0": 0,
        "M04000_0": 0,
        "M05000_0": 0,
        "M06300_0": 0,
        "M08000_0": 0,
        "M10000_0": 0,
        "M12500_0": 0,
        "M16000_0": 0,
        "M20000_0": 0,
    }
     """
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
            loudness_data = calculate_stats(N, stats, "N")
            output = {**output, **loudness_data}

        if feature == "sharpness":
            print("Calculating sharpness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            S, S_time_axis = sharpness_din(N, N_spec, time_axis, 0.5)
            sharpness_data = calculate_stats(S, stats, "S")
            output = {**output, **sharpness_data}

        if feature == "LA":
            print("Calculating LA")
            [B_A, A_A] = A_weighting(fs)
            signal_A = lfilter(B_A, A_A, signal)
            LAeq = pressure2leq(signal_A, fs, 0.125)
            LA_data = calculate_stats(LAeq, stats_dB, "LA")
            output = {**output, **LA_data}

        if feature == "LC":
            print("Calculating LC")
            [B_C, A_C] = C_weighting(fs)
            signal_C = lfilter(B_C, A_C, signal)
            LCeq = pressure2leq(signal_C, fs, 0.125)
            LC_data = calculate_stats(LCeq, stats_dB, "LC")
            output = {**output, **LC_data}

        if feature == "roughness":
            print("Calculating roughness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            R = calculate_roughness(N_spec)
            roughness_data = calculate_stats(R, stats, "R")
            output = {**output, **roughness_data}

        if feature == "fluctuation":
            print("Calculating fluctuation strength")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            FS = calculate_fluctuation(N_spec)
            fluctuation_data = calculate_stats(FS, stats, "F")
            output = {**output, **fluctuation_data}

        if feature == "frequency":
            print("Calculating frequency features")
            if B_A is None:
                [B_A, A_A] = A_weighting(fs)
                signal_A = lfilter(B_A, A_A, signal)
            M_values = calculate_M(signal_A, fs)
            output["M"] = [
                np.round(M_values[0], 4),
                np.round(M_values[1], 4),
                np.round(M_values[2], 4),
                np.round(M_values[3], 4),
                np.round(M_values[4], 4),
                np.round(M_values[5], 4),
                np.round(M_values[6], 4),
                np.round(M_values[7], 4),
                np.round(M_values[8], 4),
                np.round(M_values[9], 4),
                np.round(M_values[10], 4),
                np.round(M_values[11], 4),
                np.round(M_values[12], 4),
                np.round(M_values[13], 4),
                np.round(M_values[14], 4),
                np.round(M_values[15], 4),
                np.round(M_values[16], 4),
                np.round(M_values[17], 4),
                np.round(M_values[18], 4),
                np.round(M_values[19], 4),
                np.round(M_values[20], 4),
                np.round(M_values[21], 4),
                np.round(M_values[22], 4),
                np.round(M_values[23], 4),
                np.round(M_values[24], 4),
                np.round(M_values[25], 4),
                np.round(M_values[26], 4),
                np.round(M_values[27], 4),
                np.round(M_values[28], 4),
                np.round(M_values[29], 4),
                np.round(M_values[30], 4),
                np.round(M_values[31], 4),
                np.round(M_values[32], 4),
                np.round(M_values[33], 4),
                np.round(M_values[34], 4),
                np.round(M_values[35], 4),
                np.round(M_values[36], 4),
            ]

    return output


parameters = {
    "analysisSampleRate": 48000,
    "startTime": 0,
    "endTime": 30,
    "lowlevelFrameSize": 2048,
    "lowlevelHopSize": 1024,
    "lowlevelSilentFrames": "noise",
    "lowlevelStats": ["mean"],  # "max", "dmean", "dmean2", "dvar", "dvar2"
    "lowlevelWindowType": "blackmanharris62",
    "lowlevelZeroPadding": 0,
    "mfccStats": ["mean"],
    "gfccStats": ["mean"],
    "rhythmMaxTempo": 208,
    "rhythmMethod": "degara",
    "rhythmMinTempo": 40,
    "rhythmStats": [
        "mean"
    ],  # "mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2"
    "tonalFrameSize": 4096,
    "tonalHopSize": 1024,
    "tonalSilentFrames": "noise",
    "tonalStats": ["mean"],
    "tonalWindowType": "blackmanharris62",
    "tonalZeroPadding": 0,
}


def run_freesound_extractor(audiofile):
    """Runs Essentia standard FreesoundExtractor
    :audiofile: absolute path to the audio file to analyze
    """
    try:
        result, resultFrames = FreesoundExtractor(**parameters)(audiofile)
    except RuntimeError as e:
        raise e
    return result, resultFrames


def analyze(input):
    # Run the essentia standard FreesoundExtractor
    result, frames = run_freesound_extractor(input)
    # Calculate statistics and save all available features in a dictionary
    features = dict()
    for descriptor in frames.descriptorNames():
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
            stats_result = calculate_stats(frames[descriptor], stats, descriptor)
            features = {**features, **stats_result}
        else:
            features[descriptor] = frames[descriptor]

    return features
