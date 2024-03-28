import sys
import numpy as np
from scipy.signal import lfilter
from scipy.signal.filter_design import bilinear
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


def calculate_stats(vector: np.array, stats: list):
    output = {}
    for i, stat in enumerate(stats):
        if stat == "avg":
            output["avg"] = np.mean(vector)
        if stat == "avgdB":
            output["avgdB"] = mean_dB(vector)
        if stat == "rmc":
            output["rmc"] = np.sqrt(np.mean(np.power(vector, 3)))
        if stat == "max":
            output["max"] = np.max(vector)
        if stat == "min":
            output["min"] = np.min(vector)
        if stat == "p05":
            output["p05"] = np.percentile(vector, 95)
        if stat == "p10":
            output["p10"] = np.percentile(vector, 90)
        if stat == "p20":
            output["p20"] = np.percentile(vector, 80)
        if stat == "p30":
            output["p30"] = np.percentile(vector, 70)
        if stat == "p40":
            output["p40"] = np.percentile(vector, 60)
        if stat == "p50":
            output["p50"] = np.percentile(vector, 50)
        if stat == "p60":
            output["p60"] = np.percentile(vector, 40)
        if stat == "p70":
            output["p70"] = np.percentile(vector, 30)
        if stat == "p80":
            output["p80"] = np.percentile(vector, 20)
        if stat == "p90":
            output["p90"] = np.percentile(vector, 10)
        if stat == "p95":
            output["p95"] = np.percentile(vector, 5)
    return output


def extract_features(signal: np.array, fs: float, feature_list: list):

    # Check if feature list is empty
    if len(feature_list) == 0:
        raise ValueError("List of features is empty, please provide.")

    # Prepare output
    output = {
        "Savg_r": 0,  # Sharpness
        "Smax_r": 0,
        "S05_r": 0,
        "S10_r": 0,
        "S20_r": 0,
        "S30_r": 0,
        "S40_r": 0,
        "S50_r": 0,
        "S60_r": 0,
        "S70_r": 0,
        "S80_r": 0,
        "S90_r": 0,
        "S95_r": 0,
        "Navg_r": 0,  # Loudness
        "Nrmc_r": 0,
        "Nmax_r": 0,
        "N05_r": 0,
        "N10_r": 0,
        "N20_r": 0,
        "N30_r": 0,
        "N40_r": 0,
        "N50_r": 0,
        "N60_r": 0,
        "N70_r": 0,
        "N80_r": 0,
        "N90_r": 0,
        "N95_r": 0,
        "Favg_r": 0,  # Fluctuation Strength
        "Fmax_r": 0,
        "F05_r": 0,
        "F10_r": 0,
        "F20_r": 0,
        "F30_r": 0,
        "F40_r": 0,
        "F50_r": 0,
        "F60_r": 0,
        "F70_r": 0,
        "F80_r": 0,
        "F90_r": 0,
        "F95_r": 0,
        "LAavg_r": 0,  # LA
        "LAmin_r": 0,
        "LAmax_r": 0,
        "LA05_r": 0,
        "LA10_r": 0,
        "LA20_r": 0,
        "LA30_r": 0,
        "LA40_r": 0,
        "LA50_r": 0,
        "LA60_r": 0,
        "LA70_r": 0,
        "LA80_r": 0,
        "LA90_r": 0,
        "LA95_r": 0,
        "LCavg_r": 0,  # LC
        "LCmin_r": 0,
        "LCmax_r": 0,
        "LC05_r": 0,
        "LC10_r": 0,
        "LC20_r": 0,
        "LC30_r": 0,
        "LC40_r": 0,
        "LC50_r": 0,
        "LC60_r": 0,
        "LC70_r": 0,
        "LC80_r": 0,
        "LC90_r": 0,
        "LC95_r": 0,
        "Ravg_r": 0,  # Roughness
        "Rmax_r": 0,
        "R05_r": 0,
        "R10_r": 0,
        "R20_r": 0,
        "R30_r": 0,
        "R40_r": 0,
        "R50_r": 0,
        "R60_r": 0,
        "R70_r": 0,
        "R80_r": 0,
        "R90_r": 0,
        "R95_r": 0,
        "Tgavg_r": 0,  # Tonality
        "Tavg_r": 0,
        "Tmax_r": 0,
        "T05_r": 0,
        "T10_r": 0,
        "T20_r": 0,
        "T30_r": 0,
        "T40_r": 0,
        "T50_r": 0,
        "T60_r": 0,
        "T70_r": 0,
        "T80_r": 0,
        "T90_r": 0,
        "T95_r": 0,
        "M00005_0_r": 0,  # Frequency
        "M00006_3_r": 0,
        "M00008_0_r": 0,
        "M00010_0_r": 0,
        "M00012_5_r": 0,
        "M00016_0_r": 0,
        "M00020_0_r": 0,
        "M00025_0_r": 0,
        "M00031_5_r": 0,
        "M00040_0_r": 0,
        "M00050_0_r": 0,
        "M00063_0_r": 0,
        "M00080_0_r": 0,
        "M00100_0_r": 0,
        "M00125_0_r": 0,
        "M00160_0_r": 0,
        "M00200_0_r": 0,
        "M00250_0_r": 0,
        "M00315_0_r": 0,
        "M00400_0_r": 0,
        "M00500_0_r": 0,
        "M00630_0_r": 0,
        "M00800_0_r": 0,
        "M01000_0_r": 0,
        "M01250_0_r": 0,
        "M01600_0_r": 0,
        "M02000_0_r": 0,
        "M02500_0_r": 0,
        "M03150_0_r": 0,
        "M04000_0_r": 0,
        "M05000_0_r": 0,
        "M06300_0_r": 0,
        "M08000_0_r": 0,
        "M10000_0_r": 0,
        "M12500_0_r": 0,
        "M16000_0_r": 0,
        "M20000_0_r": 0,
    }

    # Initialize to zero data(loudnes value, and A-weigth filter) that is
    #  re-used in several feature calculations
    N = None
    A_A = None
    B_A = None

    # Go over list of desired features
    for i, feature in enumerate(feature_list):
        if feature == "loudness":
            print("Calculating loudness")
            N, N_spec, bark_axis, time_axis = loudness_zwtv(
                signal, fs, field_type="free"
            )
            stats_loudness = [
                "avg",
                "rmc",
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
            loudness_data = calculate_stats(N, stats_loudness)
            output["Navg_r"] = loudness_data["avg"]
            output["Nrmc_r"] = loudness_data["rmc"]
            output["Nmax_r"] = loudness_data["max"]
            output["N05_r"] = loudness_data["p05"]
            output["N10_r"] = loudness_data["p10"]
            output["N20_r"] = loudness_data["p20"]
            output["N30_r"] = loudness_data["p30"]
            output["N40_r"] = loudness_data["p40"]
            output["N50_r"] = loudness_data["p50"]
            output["N60_r"] = loudness_data["p60"]
            output["N70_r"] = loudness_data["p70"]
            output["N80_r"] = loudness_data["p80"]
            output["N90_r"] = loudness_data["p90"]
            output["N95_r"] = loudness_data["p95"]

        if feature == "sharpness":
            print("Calculating sharpness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            stats_sharpness = [
                "avg",
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
            S, S_time_axis = sharpness_din(N, N_spec, time_axis, 0.5)
            sharpness_data = calculate_stats(S, stats_sharpness)
            output["Savg_r"] = sharpness_data["avg"]
            output["Smax_r"] = sharpness_data["max"]
            output["S05_r"] = sharpness_data["p05"]
            output["S10_r"] = sharpness_data["p10"]
            output["S20_r"] = sharpness_data["p20"]
            output["S30_r"] = sharpness_data["p30"]
            output["S40_r"] = sharpness_data["p40"]
            output["S50_r"] = sharpness_data["p50"]
            output["S60_r"] = sharpness_data["p60"]
            output["S70_r"] = sharpness_data["p70"]
            output["S80_r"] = sharpness_data["p80"]
            output["S90_r"] = sharpness_data["p90"]
            output["S95_r"] = sharpness_data["p95"]

        if feature == "LA":
            print("Calculating LA")
            stats_LA = [
                "avgdB",
                "max",
                "min",
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
            [B_A, A_A] = A_weighting(fs)
            signal_A = lfilter(B_A, A_A, signal)
            LAeq = pressure2leq(signal_A, fs, 0.125)
            LA_data = calculate_stats(LAeq, stats_LA)
            output["LAavg_r"] = LA_data["avgdB"]
            output["LAmax_r"] = LA_data["max"]
            output["LAmin_r"] = LA_data["min"]
            output["LA05_r"] = LA_data["p05"]
            output["LA10_r"] = LA_data["p10"]
            output["LA20_r"] = LA_data["p20"]
            output["LA30_r"] = LA_data["p30"]
            output["LA40_r"] = LA_data["p40"]
            output["LA50_r"] = LA_data["p50"]
            output["LA60_r"] = LA_data["p60"]
            output["LA70_r"] = LA_data["p70"]
            output["LA80_r"] = LA_data["p80"]
            output["LA90_r"] = LA_data["p90"]
            output["LA95_r"] = LA_data["p95"]

        if feature == "LC":
            print("Calculating LC")
            stats_LC = [
                "avgdB",
                "max",
                "min",
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
            [B_C, A_C] = C_weighting(fs)
            signal_C = lfilter(B_C, A_C, signal)
            LCeq = pressure2leq(signal_C, fs, 0.125)
            LC_data = calculate_stats(LCeq, stats_LC)
            output["LCavg_r"] = LC_data["avgdB"]
            output["LCmax_r"] = LC_data["max"]
            output["LCmin_r"] = LC_data["min"]
            output["LC05_r"] = LC_data["p05"]
            output["LC10_r"] = LC_data["p10"]
            output["LC20_r"] = LC_data["p20"]
            output["LC30_r"] = LC_data["p30"]
            output["LC40_r"] = LC_data["p40"]
            output["LC50_r"] = LC_data["p50"]
            output["LC60_r"] = LC_data["p60"]
            output["LC70_r"] = LC_data["p70"]
            output["LC80_r"] = LC_data["p80"]
            output["LC90_r"] = LC_data["p90"]
            output["LC95_r"] = LC_data["p95"]

        if feature == "frequency":
            print("Calculating frequency features")
            if B_A is None:
                [B_A, A_A] = A_weighting(fs)
                signal_A = lfilter(B_A, A_A, signal)
            M_values = calculate_M(signal_A, fs)
            output["M00005_0_r"] = M_values[0]
            output["M00006_3_r"] = M_values[1]
            output["M00008_0_r"] = M_values[2]
            output["M00010_0_r"] = M_values[3]
            output["M00012_5_r"] = M_values[4]
            output["M00016_0_r"] = M_values[5]
            output["M00020_0_r"] = M_values[6]
            output["M00025_0_r"] = M_values[7]
            output["M00031_5_r"] = M_values[8]
            output["M00040_0_r"] = M_values[9]
            output["M00050_0_r"] = M_values[10]
            output["M00063_0_r"] = M_values[11]
            output["M00080_0_r"] = M_values[12]
            output["M00100_0_r"] = M_values[13]
            output["M00125_0_r"] = M_values[14]
            output["M00160_0_r"] = M_values[15]
            output["M00200_0_r"] = M_values[16]
            output["M00250_0_r"] = M_values[17]
            output["M00315_0_r"] = M_values[18]
            output["M00400_0_r"] = M_values[19]
            output["M00500_0_r"] = M_values[20]
            output["M00630_0_r"] = M_values[21]
            output["M00800_0_r"] = M_values[22]
            output["M01000_0_r"] = M_values[23]
            output["M01250_0_r"] = M_values[24]
            output["M01600_0_r"] = M_values[25]
            output["M02000_0_r"] = M_values[26]
            output["M02500_0_r"] = M_values[27]
            output["M03150_0_r"] = M_values[28]
            output["M04000_0_r"] = M_values[29]
            output["M05000_0_r"] = M_values[30]
            output["M06300_0_r"] = M_values[31]
            output["M08000_0_r"] = M_values[32]
            output["M10000_0_r"] = M_values[33]
            output["M12500_0_r"] = M_values[34]
            output["M16000_0_r"] = M_values[35]
            output["M20000_0_r"] = M_values[36]

        if feature == "roughness":
            print("Calculating roughness")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            R = calculate_roughness(N_spec)
            stats_roughness = [
                "avg",
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
            roughness_data = calculate_stats(R, stats_roughness)
            output["Ravg_r"] = roughness_data["avg"]
            output["Rmax_r"] = roughness_data["max"]
            output["R05_r"] = roughness_data["p05"]
            output["R10_r"] = roughness_data["p10"]
            output["R20_r"] = roughness_data["p20"]
            output["R30_r"] = roughness_data["p30"]
            output["R40_r"] = roughness_data["p40"]
            output["R50_r"] = roughness_data["p50"]
            output["R60_r"] = roughness_data["p60"]
            output["R70_r"] = roughness_data["p70"]
            output["R80_r"] = roughness_data["p80"]
            output["R90_r"] = roughness_data["p90"]
            output["R95_r"] = roughness_data["p95"]

        if feature == "fluctuation":
            print("Calculating fluctuation strength")
            if N is None:
                N, N_spec, bark_axis, time_axis = loudness_zwtv(
                    signal, fs, field_type="free"
                )
            FS = calculate_fluctuation(N_spec)
            stats_fluctuation = [
                "avg",
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
            fluctuation_data = calculate_stats(FS, stats_fluctuation)
            output["Favg_r"] = fluctuation_data["avg"]
            output["Fmax_r"] = fluctuation_data["max"]
            output["F05_r"] = fluctuation_data["p05"]
            output["F10_r"] = fluctuation_data["p10"]
            output["F20_r"] = fluctuation_data["p20"]
            output["F30_r"] = fluctuation_data["p30"]
            output["F40_r"] = fluctuation_data["p40"]
            output["F50_r"] = fluctuation_data["p50"]
            output["F60_r"] = fluctuation_data["p60"]
            output["F70_r"] = fluctuation_data["p70"]
            output["F80_r"] = fluctuation_data["p80"]
            output["F90_r"] = fluctuation_data["p90"]
            output["F95_r"] = fluctuation_data["p95"]

    return output
