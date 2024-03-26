import sys
import numpy as np
from scipy.signal import lfilter
from scipy.signal.filter_design import bilinear
from numpy import pi, convolve
from maad.spl import pressure2leq
from maad.util import mean_dB
from mosqito.sq_metrics import loudness_zwtv
from Mosqito.Sharpness import sharpness_din


sys.path.append("..")


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

    # Loudness values have to be computed for any other feature
    N, N_spec, bark_axis, time_axis = loudness_zwtv(signal, fs, field_type="free")

    # Go over list of desired features
    for i, feature in enumerate(feature_list):

        if feature == "loudness":
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

    return output
