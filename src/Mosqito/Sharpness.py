import sys
import numpy as np

sys.path.append("..")


from mosqito.sq_metrics import sharpness_din_from_loudness


def sharpness_din(N, N_spec, time_axis, skip):

    # Get sharpness from loudness using mosqito's function
    S = sharpness_din_from_loudness(N, N_spec, "din", 0)

    # Cut transient effect
    cut_index = np.argmin(np.abs(time_axis - skip))

    return S[cut_index:], time_axis[cut_index:]
