# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Roughness Zwicker y Fastl "Psychoacoustics: Facts and Models"

import numpy as np


# Cálculo Roughness
def acousticRoughness(specificLoudness, fmod):
    """matrix = specificLoudness.reshape(24, 10)

    # Sum along the second axis (axis=1)
    sums = np.sum(matrix, axis=1)
    specificLoudness = sums"""
    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
    R = 0.3 * (fmod / 1000) * sum(0.1 * specificLoudnessdiff)
    return R
