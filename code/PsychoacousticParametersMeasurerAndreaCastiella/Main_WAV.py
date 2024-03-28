# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Main para archivos *.wav

import math
import numpy as np
import soundfile as sf
import os
from .loudness_ISO532 import loudness_ISO532, loudness_ISO532_time, sone2phon
import sys
from .ThirdOctaveFilters import (
    ThirdOctaveBandFilter,
    ThirdOctaveSPL,
    ThirdOctaveLevelTime,
)
from .Sharpness import calc_sharpness
from .FiltroPonderacionA import filtroPonderacionA
from .Roughness import acousticRoughness
from .FluctuationStrength import acousticFluctuation, fmoddetection

# Constantes e inicialización pyaudio
""" CHUNK = 4800  # Tamaño en muestras almacenadas en cada array
RATE = 48000  # Muestras por segundo """
TimeVarying = True


# Funciones principales
def mainEstacionario(data, rate, chunk):
    # Filtro tercio de octava
    ThirdOctave = ThirdOctaveBandFilter(data, chunk)
    ThirdOctaveSPL_value = ThirdOctaveSPL(
        ThirdOctaveBands=ThirdOctave, CHUNK=chunk, RATE=rate, TimeSkip=0
    )
    # Loudness
    loudness, specLoudness, _, _ = loudness_ISO532(
        ThirdOctaveSPL_value, SoundFieldDiffuse=0
    )
    loudnessPhon = sone2phon(loudness)
    print("Loudness total en sonios: ", round(loudness, 1), " sonios")
    print("Loudness total en fonos", round(loudnessPhon, 2), " fonos")

    # Cálculo sharpness
    sharpnessZwicker, sharpnessVonBismarck, sharpnessAures = calc_sharpness(
        loudness, specLoudness
    )
    print("Sharpness Zwicker: ", sharpnessZwicker, " acum")
    print("Sharpness VB: ", sharpnessVonBismarck, " acum")
    print("Sharpness Aures: ", sharpnessAures, " acum")

    specLoudness2 = np.stack((specLoudness, specLoudness), axis=1)
    fmodFS = fmoddetection(specLoudness2, fmin=0.2, fmax=64)
    # Fluctuation strength
    FS = acousticFluctuation(specLoudness, fmodFS)
    fmodR = fmoddetection(specLoudness2, fmin=40, fmax=150)
    # Roughness
    R = acousticRoughness(specLoudness, fmodR)

    print("Fluctuation strength: ", round(FS, 2), " vacil")
    print("Roughness: ", round(R, 2), " asper")


def mainVarianteTiempo(data, rate, chunk):
    # Filtrado tercio de octava
    ThirdOctaveLevelTime_value, _, _ = ThirdOctaveLevelTime(data, rate, chunk)
    # Loudness
    loudness, specLoudness = loudness_ISO532_time(
        ThirdOctaveLevelTime_value, SoundFieldDiffuse=0, RATE=rate, CHUNK=chunk
    )

    loudnessPhon = []
    for loundness_i in loudness:
        loudnessPhon.append(sone2phon(loundness_i))

    loudnessPhon = [round(num, 1) for num in loudnessPhon]

    print("Loudness total en sonios: ", loudness, " sonios")
    print("Loudness total en fonos", loudnessPhon, " fonos")

    sharpnessZwicker = []
    sharpnessVonBismarck = []
    sharpnessAures = []

    for i in range(len(loudness)):
        sharpnessZwickerTemp, sharpnessVonBismarckTemp, sharpnessAuresTemp = (
            calc_sharpness(loudness[i], specLoudness[:, i])
        )
        sharpnessZwicker.append(sharpnessZwickerTemp)
        sharpnessVonBismarck.append(sharpnessVonBismarckTemp)
        sharpnessAures.append(sharpnessAuresTemp)

    print("Sharpness Zwicker: ", sharpnessZwicker)
    print("Sharpness VB: ", sharpnessVonBismarck)
    print("Sharpness Aures: ", sharpnessAures)

    # Detección frecuencia moduladora
    fmodFS = fmoddetection(specLoudness, fmin=0.2, fmax=64)
    fmodR = fmoddetection(specLoudness, fmin=40, fmax=150)

    FS = []
    R = []
    PA = []

    for i in range(len(loudness)):
        # Fluctuation strength
        FS.append(acousticFluctuation(specLoudness[:, i], fmodFS))
        # Roughness
        R.append(acousticRoughness(specLoudness[:, i], fmodR))

    FS = [round(num, 4) for num in FS]
    R = [round(num, 4) for num in R]

    print("Fluctuation strength: ", FS, " vacil")
    print("Roughness: ", R, " asper")
    return loudness, specLoudness, sharpnessZwicker, FS, R


def cargarWAV(wav, chunk, corr_factor=1):
    data, _ = sf.read(wav)
    data = data * corr_factor
    print(len(data))
    length = len(data) / chunk
    return data, int(length)


def calcSPL_SPLA(data, RATE):
    # Cálculo del valor rms de la señal captada (CHUNK)
    rms_corr = np.sqrt(np.mean(np.absolute(data.astype(float)) ** 2))
    # Cálculo del valor SPL
    SPL = 20 * math.log(rms_corr / (20 * (10**-6)), 10)
    print("Z-weighted", round(SPL, 1))

    # Fitlrado ponderación A
    pond_A = filtroPonderacionA(data, RATE)
    # Cálculo del valor rms de la señal ponderada A y corrección
    rms_A = np.sqrt(np.mean(np.absolute(pond_A.astype(float)) ** 2))
    rms_A_corr = rms_A
    # Cálculo del valor SPL ponderado A
    SPL_A = 20 * math.log(rms_A_corr / (20 * (10**-6)), 10)
    print("A-weighted", round(SPL_A, 1))


""" 
try:
    # Cargar archivo *.wav
    dataTotal, length = cargarWAV(wav=r'./FicherosAudio/RuidoRosa.wav', 
                                  CHUNK=CHUNK)

    # División del archivo en CHUNK
    for i in range(length):
        data = (dataTotal[CHUNK*i:CHUNK*(i+1)])
        print('data', data)
        # Cálculo SPL y SPL(A)
        calcSPL_SPLA(data)

        # Estacionario
        if not TimeVarying:
            mainEstacionario(data, RATE, CHUNK)

        # Variante en el tiempo
        if TimeVarying:
            mainVarianteTiempo(data, RATE, CHUNK)

# Salir del bucle con ctr+c
except KeyboardInterrupt:
    print('Interrupción de teclado. Finalizando programa.')
    sys.exit()
 """
