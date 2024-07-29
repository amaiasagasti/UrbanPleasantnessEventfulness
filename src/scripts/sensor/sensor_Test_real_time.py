from pydub import AudioSegment
import numpy as np
import joblib
import time
from SoundLights.dataset.auxiliary_functions import load
from CLAP.src.laion_clap import CLAP_Module
from SoundLights.dataset.dataset_functions import extract_CLAP_embeddings

# Load the audio file
gain = 1.412537545
norm_gain = 6.44
audio_r, fs = load(
    "data/listening_test_audios/freesound_564637.wav", wav_calib=gain / norm_gain, ch=1
)
print("fs ", fs)

# Parameters for simulation
chunk_duration = fs * 6
# Load the trained model - the one we are testing
model_evaluate = joblib.load("data/models/RFR_CLAP_P2.joblib")
# Load the CLAP model to generate features
print("------- code starts -----------")
model_CLAP = CLAP_Module(enable_fusion=True)
print("------- clap module -----------")
model_CLAP.load_ckpt("data/models/630k-fusion-best.pt")
print("------- model loaded -----------")
sum_predictions = 0
# Simulate real-time processing
for start in range(0, len(audio_r), chunk_duration):
    end = start + chunk_duration
    audio_chunk = audio_r[start:end]
    start_time = time.time()
    # Extract features
    features = model_CLAP.get_audio_embedding_from_data([audio_chunk], use_tensor=False)
    # Predict using the trained model
    prediction = model_evaluate.predict(features)
    finish_time = time.time()
    sum_predictions = sum_predictions + prediction
    print("PLEASANTNESS ", prediction, "TIME ", finish_time - start_time)

print("Mean prediction ", sum_predictions / (len(audio_r) / chunk_duration))
