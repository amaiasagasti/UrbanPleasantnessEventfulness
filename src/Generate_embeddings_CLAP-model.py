import os
import time
import json
import pandas as pd
from CLAP.src.laion_clap import CLAP_Module

import sys

print("Active Python environment:", sys.prefix)

## Fill in
audioFolderPath = "data/ARAUS-extended_soundscapes/"
output_dir = "data/embedding_features/"
csv_name = "data/csv_files/SoundLights_Freesound.csv"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Import csv
ARAUScsv = pd.read_csv(csv_name)

# Determine the output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("------- output directory checked -----------")

# Get the list of audio files
audio_paths = []
for dirpath, dirnames, files in os.walk(audioFolderPath):
    print(dirpath, dirnames)
    dirpath_split = dirpath.split("soundscapes_augmented")

    dirnames.sort()
    files.sort()
    for file in files:
        if file.endswith(".wav"):
            audio_path = dirpath + "/" + file
            audio_paths.append(audio_path)
assert len(audio_paths) > 0, f"No audio files found"
print(f"Found {len(audio_paths)} audio files")
print("------- audio files listed -----------")

# Load the model
print("------- code starts -----------")
model = CLAP_Module(enable_fusion=True)
print("------- clap module -----------")
model.load_ckpt("data/models/630k-fusion-best.pt")  # "data/models/630k-fusion-best.pt"
print("------- model loaded -----------")


# Define embedding extractor function
def extract_embeddings(model, audio_path):
    # Process
    embeddings = model.get_audio_embedding_from_filelist(
        x=[audio_path], use_tensor=False
    ).tolist()
    return embeddings


print("------- function created -----------")


# Process each audio clip
start_time = time.time()
for i, audio_path in enumerate(audio_paths):
    # Create the output path
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    print(fname)
    output_path = os.path.join(output_dir, f"{fname}.json")
    print("Output path ", output_path)
    # Find audio info in JSON
    file_split = fname.split("_")
    file_fold = int(file_split[1])
    file_participant = "ARAUS_" + file_split[3]
    file_stimulus = int(file_split[5].split(".")[0])
    audio_info_aug = ARAUScsv[ARAUScsv["info.fold"] == file_fold]
    audio_info_aug = audio_info_aug[
        audio_info_aug["info.stimulus_index"] == file_stimulus
    ]
    audio_info_aug = audio_info_aug[
        audio_info_aug["info.participant"] == file_participant
    ]
    gain = audio_info_aug["info.wav_gain"].values[0]

    # Add basic info about audio to dictionary
    audio_info = {
        "file": fname + ".wav",
        "participant": file_participant,
        "fold": int(audio_info_aug["info.fold"].values[0]),
        "soundscape": audio_info_aug["info.soundscape"].values[0],
        "masker": audio_info_aug["info.masker"].values[0],
        "smr": int(audio_info_aug["info.smr"].values[0]),
        "stimulus_index": int(audio_info_aug["info.stimulus_index"].values[0]),
        "wav_gain": float(audio_info_aug["info.wav_gain"].values[0]),
        "time_taken": audio_info_aug["info.time_taken"].values[0],
        "is_attention": int(audio_info_aug["info.is_attention"].values[0]),
        "pleasant": int(audio_info_aug["info.pleasant"].values[0]),
        "eventful": int(audio_info_aug["info.eventful"].values[0]),
        "chaotic": int(audio_info_aug["info.chaotic"].values[0]),
        "vibrant": int(audio_info_aug["info.vibrant"].values[0]),
        "uneventful": int(audio_info_aug["info.uneventful"].values[0]),
        "calm": int(audio_info_aug["info.calm"].values[0]),
        "annoying": int(audio_info_aug["info.annoying"].values[0]),
        "monotonous": int(audio_info_aug["info.monotonous"].values[0]),
        "appropriate": int(audio_info_aug["info.appropriate"].values[0]),
        "P_ground_truth": audio_info_aug["info.P_ground_truth"].values[0],
        "E_ground_truth": audio_info_aug["info.E_ground_truth"].values[0],
        "Leq_R_r": audio_info_aug["info.Leq_R_r"].values[0],
        "masker_bird": int(audio_info_aug["info.masker_bird"].values[0]),
        "masker_construction": int(
            audio_info_aug["info.masker_construction"].values[0]
        ),
        "masker_silence": int(audio_info_aug["info.masker_silence"].values[0]),
        "masker_traffic": int(audio_info_aug["info.masker_traffic"].values[0]),
        "masker_water": int(audio_info_aug["info.masker_water"].values[0]),
        "masker_wind": int(audio_info_aug["info.masker_wind"].values[0]),
    }
    # Extract the embeddings
    print("------- extracting embeddings -----------")
    embeddings = extract_embeddings(model, audio_path)
    print("------- embeddings extracted -----------")
    # Save results
    with open(output_path, "w") as outfile:
        json.dump({"info": audio_info, "embeddings": embeddings}, outfile, indent=4)
        """         except Exception as e:
            print(f"Error processing {audio_path}: {repr(e)}")
        except KeyboardInterrupt:
            print(f"Interrupted by user.")
            break """
    # Print progress
    if (i + 1) % 1000 == 0 or i == 0 or i + 1 == len(audio_paths):
        print(f"[{i+1:>{len(str(len(audio_paths)))}}/{len(audio_paths)}]")
total_time = time.time() - start_time
print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
print(f"Average time/file: {total_time/len(audio_paths):.2f} sec.")

#############
print("Done!\n")
