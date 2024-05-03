import os
import time
import json
from CLAP.src.laion_clap import CLAP_Module

import sys

print("Active Python environment:", sys.prefix)

# Get the list of audio files
audio_paths = []
audioFolderPath = "data/ARAUS-extended_soundscapes/"
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


# Determine the output directory
output_dir = "data/embedding_features/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("------- output directory checked -----------")


# Process each audio clip
start_time = time.time()
for i, audio_path in enumerate(audio_paths):
    # Create the output path
    fname = os.path.splitext(os.path.basename(audio_path))[0]
    print(fname)
    output_path = os.path.join(output_dir, f"{fname}.json")
    print("Output path ", output_path)
    # Extract the embeddings
    print("------- extracting embeddings -----------")
    embeddings = extract_embeddings(model, audio_path)
    print("------- embeddings extracted -----------")
    # Save results
    with open(output_path, "w") as outfile:
        json.dump(
            {"audio_path": audio_path, "embeddings": embeddings}, outfile, indent=4
        )
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
