import os
from tqdm import tqdm
import json

def create_json_voxceleb2(video_dataset_path, audio_dataset_path, output_json_path):
    # find all the video files
    video_files = []
    for root, dirs, files in os.walk(video_dataset_path):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    
    # find all the audio files
    audio_files = []
    for root, dirs, files in os.walk(audio_dataset_path):
        for file in files:
            if file.endswith(".m4a"):
                audio_files.append(os.path.join(root, file))

    # check if the number of video files and audio files are the same
    assert len(video_files) == len(audio_files), "Number of video files and audio files are not the same"

    # save names of video and audio files in id_sense_index format
    video_files_dict = {}
    for file in video_files:
        parts = file.split(os.sep)
        id_sense_index = f"{parts[-3]}_{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
        video_files_dict[id_sense_index] = file

    audio_files_dict = {}
    for file in audio_files:
        parts = file.split(os.sep)
        id_sense_index = f"{parts[-3]}_{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
        audio_files_dict[id_sense_index] = file

    # check if the names of video and audio files are the same
    video_keys = set(video_files_dict.keys())
    audio_keys = set(audio_files_dict.keys())
    assert video_keys == audio_keys, "Video and audio files do not match"

    # save the video and audio files in a json file
    data = []
    for key in tqdm(video_keys, desc="Saving to JSON"):
        entry = {
            "id": key,
            "wav": audio_files_dict[key],
            "video_path": video_files_dict[key],
            "labels": 0
        }
        data.append(entry)
    
    output_data = {"data": data}
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)

create_json_voxceleb2("/data/public_datasets/VoxCeleb2/train/mp4/", "/data/public_datasets/VoxCeleb2/train/aac/", "train_data.json")
create_json_voxceleb2('/data/public_datasets/VoxCeleb2/test/mp4/', '/data/public_datasets/VoxCeleb2/test/aac/', 'test_data.json')