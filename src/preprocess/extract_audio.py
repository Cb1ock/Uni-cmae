import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
args = parser.parse_args()

input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

# first resample audio
for i in range(input_filelist.shape[0]):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s}'.format(input_f, output_f_1)) # save an intermediate file

# then extract the first channel
for i in range(input_filelist.shape[0]):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
    output_f_2 = args.target_fold + '/' + video_id + '.wav'
    os.system('sox {:s} {:s} remix 1'.format(output_f_1, output_f_2))
    # remove the intermediate file
    os.remove(output_f_1)
'''
检测音频文件的采样率和通道数，采样率应为16000Hz，通道数应为1
'''
import os
import numpy as np
import argparse
import json
import subprocess

data_json_file_1 = '/home/hao/Project/cav-mae/egs/voxceleb2/test_data.json'
data_json_file_2 = '/home/hao/Project/cav-mae/egs/voxceleb2/train_data.json'

def process_json_data(data_json_file):
    with open(data_json_file, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data['data']:
        processed_data.append([item['id'], item['wav'], item['video_path'], item['labels']])
    
    data_np = np.array(processed_data, dtype=str)
    
    decoded_data = []
    for np_item in data_np:
        datum = {
            'id': np_item[0],
            'wav': np_item[1],
            'video_path': np_item[2],
            'labels': np_item[3]
        }
        decoded_data.append(datum)
    
    return decoded_data

def get_sample_rate(file_path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return int(result.stdout)

def get_channel_count(file_path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels', '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return int(result.stdout)

def process_audio_files(audio_files, target_fold):
    if not os.path.exists(target_fold):
        os.makedirs(target_fold)
    
    for audio_file in audio_files:
        ext_len = len(audio_file.split('/')[-1].split('.')[-1])
        audio_id = audio_file.split('/')[-1][:-ext_len-1]
        intermediate_f = os.path.join(target_fold, audio_id + '_intermediate.wav')
        output_f = os.path.join(target_fold, audio_id + '.wav')


        # Check sample rate
        sample_rate = get_sample_rate(intermediate_f)
        if sample_rate != 16000:
            print(f'Processing {audio_file} with sample rate {sample_rate} Hz')
            #os.system('ffmpeg -i {:s} -vn -ar 16000 {:s}'.format(intermediate_f, output_f))
        else:
            print(f'Processing {audio_file} with sample rate {sample_rate} Hz')
            #os.system('ffmpeg -i {:s} -vn {:s}'.format(intermediate_f, output_f))

        # Check channel count
        channel_count = get_channel_count(output_f)
        if channel_count != 1:
            print(f'Extracting first channel from {output_f} with {channel_count} channels')
        elif channel_count == 1:
            print(f'Processing {output_f} with {channel_count} channel')
            #os.system('sox {:s} {:s} remix 1'.format(output_f, output_f))

        # # Remove the intermediate file if it was created
        # if audio_file.endswith('.m4a'):
        #     os.remove(intermediate_f)

data_1 = process_json_data(data_json_file_1)
data_2 = process_json_data(data_json_file_2)

wavs_1 = [item['wav'] for item in data_1]
wavs_2 = [item['wav'] for item in data_2]

target_fold = './sample_audio/'

process_audio_files(wavs_1, target_fold)
process_audio_files(wavs_2, target_fold)