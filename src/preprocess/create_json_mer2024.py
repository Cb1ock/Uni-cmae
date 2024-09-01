import os
import csv
import json
import random

audio_path = '/data/public_datasets/MER_2024/audio-labeled_16k'
video_path = '/data/public_datasets/MER_2024/video-labeled'
label_path = '/data/public_datasets/MER_2024/label-disdim.csv'
save_path = '/home/hao/Project/uni-cmae/egs/MER2024'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取标签文件
labels = []
with open(label_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row)

# 创建标签映射字典
label_mapping = {}
label_index = 0
for label in labels:
    discrete_label = label['discrete']
    if discrete_label not in label_mapping:
        label_mapping[discrete_label] = label_index
        label_index += 1

# 构建 JSON 数据结构
data = []
for label in labels:
    sample_id = label['name']
    discrete_label = label['discrete']
    label_index = label_mapping[discrete_label]
    
    data.append({
        "id": sample_id,
        "wav": os.path.join(audio_path, f"{sample_id}.wav"),
        "video_path": os.path.join(video_path, f"{sample_id}.mp4"),
        "labels": label_index
    })

# 打乱数据
random.shuffle(data)

# 按照 80% 训练数据和 20% 测试数据进行划分
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# 保存训练数据 JSON 文件
train_json_data = {"data": train_data}
train_json_file_path = os.path.join(save_path, 'train_data.json')
with open(train_json_file_path, 'w') as train_json_file:
    json.dump(train_json_data, train_json_file, indent=4)

# 保存测试数据 JSON 文件
test_json_data = {"data": test_data}
test_json_file_path = os.path.join(save_path, 'test_data.json')
with open(test_json_file_path, 'w') as test_json_file:
    json.dump(test_json_data, test_json_file, indent=4)

# 保存标签映射文件
label_mapping_file_path = os.path.join(save_path, 'class_labels_indices.csv')
with open(label_mapping_file_path, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['index', 'mid', 'display_name'])
    for label, index in label_mapping.items():
        writer.writerow([index, index, label])