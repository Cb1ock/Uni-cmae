# -*- coding: utf-8 -*- # @Author: hao cheng  # @Date: 2024-08-23 15:16:41  # @Last Modified by:   hao cheng  # @Last Modified time: 2024-08-23 15:16:41 # -*- coding: utf-8 -*- # @Author: hao cheng  # @Date: 2024-08-21 02:40:45  # @Last Modified by:   hao cheng  # @Last Modified time: 2024-08-21 02:40:45 # -*- coding: utf-8 -*- # @Author: hao cheng  # @Date: 2024-08-19 13:37:23  # @Last Modified by:   hao cheng  # @Last Modified time: 2024-08-19 13:37:23 # -*- coding: utf-8 -*- # @Author: hao cheng  # @Date: 2024-08-19 13:37:21  # @Last Modified by:   hao cheng  # @Last Modified time: 2024-08-19 13:37:21 # -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path
import cv2
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import PIL

import decoder.decoder as decoder
import decoder.video_container as container
from decoder.decoder import get_start_end_idx, temporal_sampling
from decoder.transform import create_random_augment
from decoder.random_erasing import RandomErasing
import decoder.utils as utils
from torchvision import transforms

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, video_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.video_conf = video_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        self.dataset_type = self.video_conf.get('dataset_type', 'video')
        print(f'using {self.dataset_type} data as video input')
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])
        
        #######################################################################################################
        self.aa_type = self.video_conf.get('aa_type', "rand-m7-n4-mstd0.5-inc1")
        self.pretrain_rand_flip = self.video_conf.get('pretrain_rand_flip', True)
        self.pretrain_rand_erase_prob = self.video_conf.get('pretrain_rand_erase_prob', 0.25)
        self.pretrain_rand_erase_mode = self.video_conf.get('pretrain_rand_erase_mode', "pixel")
        self.pretrain_rand_erase_count = self.video_conf.get('pretrain_rand_erase_count', 1)
        self.pretrain_rand_erase_split = self.video_conf.get('pretrain_rand_erase_split', False)

        self.jitter_aspect_relative = self.video_conf.get('jitter_aspect_relative', [0.75, 1.3333])
        self.jitter_scales_relative = self.video_conf.get('jitter_scales_relative', [0.5, 1.0])

        print(
            f"jitter_aspect_relative {self.jitter_aspect_relative} jitter_scales_relative {self.jitter_scales_relative}"
        )

        self._repeat_aug = self.video_conf.get('repeat_aug', 1)
        self._video_meta = {}
        self._num_retries = self.video_conf.get('num_retries', 10)

        self._train_jitter_scales = self.video_conf.get('train_jitter_scales', (256, 320))
        self._train_crop_size = self.video_conf.get('train_crop_size', 224)
        self._train_random_horizontal_flip = self.video_conf.get('train_random_horizontal_flip', True)

        self._test_num_ensemble_views = self.video_conf.get('test_num_ensemble_views', 10)
        self._test_num_spatial_crops = self.video_conf.get('test_num_spatial_crops', 3)
        self._test_crop_size = self.video_conf.get('test_crop_size', 256)

        self._sampling_rate = self.video_conf.get('sampling_rate', 4)
        self._num_frames = self.video_conf.get('num_frames', 16)
        self._target_fps = self.video_conf.get('target_fps', 30)

        self._mean = self.video_conf.get('mean', (0.45, 0.45, 0.45))
        self._std = self.video_conf.get('std', (0.225, 0.225, 0.225))

        self.rand_aug = True
        self._inverse_uniform_sampling = self.video_conf.get('inverse_uniform_sampling', False)
        self._use_offset_sampling = self.video_conf.get('use_offset_sampling', True)

        # print(self)
        # print(locals())

        if self.mode in ["pretrain", "finetune", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = self._test_num_ensemble_views * self._test_num_spatial_crops

        print("Constructing DATASET {}...".format(self.mode))
        if self.mode in ["pretrain", "val", "test"]:
            self.rand_aug = False
            print("Perform standard augmentation")
        else:
            self.rand_aug = self.rand_aug
            print("Perform rand augmentation")
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['id'], data_json[i]['wav'], data_json[i]['video_path'], data_json[i]['labels']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['id'] = np_data[0]
        datum['wav'] = np_data[1]
        datum['video_path'] = np_data[2]
        datum['labels'] = np_data[3]
        return datum

    def get_video(self, filename):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["pretrain", "finetune", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale, max_scale = self._train_jitter_scales
            crop_size = self._train_crop_size
        elif self.mode in ["test"]:
            self._spatial_temporal_idx = list(range(self._num_clips))
            index = random.randint(0, len(self._spatial_temporal_idx) - 1)
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self._test_num_spatial_crops
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self._test_num_spatial_crops)
                if self._test_num_spatial_crops > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self._test_crop_size] * 3
                if self._test_num_spatial_crops > 1
                else [self._train_jitter_scales[0]] * 2 + [self._test_crop_size]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        sampling_rate = self._sampling_rate
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):

            if self.dataset_type == 'video':
                # 读取视频为2进制文件，如果不是test模式，并且读取失败的话，随机选择另一个视频 # TODO：随机选择另一个视频
                with open(filename, "rb") as fp:
                    video_container = fp.read()

                video_meta = {}
                # Decode video. Meta info is used to perform selective decoding.
                frames, fps, decode_all_video = decoder.decode(
                    video_container,
                    sampling_rate,
                    self._num_frames,
                    temporal_sample_index,
                    self._test_num_ensemble_views,
                    video_meta=video_meta,
                    target_fps=self._target_fps,
                    max_spatial_scale=min_scale,
                    use_offset=self._use_offset_sampling,
                    rigid_decode_all_video=self.mode in ["pretrain"],
                )
                if self.dataset=='voxceleb2':
                    frames = utils.random_crop_with_roi(frames,
                                                        crop_bottom=64,
                                                        crop_left=32,
                                                        crop_right=32
                                                                )
            elif self.dataset_type == 'frame':

                def resize_frames(frames, size):
                    """
                    Resize the middle two dimensions of the frames to the specified size.
                    Args:
                        frames (tensor): Video frames of shape T x H x W x C.
                        size (tuple): The desired size (height, width).
                    Returns:
                        frames_resized (tensor): Resized frames.
                    """
                    T, H, W, C = frames.shape
                    frames_resized = torch.zeros((T, size[0], size[1], C), dtype=frames.dtype)
                    for t in range(T):
                        frame = frames[t]
                        # Change from HWC to CHW using einsum
                        frame_chw = torch.einsum('hwc->chw', frame)
                        resized_frame = F.resize(frame_chw, size)
                        # Change from CHW back to HWC using einsum
                        resized_frame_hwc = torch.einsum('chw->hwc', resized_frame)
                        frames_resized[t] = resized_frame_hwc
                    return frames_resized
                
                frames = decoder.load_frames_from_folder(filename)
                fps = 24
                decode_all_video = True
                frames = resize_frames(frames, (self.im_res, self.im_res))

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            # if frames is None:
            #     print(
            #         "Failed to decode video idx {} from {}; trial {}".format(
            #             index, self._path_to_videos[index], i_try
            #         )
            #     )
            #     if self.mode not in ["test"] and i_try > self._num_retries // 2:
            #         # let's try another one
            #         index = random.randint(0, len(self._path_to_videos) - 1)
            #     continue
            
            # TODO：if decoding failed (wrong format, video is too short, and etc), select another video

            frames_list = []

            if self.rand_aug: # set it True while Fine-tuning, False while Pre-training
                for i in range(self._repeat_aug):
                    clip_sz = sampling_rate * self._num_frames / self._target_fps * fps # 总共采了多少帧 4*16/30*24 按照_target_fps去采样的
                    start_idx, end_idx = get_start_end_idx(
                        frames.shape[0],
                        clip_sz,
                        temporal_sample_index if decode_all_video else 0,
                        self._test_num_ensemble_views if decode_all_video else 1,
                        use_offset=self._use_offset_sampling,
                    )
                    # Perform temporal sampling from the decoded video.
                    new_frames = temporal_sampling(
                        frames, start_idx, end_idx, self._num_frames
                    )
                    new_frames = self._aug_frame(
                        new_frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )
                    frames_list.append(new_frames)

            else:
                # T H W C -> C T H W.
                for i in range(self._repeat_aug):
                    clip_sz = sampling_rate * self._num_frames / self._target_fps * fps
                    start_idx, end_idx = get_start_end_idx(
                        frames.shape[0],
                        clip_sz,
                        temporal_sample_index if decode_all_video else 0,
                        self._test_num_ensemble_views if decode_all_video else 1,
                        use_offset=self._use_offset_sampling,
                    )
                    # Perform temporal sampling from the decoded video.
                    new_frames = temporal_sampling(
                        frames, start_idx, end_idx, self._num_frames
                    )

                    new_frames = utils.tensor_normalize(
                        new_frames, self._mean, self._std
                    )
                    new_frames = new_frames.permute(3, 0, 1, 2)

                    scl, asp = (
                        self.jitter_scales_relative,
                        self.jitter_aspect_relative,
                    )
                    relative_scales = (
                        None
                        if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
                        else scl
                    )
                    relative_aspect = (
                        None
                        if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
                        else asp
                    )

                    # Perform data augmentation.
                    new_frames = utils.spatial_sampling(
                        new_frames,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self._train_random_horizontal_flip,
                        inverse_uniform_sampling=self._inverse_uniform_sampling,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                    )
                    frames_list.append(new_frames)

            frames = torch.stack(frames_list, dim=0)
            frames = frames.squeeze(0)

            return frames
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )
        
    def video_mixup(self, filename1, filename2, mix_lambda = 1):
        if self.mode in ["pretrain", "finetune"]:
            frames1 = self.get_video(filename1)
            frames2 = self.get_video(filename2)
            frames = mix_lambda * frames1 + (1 - mix_lambda) * frames2
            frames = frames - frames.mean()
        
            frames = frames - frames.mean()
        return frames

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.aa_type,
            interpolation="bicubic",
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames,
            (0.45, 0.45, 0.45),
            (0.225, 0.225, 0.225),
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.jitter_scales_relative,
            self.jitter_aspect_relative,
        )
        relative_scales = (
            None
            if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
            else scl
        )
        relative_aspect = (
            None
            if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
            else asp
        )
        # frames = utils.spatial_sampling(
        #     frames,
        #     spatial_idx=spatial_sample_index,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        #     crop_size=crop_size,
        #     random_horizontal_flip=self.pretrain_rand_flip,
        #     inverse_uniform_sampling=False,
        #     aspect_ratio=relative_aspect,
        #     scale=relative_scales,
        #     motion_shift=False,
        # )

        if self.pretrain_rand_erase_prob > 0.0:
            erase_transform = RandomErasing(
                self.pretrain_rand_erase_prob,
                mode=self.pretrain_rand_erase_mode,
                max_count=self.pretrain_rand_erase_count,
                num_splits=self.pretrain_rand_erase_count,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames
    
    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)
    
    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index): # video shape:(B, C, T, H, W), audio shape:(B, T, F)
        datum = self.data[index]
        datum = self.decode_data(datum)
        if self.dataset == 'MAFW':
            datum['video_path'] = datum['video_path'] + "/" + datum['id']
        if random.random() < self.mixup: # TODO：mixup fix
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)

            if self.dataset == 'MAFW':
                mix_datum['video_path'] = mix_datum['video_path'] + "/" + mix_datum['id']
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            try:
                image = self.video_mixup(datum['video_path'], mix_datum['video_path'], mix_lambda)
            except:
                image = torch.zeros([3, 16, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading video')
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

        else:
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            try:
                fbank = self._wav2fbank(datum['wav'], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            #try:
            image = self.get_video(datum['video_path'])
            # except:
            #     image = torch.zeros([3, 16, self.im_res, self.im_res]) + 0.01
            #     print('there is an error in loading image')
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # image shape is [3, 16, im_res, im_res]

        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples