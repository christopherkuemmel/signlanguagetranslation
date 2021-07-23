"""Sanity check the resnext 101 3d kinetics."""

import torch
from torchvision.transforms import Compose
import data.transforms as tr
from data.phoenix2014Tdataset import Phoenix2014TDataset, pad_collate
from models.resnext_101_kinetics import load_resnext101_3d_kinetics
from torch.utils.data import DataLoader
import pandas as pd
from typing import List
import numpy as np
import glob
import cv2
import torch.nn.functional as F
import json

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(42)


def load_kinetics_labels(csv_path: str) -> List[str]:
    """Create kinetics class label list.

    Args:
        csv_path (str): The path to the kinetics csv file.

    Returns:
        List[str]: list of unique kinetics class names.
    """
    # class_labels_map = {}
    # index = 0
    # with open('notebook/models/_data/kinetics400/kinetics.json', 'r') as f:
    #     data = json.load(f)
    # for class_label in data['labels']:
    #     class_labels_map[class_label] = index
    #     index += 1
    # x = class_labels_map
    kinetics_df = pd.read_csv(csv_path)
    return sorted(kinetics_df['label'].unique().tolist())


def load_video(path: str, image_size: int) -> np.ndarray:
    """Loads and returns a video.

    Args:
        path (str): Path to the video.
        image_size (int): The dimensions of the video (height x width).

    Returns:
        np.ndarray: The video as array.
    """
    video = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video.append(cv2.resize(frame, (image_size, image_size)))
    cap.release()
    video = np.array(video)
    video = np.flip(video, -1).copy()  # Flip color channels, bgr2rgb
    return video


def phoenix_loader():
    """Create a PHOENIX-2014-T DataLoader."""
    transforms = Compose([
        tr.Resize(200),
        tr.WindowDepth(16),
    ])

    loader_params = {
        'batch_size': 1,
        'collate_fn': pad_collate,
        'shuffle': False,
        'num_workers': 0,
    }

    # define dataset
    train_set = Phoenix2014TDataset('data/PHOENIX-2014-T-release-v3', 'train', transforms)
    return DataLoader(train_set, **loader_params)


def kinetics_loader(image_size: int):
    """Create a kinetics 400 iterable.

    Args:
        image_size (int): The dimensions of the video (heigth x width).
    """
    norm_value = 1
    videos = []
    # mean_tensor = torch.tensor([0, 0, 0])
    std_tensor = torch.tensor([1, 1, 1])

    # mean_tensor = torch.tensor([114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value])  # activity net
    mean_tensor = torch.tensor([110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value])  # kinetics
    # std_tensor = torch.tensor([38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value])

    sorted_list = sorted(glob.glob('notebook/models/_data/kinetics400/**.mp4'))
    for video_path in sorted_list:
        video = load_video(video_path, image_size)
        if not video.shape[0] % 16 == 0:
            pad_size = 16 - video.shape[0] % 16
            pad_array = np.zeros((pad_size, image_size, image_size, 3))
            video = np.concatenate((video, pad_array))
        video = torch.tensor(video).float()
        video = video.permute(0, 3, 1, 2)

        video = video / norm_value
        for frame in video:
            for t, m, s in zip(frame, mean_tensor, std_tensor):
                t.sub_(m).div_(s)

        video = video.view(1, -1, 16, 3, image_size, image_size)
        video = video.permute((0, 1, 3, 2, 4, 5))
        videos.append({'video': video, 'name': video_path})

    return videos


def check_resnext():
    """Check ResNext 101 3d kinetics."""

    # define loader/ iterator
    # train_loader = phoenix_loader()
    image_size = 112
    train_loader = kinetics_loader(image_size)

    # define model
    feature_extractor = load_resnext101_3d_kinetics(image_size, 16).to(DEVICE)
    feature_extractor.eval()

    class_names = load_kinetics_labels('notebook/models/_data/kinetics400/train.csv')

    with torch.no_grad():

        for step, batch in enumerate(train_loader):

            sample = batch['video'].to(DEVICE, non_blocking=True)
            _, _, color, window_depth, heigth, width = sample.size()
            sample = sample.view(-1, color, window_depth, heigth, width)

            predictions = []
            for video_slice in range(sample.size(0) // 1):
                prediction = feature_extractor(sample[video_slice * 1:video_slice * 1 + 1])
                predictions.append(prediction.cpu())

            predictions = torch.cat(predictions)
            # predictions = F.softmax(predictions, dim=1)
            # sorted_scores, locs = torch.topk(predictions, k=1)
            average_scores = torch.mean(predictions, dim=0)
            sorted_scores, locs = torch.topk(average_scores, k=10)

            video_results = []
            for i in range(sorted_scores.size(0)):
                # video_results.append({'label': class_names[locs[i]], 'score': sorted_scores[i]})
                video_results.append(class_names[locs[i]])

            print(f"ID: {batch['name']} - {video_results}")


if __name__ == "__main__":
    check_resnext()

### Results

## train set
# label,youtube_id,time_start,time_end,split
#0 testifying,---QUuC4vJs,84,94,train
#1 eating spaghetti,--3ouPhoy2A,20,30,train
#2 dribbling basketball,--4-0ihtnBU,58,68,train
#3 playing tennis,--56QUhyDQM,185,195,train
#5 climbing a rope,--EaS9P7ZdQ,13,23,train
#6 brushing teeth,--IPbe5ZMCI,2,12,train
#7 balloon blowing,--Ntf6n-j9Q,17,27,train
#8 feeding birds,--PyMoD3_eg,20,30,train
#9 skiing (not slalom or crosscountry),--TBx-Spzis,148,158,train

## predictions
#0 ['testifying', 'answering questions', 'auctioning', 'giving or receiving award', 'news anchoring', 'applauding', 'laughing', 'texting', 'playing organ', 'crying']
#1 ['eating spaghetti', 'making pizza', 'eating cake', 'dining', 'tasting food', 'eating burger', 'eating doughnuts', 'baking cookies', 'eating hotdog', 'eating ice cream']
#2 ['dribbling basketball', 'playing basketball', 'shooting basketball', 'skipping rope', 'dunking basketball', 'playing badminton', 'playing volleyball', 'dodgeball', 'juggling soccer ball', 'throwing ball']
#3 ['playing tennis', 'playing basketball', 'throwing ball', 'dribbling basketball', 'shooting basketball', 'dodgeball', 'dunking basketball', 'playing squash or racquetball', 'playing badminton', 'playing volleyball']
#5 ['abseiling', 'climbing a rope', 'trimming trees', 'trapezing', 'climbing ladder', 'cleaning windows', 'rock climbing', 'bungee jumping', 'sailing', 'climbing tree']
#6 ['brushing teeth', 'eating carrots', 'washing hands', 'eating ice cream', 'sticking tongue out', 'eating chips', 'playing recorder', 'cleaning toilet', 'eating watermelon', 'tasting food']
#7 ['balloon blowing', 'exercising with an exercise ball', 'contact juggling', 'playing trumpet', 'playing didgeridoo', 'smoking hookah', 'playing harmonica', 'stretching leg', 'playing trombone', 'drinking']
#8 ['marching', 'busking', 'celebrating', 'playing bagpipes', 'skateboarding', 'cheerleading', 'zumba', 'robot dancing', 'dancing gangnam style', 'slacklining']
#9 ['skiing crosscountry', 'skiing (not slalom or crosscountry)', 'ski jumping', 'bungee jumping', 'abseiling', 'snowkiting', 'snowboarding', 'ice climbing', 'biking through snow', 'sled dog racing']

## Notes
# * do not normalise (range 0-1) for input videos
# * opencv reads bgr not rgb!
# * subtracting mean, helps improving performance
