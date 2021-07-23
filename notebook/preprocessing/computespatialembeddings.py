# -*- coding: utf-8 -*-
"""Contains a notebook script for computing spatial embeddings for the GebaerdenLernen and the RWTH Phoenix 2014T Weather dataset."""
import glob
import os
import time

import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import Compose

from data.phoenix2014Tdataset import Phoenix2014TDataset
from data.transforms import ToTensor
from models.resnext import load_resnext50_32x4d

# PyTorch Settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dir(output_dir: str) -> None:
    """Creating a new directory.
    
    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_model() -> torch.nn.Module:
    """Loads and returns the ResNext50_32x4d model without the last layer in eval mode.

    Returns:
        torch.nn.Module: ResNext50_32x4d model without the last layer in eval mode.
    """
    # loading resnet model
    model = load_resnext50_32x4d(remove_last_layer=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def process_gebaerdenlernen_embeddings(
        output_dir: str = './output/gebaerdenlernen') -> None:
    """Compute spatial embeddings from the GebaerdenLernen dataset.

    Args:
        output_dir (str): The directory where the spatial embeddings should be saved.
    """
    create_dir(output_dir)
    model = load_model()

    idx = 0
    start_time = time.time()

    video_paths = glob.glob('./data/gebaerdenlernen/features/mp4/*.mp4')

    with torch.no_grad():
        for video in video_paths:

            video_name = video.split('/')[-1][:-4]  # filename - file ending

            # Lormen videos are way longer than the others. Therefore, we need to resize them to a smaller size.
            if video.split('/')[-1][:6] == 'lormen':
                width, heigth = 120, 160  # 25% image size (original = (480/640))
            else:
                width, heigth = 360, 480  # 75% image size (original = (480/640))

            # get video
            images = []
            cap = cv2.VideoCapture(video)
            while (cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(cv2.resize(frame, (width, heigth)))

            # transpose from numpy axis to pytorch tensors | numpy image: H x W x C torch image: C X H X W
            inputs = torch.tensor(np.array(images).transpose(
                (0, 3, 1, 2))).float().to(DEVICE)

            # predict
            outputs = model(inputs)

            # save embedding
            torch.save(outputs.detach(),
                       os.path.join(output_dir, video_name + '.pt'))
            print(
                f"Idx: {idx:3}/{len(video_paths)} - {idx/len(video_paths)*100:5.2f} - Time since start: {time.time() - start_time:8.4f} - Saving video: {video_name}"
            )
            idx += 1


def process_rwth_phoenix_embeddings(output_dir: str = './output/phoenix',
                                    split_type: str = 'train') -> None:
    """Compute spatial embeddings from the RWTH Phoenix 2014T dataset.

    Args:
        output_dir (str): The directory where the spatial embeddings should be saved.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """
    create_dir(output_dir)
    model = load_model()

    # DatasetLoader settings
    transforms = Compose([ToTensor()])
    params = {
        'num_workers': 0,
        'pin_memory': True
    }  #! only works with batch_size = 1

    dataset = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3',
                                  split_type=split_type,
                                  transform=transforms)
    dataset_generator = data.DataLoader(dataset, **params)

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataset_generator):
            # flatten input for resnext model
            inputs = torch.flatten(sample_batched['video'],
                                   end_dim=1).to(DEVICE)

            # predict
            outputs = model(inputs)

            video_name = sample_batched['name'][
                0]  #! only works with batch_size = 1

            # save embedding
            torch.save(outputs.detach(),
                       os.path.join(output_dir, video_name + '.pt'))
            print(
                f"Idx: {batch_idx:3}/{len(dataset_generator)} - {batch_idx/len(dataset_generator)*100:5.2f} - Time since start: {time.time() - start_time:8.4f} - Saving video: {video_name}"
            )


if __name__ == "__main__":
    process_gebaerdenlernen_embeddings(
        output_dir='./output/gebaerdenembeddings')
    process_rwth_phoenix_embeddings(
        output_dir='./output/phoenixembeddings/train', split_type='train')
    process_rwth_phoenix_embeddings(output_dir='./output/phoenixembeddings/dev',
                                    split_type='dev')
    process_rwth_phoenix_embeddings(
        output_dir='./output/phoenixembeddings/test', split_type='test')
