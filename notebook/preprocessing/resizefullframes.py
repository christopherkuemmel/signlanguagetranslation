"""This script contains methods to manipulate/ resize the sign language datasets."""

import os

import cv2
from torch.utils import data
from torchvision.transforms import Compose

from data.phoenix2014Tdataset import Phoenix2014TDataset
from data.transforms import Resize


def create_dir(output_dir: str) -> None:
    """Creating a new directory.

    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def resize_phoenix_to_227_px(output_dir: str = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/resizedFrame-227x227px', split_type: str = 'train'):
    """Resizes the RWTH Phoenix Weather 2014T dataset to 227x227 pixels.

    Args:
        output_dir (str): The directory to save all images.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """
    output_dir = os.path.join(output_dir, split_type)
    create_dir(output_dir)

    # DatasetLoader settings
    transforms = Compose([Resize(227)])
    params = {'num_workers': 12}  #! only works with batch_size = 1

    dataset = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type=split_type, transform=transforms)
    dataset_generator = data.DataLoader(dataset, **params)

    for batch in dataset_generator:
        video_name = batch['name']
        create_dir(os.path.join(output_dir, video_name[0]))
        for idx, frame in enumerate(batch['video']):
            frame = frame.squeeze(0).numpy()
            cv2.imwrite(os.path.join(output_dir, video_name[0], f'{idx+1:04}.png'), frame)


if __name__ == "__main__":
    resize_phoenix_to_227_px(split_type='train')
    resize_phoenix_to_227_px(split_type='test')
    resize_phoenix_to_227_px(split_type='dev')
