"""Contains the torch.Dataset inherited class for loading the RWTH Phoenix Weather 2014T dataset."""

import glob
import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.transforms import ToTensor


class Phoenix2014TDataset(Dataset):
    """Phoenix 2014 T dataset reader."""
    def __init__(self, root_dir: str, split_type: str, transform: Callable = None) -> None:
        """
        Args:
            path (string): Path to the dataset root folder.
            split_type (string): Defines which dataset split the class returns. e.g. dev, train, train-complex or test.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split_type = split_type

        self.csv = pd.read_csv(self.root_dir + '/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.' + split_type + '.corpus.csv', sep='|')

        self.transform = transform

    def __len__(self) -> int:
        """Computes the lenght of the dataset."""
        return len(self.csv)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a specific item of the dataset.

        Args:
            idx (int): Index of specific element of the dataset.

        Returns:
            element (dict): Specific element of the dataset with the dictionary keys ('name', 'video', 'glosses', 'translation').

        Raises:
            IndexError: If index is out of range on the dataset.
        """
        if idx > self.__len__():
            raise IndexError
        name = self.csv.loc[idx, 'name']
        image_dir = os.path.join(self.root_dir + '/PHOENIX-2014-T/features/fullFrame-210x260px/', self.split_type, name, "*.png")

        # retrieve images, glosses and translations
        images = np.array([cv2.imread(image) for image in glob.glob(image_dir)])
        # expand dim. i.e. create a depth dim. (defaults to 1)
        images = np.expand_dims(images, axis=1)
        glosses, translation = self.get_targets(idx)

        # create return dict
        sample = {'name': name, 'video': images, 'glosses': glosses, 'translation': translation}

        # apply the transformations
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_targets(self, idx: int) -> tuple:
        """Retrieves the target gloss and translation(word) values for the requested element of the dataset.

        Args:
            idx (int): Index of specific element of the dataset.

        Returns:
            glosses, translation (list(str), list(str)): List of target glosses and translations from the specific element of the dataset.
        """
        glosses = self.csv.loc[idx, 'orth'].split()
        translation = self.csv.loc[idx, 'translation'].split()
        return glosses, translation


class Pad(object):
    """Pads the frame word and gloss count in a sample to a given size. In order to have the same frame, word and gloss dimensions over the dataset."""
    def __init__(self,
                 frame_count: int = 475,
                 word_count: int = 52,
                 gloss_count: int = 30,
                 fill_frame: int = 0,
                 fill_word: str = '<UNK>',
                 fill_gloss: str = '<UNK>') -> None:
        """
        Args:
            frame_count (int): Desired frame count.
            word_count (int): Desired word count.
            gloss_count (int): Desired gloss count.
            fill_frame (double): Pixel fill value.
            fill_word (str): Word fill value.
            fill_gloss (str): Gloss fill value.
        """
        self.max_frame_count = frame_count
        self.max_word_count = word_count
        self.max_gloss_count = gloss_count
        self.fill_frame = fill_frame
        self.fill_word = fill_word
        self.fill_gloss = fill_gloss

        print('Pad Transform for Phoenix2014TDataset is deprecated. Please use `data.masking.mask_pad()` instead.')

        ## Default values came from previous analysis of the dataset. No guarantees that those are correct.
        # train-set max word count: 52, max gloss count: 30, max frame count: 475
        # dev-set   max word count: 32, max gloss count: 18, max frame count: 272
        # test-set  max word count: 30, max gloss count: 18, max frame count: 242

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Element of the dataset with the dictionary keys ('video', 'glosses', 'translation', 'pad_video_mask', 'pad_gloss_mask', 'pad_translation_mask')
        """
        # TODO: handle 5-dim videos or remove method completely

        # retrieve sample meta information
        frame_count, C, H, W = sample['video'].size()
        word_count = len(sample['translation'])
        gloss_count = len(sample['glosses'])

        # create padded arrays with fill values
        padded_frame_array = torch.full((self.max_frame_count, C, H, W), self.fill_frame)
        padded_word_array = [self.fill_word for idx in range(self.max_word_count)]
        padded_gloss_array = [self.fill_gloss for idx in range(self.max_gloss_count)]

        # insert values into padded arrays
        padded_frame_array[:frame_count] = sample['video']
        padded_word_array[:word_count] = sample['translation']
        padded_gloss_array[:gloss_count] = sample['glosses']

        # replace sample with padded arrays
        sample['video'] = padded_frame_array
        sample['translation'] = padded_word_array
        sample['glosses'] = padded_gloss_array

        # add masking array
        sample['pad_video_mask'] = torch.zeros([self.max_frame_count], dtype=torch.bool)
        sample['pad_translation_mask'] = torch.zeros([self.max_word_count], dtype=torch.bool)
        sample['pad_gloss_mask'] = torch.zeros([self.max_gloss_count], dtype=torch.bool)

        sample['pad_video_mask'][:frame_count] = 1
        sample['pad_translation_mask'][:word_count] = 1
        sample['pad_gloss_mask'][:gloss_count] = 1

        return sample


def pad_collate(batch: list) -> dict:
    """Custom collate method which returns the current batch with padded values.

    Args:
        batch (list): The list of items of the batch (unpadded).

    Returns:
        dict: Padded batch elements of the dataset with the dictionary keys 
              ('name', 'video', 'glosses', 'translation'). The value from key 'name' may not be unique!
    """
    token_pad_value = ('<UNK>', )
    batch_size = len(batch)

    # if video is not a torch tensor make one
    if not isinstance(batch[0]['video'], torch.Tensor):
        for sample in batch:
            to_tensor = ToTensor()
            to_tensor(sample)

    if batch_size == 1:
        return {
            'name': [batch[0]['name']],
            'video': batch[0]['video'].unsqueeze(0),
            'translation': [[(sample, ) for sample in batch[0]['translation']]],
            'glosses': [[(sample, ) for sample in batch[0]['glosses']]]
        }

    # retrieve sample meta information
    name = [sample['name'] for sample in batch]
    _, color, depth, heigth, width = batch[0]['video'].size()
    video_length = [sample['video'].size(0) for sample in batch]
    translation_length = [len(sample['translation']) for sample in batch]
    gloss_length = [len(sample['glosses']) for sample in batch]

    # create padded arrays with fill values
    padded_frames = torch.zeros((batch_size, max(video_length), color, depth, heigth, width))
    padded_translations = [[token_pad_value for idx in range(max(translation_length))] for batch in range(batch_size)]
    padded_glosses = [[token_pad_value for idx in range(max(gloss_length))] for batch in range(batch_size)]

    # insert values into padded arrays
    for sample_idx in range(batch_size):
        padded_frames[sample_idx][:video_length[sample_idx]] = batch[sample_idx]['video']
        padded_translations[sample_idx][:translation_length[sample_idx]] = [(sample, ) for sample in batch[sample_idx]['translation']]
        padded_glosses[sample_idx][:gloss_length[sample_idx]] = [(sample, ) for sample in batch[sample_idx]['glosses']]

    return {'name': name, 'video': padded_frames, 'translation': padded_translations, 'glosses': padded_glosses}
