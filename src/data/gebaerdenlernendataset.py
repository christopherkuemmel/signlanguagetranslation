import os
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GebaerdenLernenDataset(Dataset):
    """Gebaerden Lernen dataset reader."""
    def __init__(self, root_dir: str, file_type: Optional[str] = 'mp4', transform: Optional[Callable] = None) -> None:
        """
        Args:
            root_dir (str): Path to the dataset folder
            file_type (str): Which file type should be loaded

        Raises:
            NotImplementedError if file type other than 'mp4' is specified
        """

        if file_type != 'mp4':
            raise NotImplementedError

        self.root_dir = root_dir
        self.file_type = file_type
        self.transform = transform
        self.csv = pd.read_csv(os.path.join(self.root_dir, 'annotations/gebaerdenlernen.' + self.file_type + '.csv'), sep='|')

    def __len__(self) -> int:
        """Computes the lenght of the dataset."""
        return len(self.csv)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a specific item of the dataset.

        Args:
            idx (int): Index of specific element of the dataset.

        Returns:
            element (dict): Specific element of the dataset with the dictionary keys ('name', 'video', 'glosses'). The value from key 'name' may not be unique!

        Raises:
            IndexError: If index is out of range on the dataset.
        """
        if idx > self.__len__():
            raise IndexError
        image_dir = os.path.join(self.root_dir, self.csv.loc[idx, 'video'])
        name = image_dir.split('/')[-1][:-4]

        if self.file_type == 'mp4':
            images = []
            cap = cv2.VideoCapture(image_dir)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(frame)
        else:
            raise NotImplementedError

        # TODO: expand image array to have a depth dim

        gloss = self.csv.loc[idx, 'gloss'].split()

        # create return dict
        sample = {'name': name, 'video': np.array(images), 'gloss': gloss}

        # apply the transformations
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def download(path: str = '', file_type: str = 'mp4'):
        """Downlaods and saves the dataset locally

        Args:
            path (str): Where the dataset should be downloaded
            file_type (str): Which file_type should be downloaded. Default is 'mp4'.
                Type options {'mp4', 'npy', 'pt'}

        Raises:
            NotImplementedError
        """
        raise NotImplementedError


def pad_collate(batch: list) -> dict:
    """TODO: Custom collate method which returns the current batch with padded values.

    Args:
        batch (list): The list of items of the batch (unpadded).

    Returns:
        dict: Padded batch elements of the dataset with the dictionary keys ('name', 'video', 'glosses'). The value from key 'name' may not be unique!
    """
    # TODO: handle 5-dim input
    raise NotImplementedError
