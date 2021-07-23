"""Contains the torch.Dataset inherited class for loading the DGSCorpus."""
import json
import os
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import ToTensor


class DGSDataset(Dataset):
    """DGS Corpus dataset reader."""
    def __init__(self, root_dir: str, token_level: str, language: str, transform: Optional[Callable] = None) -> None:
        """
        Args:
            root_dir (str): Path to the dataset folder
            token_level (str): The level of tokenization. Either sentnce- or gloss-level.
            language (str): The language of the transcription
            transform (callable, optional): Optional transform to be applied on a sample.

        Raises:
            NotImplementedError if file type other than 'sentence-level' is specified.
        """
        if token_level != 'sentence-level':
            raise NotImplementedError

        self.root_dir = root_dir
        self.transform = transform
        self.token_level = token_level
        self.language = language

        self.feature_dir = os.path.join(root_dir, 'features', token_level, language)
        self.annotation_dir = os.path.join(root_dir, 'annotations', token_level)

        with open(os.path.join(self.annotation_dir, 'DGSCorpus.' + language + '.json'), 'r') as annotations_file:
            json_str = annotations_file.read()
        self.annotations = json.loads(json_str)

    def __len__(self) -> int:
        """Computes the lenght of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a specific item of the dataset.

        Args:
            idx (int): Index of specific element of the dataset.

        Returns:
            element (dict): Specific element of the dataset with the dictionary keys 
                            ('name', 'video', 'glosses', 'translation', 'lexeme', 'mouth'). The value from key 'name' may not be unique!

        Raises:
            IndexError: If index is out of range on the dataset.
        """
        if idx > self.__len__():
            raise IndexError

        name = self.annotations[idx]['name']
        image_dir = os.path.join(self.feature_dir, name, name + '.mp4')
        # image_dir = os.path.join(self.root_dir, self.csv.loc[idx, 'video'])
        # name = image_dir.split('/')[-1][:-4]

        # load video
        images = []
        cap = cv2.VideoCapture(image_dir)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            images.append(frame)

        # TODO: expand image array to have a depth dim

        # TODO: filter special chars?
        translation = self.annotations[idx]['translation']['text'].split()

        # helper function to compute corresponding frames of reference timestamps
        # reference timestamp // (1 / fps * 1000) (duration of one frame in ms)
        ref_to_frame = lambda ref: self.annotations[idx]['timestamps'][ref] // int(1 / 50 * 1000)

        glosses = [(gloss['text_simplified'], ref_to_frame(gloss['start_ref']), ref_to_frame(gloss['end_ref'])) for gloss in self.annotations[idx]['signs']]
        lexeme = [(lexem['text_simplified'], ref_to_frame(lexem['start_ref']), ref_to_frame(lexem['end_ref'])) for lexem in self.annotations[idx]['lexeme']]
        mouth = [(mouth['text'], ref_to_frame(mouth['start_ref']), ref_to_frame(mouth['end_ref'])) for mouth in self.annotations[idx]['mouth']]

        # create return dict
        #! glosses == signs
        sample = {'name': name, 'video': np.array(images), 'translation': translation, 'glosses': glosses, 'lexeme': lexeme, 'mouth': mouth}

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
    """Custom collate method which returns the current batch with padded values.

    Args:
        batch (list): The list of items of the batch (unpadded).

    Returns:
        dict: Padded batch elements of the dataset with the dictionary keys 
              ('name', 'video', 'glosses', 'translation', 'lexeme', 'mouth'). The value from key 'name' may not be unique!
    """
    token_pad_value = ('<UNK>', )
    batch_size = len(batch)

    # if video is not a torch tensor make one
    if not isinstance(batch[0]['video'], torch.Tensor):
        for sample in batch:
            to_tensor = ToTensor()
            to_tensor(sample)

    # TODO: handle 5-dim input

    # retrieve sample meta information
    name = [sample['name'] for sample in batch]
    _, C, H, W = batch[0]['video'].size()
    frame_count = [sample['video'].size(0) for sample in batch]
    translation_count = [len(sample['translation']) for sample in batch]
    gloss_count = [len(sample['glosses']) for sample in batch]
    lexeme_count = [len(sample['lexeme']) for sample in batch]
    mouth_count = [len(sample['mouth']) for sample in batch]

    # create padded arrays with fill values
    padded_frames = torch.zeros((batch_size, max(frame_count), C, H, W))
    padded_translations = [[token_pad_value for idx in range(max(translation_count))] for batch in range(batch_size)]
    padded_glosses = [[token_pad_value for idx in range(max(gloss_count))] for batch in range(batch_size)]
    padded_lexeme = [[token_pad_value for idx in range(max(lexeme_count))] for batch in range(batch_size)]
    padded_mouth = [[token_pad_value for idx in range(max(mouth_count))] for batch in range(batch_size)]

    # insert values into padded arrays
    for sample_idx in range(batch_size):
        padded_frames[sample_idx][:frame_count[sample_idx]] = batch[sample_idx]['video']
        padded_translations[sample_idx][:translation_count[sample_idx]] = batch[sample_idx]['translation']
        padded_glosses[sample_idx][:gloss_count[sample_idx]] = batch[sample_idx]['glosses']
        padded_lexeme[sample_idx][:lexeme_count[sample_idx]] = batch[sample_idx]['lexeme']
        padded_mouth[sample_idx][:mouth_count[sample_idx]] = batch[sample_idx]['mouth']

    return {'name': name, 'video': padded_frames, 'translation': padded_translations, 'glosses': padded_glosses, 'lexeme': padded_lexeme, 'mouth': padded_mouth}
