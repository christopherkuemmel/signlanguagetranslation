from typing import Tuple, Union

import cv2
import fastBPE
import numpy as np
import torch
from sacremoses import MosesDetokenizer, MosesTokenizer


class Resize(object):
    """Resize the video (images) in a sample to a given size."""
    def __init__(self, output_size: Union[Tuple[int, int], int]) -> None:
        """
        Args:
            output_size (tuple or int): Desired output size. If single int is given the output will be cubed.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the resized video (as torch.tensor).
        """
        # TODO: sample['video'] := Union[torch.Tensor, PIL.Image, np.ndarray]

        # convert to numpy images for opencv resize method
        if isinstance(sample['video'], torch.Tensor):
            to_numpy = ToNumpy()
            sample = to_numpy(sample)
        elif not isinstance(sample['video'], np.ndarray):
            raise NotImplementedError('Given video is neither of type np.ndarray nor torch.tensor!')

        # resize every image in video
        if len(sample['video'].shape) == 5:
            sample['video'] = np.array([[cv2.resize(image, self.output_size) for image in depth_window] for depth_window in sample['video']])
        elif len(sample['video'].shape) == 4:
            sample['video'] = np.array([cv2.resize(image, self.output_size) for image in sample['video']])
        elif len(sample['video'].shape) == 3:
            sample['video'] = cv2.resize(sample['video'], self.output_size)
        else:
            raise TypeError(f"Given video has {len(sample['video'].shape)} dimensions! This class only supports dimensions of length 3, 4 or 5.")

        # convert back to tensors
        to_tensor = ToTensor()
        sample = to_tensor(sample)

        return sample


class ToNumpy(object):
    """Convert tensors in sample to np.ndarrays."""
    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the converted (and reshaped) video.
        """
        # TODO: sample['video'] := Union[torch.Tensor, PIL.Image]

        # if video has a depth dim
        if len(sample['video'].size()) == 5:
            # swap color axis because (multiple images with additional depth dim)
            # torch image: T x C x D x H X W
            # numpy image: T x D x H x W x C
            sample['video'] = sample['video'].permute(0, 2, 3, 4, 1).numpy()
        elif len(sample['video'].size()) == 4:
            # swap color axis because (multiple images)
            # torch image: T x C X H X W
            # numpy image: T x H x W x C
            sample['video'] = sample['video'].permute(0, 2, 3, 1).numpy()
        elif len(sample['video'].size()) == 3:
            # swap color axis because (single image)
            # torch image: T x C X H X W
            # numpy image: T x H x W x C
            sample['video'] = sample['video'].permute(1, 2, 0).numpy()

        return sample


class ToTensor(object):
    """Convert list of np.ndarrays in sample to tensors."""
    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Element of the dataset with the dictionary keys ('video', 'glosses', 'translation')
        """
        # TODO: sample['video'] := Union[np.ndarray, PIL.Image]

        # if video has a depth dim
        if len(sample['video'].shape) == 5:
            # swap color axis because (multiple images with additional depth dim)
            # numpy image: D x H x W x C
            # torch image: C x D x H X W
            video = [image.transpose((3, 0, 1, 2)) for image in sample['video']]
            sample['video'] = torch.tensor(np.array(video), dtype=torch.float32)
        elif len(sample['video'].shape) == 4:
            # swap color axis because (multiple images)
            # numpy image: H x W x C
            # torch image: C X H X W
            video = [image.transpose((2, 0, 1)) for image in sample['video']]
            sample['video'] = torch.tensor(np.array(video), dtype=torch.float32)
        elif len(sample['video'].shape) == 3:
            # swap color axis because (single image)
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['video'] = torch.tensor(np.array(sample['video'].transpose(2, 0, 1)), dtype=torch.float32)

        return sample


class Mean(object):
    """Subtracts the given mean (np.ndarray) from the sample video tensors."""
    def __init__(self, mean: np.ndarray) -> None:
        """
        Args:
            mean (np.ndarray): Mean array which should be subtracted from the sample images.
        """
        # TODO: mean := Union[np.ndarray, torch.tensor]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        self.mean = torch.tensor(mean.transpose((2, 0, 1)), dtype=torch.float32)

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Element of the dataset with the dictionary keys ('video', 'glosses', 'translation')
        """
        # TODO: take care of 5-dim input
        # TODO: sample['video'] := Union[np.ndarray, torch.tensor]
        # TODO: do not subtract mean from padded zero valued arrays
        sample['video'] = sample['video'] - self.mean
        return sample


class WindowDepth(object):
    """Reshapes the input (video) frames to have a depth dimension."""
    def __init__(self, window_size: int) -> None:
        """
        Args:
            window_size (int): The size of the window to reshape the frames to.
        """
        self.window_size = window_size

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Element of the dataset with the dictionary keys ('video', 'glosses', 'translation')
        """
        # if video is not a torch tensor make one
        if not isinstance(sample['video'], torch.Tensor):
            to_tensor = ToTensor()
            to_tensor(sample)

        seq_len, color, depth, heigth, width = sample['video'].size()

        # permute color and depth dims for further processing
        # T x C x D x H x W -> T x D x C x H x W
        sample['video'] = sample['video'].permute(0, 2, 1, 3, 4)

        # flatten seq_len and depth
        sample['video'] = sample['video'].contiguous().flatten(end_dim=1)

        frame_count = seq_len * depth

        # if frame_count of video is dividable by window_size we can directly reshape. otherwise we need to pad the video with zero frames.
        if not frame_count % self.window_size == 0:
            pad_size = self.window_size - frame_count % self.window_size
            pad_tensor = torch.zeros((pad_size, color, heigth, width))
            sample['video'] = torch.cat((sample['video'], pad_tensor))

        sample['video'] = sample['video'].view(-1, self.window_size, color, heigth, width)

        # revert processing permutation
        # T x D x C x H x W -> T x C x D x H x W
        sample['video'] = sample['video'].permute(0, 2, 1, 3, 4)
        return sample


class Tokenize(object):
    """Tokenizes the input with the `sacremoses` tokenizer.

    https://github.com/alvations/sacremoses
    """
    def __init__(self, language: str = 'de') -> None:
        """
        Args:
            language (str): Input language (default is `de`).
        """
        self.language = language
        self.tokenizer = MosesTokenizer(lang=language)

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the tokenized strings (for all keys containing (lists of) strings).
        """

        for key in sample.keys():
            if key == 'name' or isinstance(sample[key], (torch.Tensor, np.ndarray)):
                continue
            if isinstance(sample[key], list):
                sample[key] = self.tokenizer.tokenize(' '.join(sample[key]))
            else:
                sample[key] = self.tokenizer.tokenize(sample[key])
        return sample


class Detokenize(object):
    """Detokenizes the input with the `sacremoses` tokenizer.

    https://github.com/alvations/sacremoses
    """
    def __init__(self, language: str = 'de') -> None:
        """
        Args:
            language (str): Input language (default is `de`).
        """
        self.language = language
        self.detokenizer = MosesDetokenizer(lang=language)

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the detokenized strings (for all keys containing (lists of) strings).
        """
        for key in sample.keys():
            if key == 'name' or isinstance(sample[key], (torch.Tensor, np.ndarray)):
                continue
            if isinstance(sample[key], list):
                sample[key] = self.detokenizer.detokenize(sample[key]).split()
            else:
                sample[key] = self.detokenizer.detokenize(sample[key].split()).split()
        return sample


class ApplyBPE(object):
    """Apply fastBPE on input.

    https://github.com/glample/fastBPE
    """
    def __init__(self, bpe_codes_path: str, vocab_path: str) -> None:
        """
        Args:
            bpe_codes_path (str): Path to bpe codes.
            vocab_path (str): Path to vocab.
        """
        self.bpe = fastBPE.fastBPE(bpe_codes_path, vocab_path)

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the bpe applid strings (for all keys containing (lists of) strings).
        """
        for key in sample.keys():
            if key == 'name' or isinstance(sample[key], (torch.Tensor, np.ndarray)):
                continue
            sample[key] = ' '.join(self.bpe.apply(sample[key])).split()
        return sample


class RemoveBPE(object):
    """Remove fastBPE on input.

    https://github.com/glample/fastBPE
    """
    def __init__(self, bpe_symbol: str = '@@ ') -> None:
        """
        Args:
            bpe_codes_path (str): The BPE symbol to decode.
        """
        self.bpe_symbol = bpe_symbol

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset containing the bpe applid strings (for all keys containing (lists of) strings).
        """
        for key in sample.keys():
            if key == 'name' or isinstance(sample[key], (torch.Tensor, np.ndarray)):
                continue
            sample[key] = (' '.join(sample[key]) + ' ').replace('@@ ', '').rstrip()
        return sample


class ToGPU(object):
    """Moves the batch to the selected GPU."""
    def __init__(self, device: torch.device):
        """
        Args:
            device (torch.device): The gpu device to pass the sample to. 
        """
        self.device = device

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): Element of the dataset to transform.

        Returns:
            element (dict): Same element of the dataset with all tensors moved to device.
        """
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].to(self.device, non_blocking=True)
        return sample
