# Copyright 2019 Christopher Kümmel

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains helper functions for masking padded datasets."""
import torch


def mask_pad(batch: dict) -> dict:
    """Masks all items in a given sample (dict) and returns them.
    Args:
        batch (dict): The sample batch to mask.

    Returns:
        dict: A new dict with the same keys as the given sample but with boolean masks.

    Raises:
        NotImplementedError: If batch contains other elements then list of strings or torch.Tensor
    """
    _batch = {}
    for key in batch.keys():
        # special case: name key is never padded
        if key == 'name':
            _batch[key] = torch.tensor([False for item in batch[key]])

        # check type of keys
        elif isinstance(batch[key], torch.Tensor):
            # handle 4- and 5-dimensinal input
            # videos may have an additional depth dimension
            if len(batch['video'].size()) == 6:
                _batch[key] = (batch['video'].sum(dim=(2, 3, 4, 5)) == 0.)
            elif len(batch['video'].size()) == 5:
                _batch[key] = (batch['video'].sum(dim=(2, 3, 4)) == 0.)
            else:
                raise TypeError('The input tensor is neither 4- or 5-dimensional!')
        elif isinstance(batch[key], list):
            _batch[key] = torch.tensor([[item[0] == '<UNK>' for item in sample] for sample in batch[key]])
        else:
            raise NotImplementedError
    return _batch
