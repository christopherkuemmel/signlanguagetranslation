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

import logging
import os
from ast import literal_eval

import torch
from torch.nn.functional import one_hot


class OneHotEncoding():
    """OneHotEncoding for a list of words."""

    def __init__(self, word_space: list = [], file_path: str = '') -> None:
        """OneHotEncoding for a given list of words.

        Args:
            word_space (list(str), optional): Input list of words to create a dictionary from for the one-hot encoding. (doubled entries are okay)
            path (str, optional): Path to an existing one-hot encoding file or where to save the new file
        """
        assert ((word_space != [] and file_path != '') or
                (word_space == [] and file_path != '')), 'You either have to specify a word_space and a path (new encoding with output path) or just a path (existing encoding) parameter!'

        if word_space != []:
            # start with EOS and SOS
            self.word_space = {'<EOS>', '<SOS>', '<UNK>'}

            # add word_space
            self.word_space.update([word.lower() for word in word_space])

            # define a mapping of words to integers
            self.word2index = dict((w, i) for i, w in enumerate(self.word_space))
            self.index2word = dict((i, w) for i, w in enumerate(self.word_space))

            # save one-hot encodings as file for debugging purposes
            output_dir = os.path.join(file_path, 'encoding')

            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except:
                    logger = logging.getLogger('SignLanguageTranslation')
                    logger.error("Error while try to create directory for one-hot encoding.")

            with open(os.path.join(output_dir, 'one-hot-encoding_word2index.txt'), 'w+') as one_hot_file_w2i:
                one_hot_file_w2i.write(str(self.word2index))
            with open(os.path.join(output_dir, 'one-hot-encoding_index2word.txt'), 'w+') as one_hot_file_i2w:
                one_hot_file_i2w.write(str(self.word2index))
        else:
            # load one-hot encoding
            encoding_file = open(file_path, 'r', encoding='utf-8')
            self.word2index = literal_eval(encoding_file.readline())
            self.index2word = dict([[v, k] for k, v in self.word2index.items()])
            self.word_space = [k for k, v in self.word2index.items()]

    def __len__(self):
        return len(self.word_space)

    def __call__(self, words: list) -> torch.tensor:
        """Creates one-hot encoded tensors based on a given list of words.

        Args:
            words (list(str)): Input list of words to create a tensor of one-hot encoded words.

        Returns:
            torch.tensor(): Tensor with the dimensions (len(words), len(word_space)) that consists of zeros and one at the index of the input words classes.
        """
        # check if given words are a list
        if not isinstance(words, list):
            words = [words]

        # retrieve indexes for each word
        idx_word_list = [self.word2index[word] for word in words]

        # create one-hot encoded tensor for all words
        one_hot_encoded_words = one_hot(torch.tensor(idx_word_list), num_classes=len(self.word_space))

        return one_hot_encoded_words
