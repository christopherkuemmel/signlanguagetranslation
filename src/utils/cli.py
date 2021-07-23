# -*- coding: utf-8 -*-
# Copyright 2019 Christopher Kümmel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the CLI Arguments."""

import argparse


def parse_arguments():
    """This method parses the given command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    arg_parser = argparse.ArgumentParser(description='ROSITA')

    # print & plot settings
    arg_parser.add_argument('--print_every', type=int, default=1, help='Print loss every n batchs')
    arg_parser.add_argument('--plot_every', type=int, default=100, help='Plot loss every n batchs')

    # training hyperparameter
    arg_parser.add_argument('--batch_size', type=int, default=1, help='Mini-Batch size')
    arg_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    arg_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning Rate')
    arg_parser.add_argument('--clip_gradients', type=int, default=5, help='Gradient clipping value')
    arg_parser.add_argument('--ignore_pad_idx', action='store_true', help='Whether to ignore padding token at loss computation')
    arg_parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer (`adam`, `sgd`)')

    # machine settings
    arg_parser.add_argument('--seed', type=int, default=42, help='Manual Seed to ensure reproducibility')
    arg_parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the DataLoader')
    arg_parser.add_argument('--enable_apex', action='store_true', help='Use apex for mixed precision training')

    # dataset settings
    arg_parser.add_argument('--image_size', type=int, default=200, help='Size of input images to resizing')
    arg_parser.add_argument('--window_size', type=int, default=16, help='Feature Extractor Window Size')
    arg_parser.add_argument('--not_shuffle', action='store_false', help='Do not shuffle Dataset')
    arg_parser.add_argument('--dataset_path', type=str, default='data/PHOENIX-2014-T-release-v3', help='Path to the PHOENIX-2014-T-release-v3 dataset')
    arg_parser.add_argument('--bpe_path', type=str, default='data/bpe', help='Path to the bpe codes and the vocab')
    arg_parser.add_argument('--flip_sequence', action='store_true', help='Flip input sequence for encoder')
    arg_parser.add_argument('--label_key', type=str, default='translation', help='Which label target to train on (`translation`, `gloss`, ..')
    arg_parser.add_argument('--preload_gpu', action='store_true', help='Preload batches to gpu device.')

    # evaluation settings
    arg_parser.add_argument('--eval_on', type=str, default='dev', help='Which dataset should be evaluated (dev|test)')
    arg_parser.add_argument('--scorers',
                            default=['bleu', 'wer', 'meteor'],
                            action='append',
                            help='What scorers to use for evaluation (bleu, wer, rogue1, rogue2, meteor, cider')

    # output settings
    arg_parser.add_argument('--output_dir', type=str, default='./output', help='Directory where to save the training output')

    return arg_parser.parse_args()
