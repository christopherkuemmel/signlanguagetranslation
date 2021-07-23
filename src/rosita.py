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
"""Main class and entrypoint for ROSITA."""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fairseq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import data.transforms as tr
import utils
from data.masking import mask_pad
from data.phoenix2014Tdataset import Phoenix2014TDataset, pad_collate
from evaluation.metrics import Scorers
from models.sign2text import (FeatureExtractor, Sign2Text, TranslationDecoder, TranslationEncoder)

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

LOGGER = logging.getLogger()


def init_weights(module: nn.Module, range: float):
    """Uniformly initializes a torch.nn.Module (inplace).

    Args:
        module (nn.Module): The module to initialize weights.
        range (float): The range for the uniform distribution.
    """
    for _, param in module.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def create_dir(output_dir: str) -> None:
    """Creating a new directory.

    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            LOGGER.error("Error while try to create directory for saving torch model.")
            raise OSError


def binarize(sentences: List[List[Tuple[str, ]]], vocab: fairseq.data.dictionary.Dictionary, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
    """Binarizes the given sentences to LongTensor based on the given vocab dictionary.

    Args:
        sentences (List[List[Tuple[str,]]]): The sentences to binarize with size (batch, words (target_length)).
        vocab (fairseq.data.dictionary.Dictionary): The vocab to binarize with.
        add_bos (bool): Whether `begin of sentence` (bos-token) should be prepended.
        add_eos (bool): Whether `end of sentence` (eos-token) should be appended.

    Returns:
        torch.LongTensor: The binarized token tensor.
    """
    if add_bos:
        # add bos token to all sentences
        sentences = [[('<s>', )] + sentence for sentence in sentences]
    # join all words from a sentence for each sentence in batch
    # then create LongTensors from each sentence based on the vocab
    return torch.stack(
        [vocab.encode_line(' '.join([word[0] for word in sentence]), add_if_not_exist=False, append_eos=add_eos).long() for sentence in sentences])


def debinarize(tokens: torch.LongTensor, vocab: fairseq.data.dictionary.Dictionary) -> List[List[Tuple[str, ]]]:
    """Binarizes the given sentences to LongTensor based on the given vocab dictionary.

    Args:
        tokens (torch.LongTensor): The token tensor to debinarize.
        vocab (fairseq.data.dictionary.Dictionary): The vocab to debinarize with.

    Returns:
        List[List[Tuple[str,]]]: List of sentences (List of words).
    """
    #! sample['translation'] != debinarize(binarize(y, decoder.vocab), decoder.vocab) -> lost of information of positions (within tuple)
    return [[(token, ) for token in vocab.string(tensor).split()] for tensor in tokens]


def compute_metrics(scorers: Scorers, hypothesis: List[List[Tuple[str, ]]],
                    references: List[List[Tuple[str, ]]]) -> Dict[str, Tuple[Optional[float], Optional[List[float]], Optional[Dict[str, float]]]]:
    """" Computes the metric scores.

    Args:
        scorers (Scorers): The scorers to compute the metrics.
        hypothesis (List[List[Tuple[str, ]]]): The list of sentences predicted by the model.
            Each element contains a List of sentencens -> List of Tuples -> Words, (int).
        references (List[List[Tuple[str, ]]]): The list of list of references.
            Each reference set needs to have the same amout of sentences like the hypothesis!
            Each element contains a List of sentencens -> List of Tuples -> Words, (int).

    Returns:
        Dict[str, Tuple[Optional[float], Optional[List[float]]], Optional[Dict[str, float]]]: from Scorers.compute_metrics()
            str: The name of the scorer.
            tuple:
                - corpus_score: Optional[float] = None
                - sent_scores: Optional[List[float]] = None
                - group_scores: Optional[Dict[str, float]] = None
    """
    def remove_tuple(list_of_tuples: List[List[Tuple[str, ]]]) -> List[List[str]]:
        """Removes the inner tuple in the list of lists."""
        return [[token[0] for token in sentence] for sentence in list_of_tuples]

    remove_bpe = tr.RemoveBPE()

    hypothesis = remove_tuple(hypothesis)
    references = remove_tuple(references)

    hypothesis = [remove_bpe({'translation': sentence})['translation'] for sentence in hypothesis]
    references = [remove_bpe({'translation': sentence})['translation'] for sentence in references]

    #? do we need to detokenize?
    # detokenize = tr.Detokenize()
    # hypothesis = [' '.join(detokenize({'translation': sentence})['translation']) for sentence in hypothesis]
    # references = [' '.join(detokenize({'translation': sentence})['translation']) for sentence in references]

    return scorers.compute_scores(hypothesis, [references])


def train(model: nn.Module, data_loader: DataLoader, optimizer, criterion, args: argparse.Namespace, device) -> float:
    """Training Loop.

    Args:
        model (torch.nn.Module): The model to train with.
        data_loader (torch.utils.data.DataLoader): The DataLoader with the initialized dataset (including transforms!).
        optimizer (torch.optim.): Pytorch optimizer to adjust model parameters.
        criterion (torch.nn.): Pytorch loss function.
        args (argparse.Namespace): Rosita cli arguments.

    Returns:
        float: The loss of one full epoch (normalised).
    """
    model.train()

    loss = 0.
    start_time = time.time()

    for step, batch in enumerate(data_loader):
        sample = batch['video'].to(device, non_blocking=True)
        sample_mask = mask_pad(batch)['video'].to(device, non_blocking=True)

        label = batch[args.label_key]
        label = binarize(label, model.decoder.vocab).to(device, non_blocking=True)

        optimizer.zero_grad()

        # TODO: teacher forcing
        sample = model(sample, sample_mask, label)

        # TODO: different loss functions need different tensor sizes
        sample = criterion(sample[0].view(-1, sample[0].size(-1)), label.view(-1))

        if args.enable_apex:
            with amp.scale_loss(sample, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            sample.backward()

        optimizer.step()

        loss += sample.item()

        if step % args.print_every == 0 and not step == 0:
            LOGGER.info(
                f'  Batch {step:6}/{len(data_loader):6}    Loss: {sample.item():7.4f}    Remaining: {utils.time.remaining(start_time, (step/len(data_loader)))}'
            )

    LOGGER.info(f'  Epoch took: {utils.time.asHMS(time.time() - start_time)}')
    LOGGER.info(f'  Average batch loss: {loss / len(data_loader):7.4f}')
    return loss / len(data_loader)


def evaluate(model: nn.Module, data_loader: DataLoader, criterion, args: argparse.Namespace,
             device) -> Tuple[float, Tuple[List[List[Tuple[str, ]]], List[List[Tuple[str, ]]]]]:
    """Evaluation Loop.

    Args:
        model (torch.nn.Module): The model to evaluate with.
        data_loader (torch.utils.data.DataLoader): The DataLoader with the initialized dataset (including transforms!).
        criterion (torch.nn.): Pytorch loss function.
        args (argparse.Namespace): Rosita cli arguments.

    Returns:
        Tuple[float, Tuple[List[List[Tuple[str, ]]], List[List[Tuple[str, ]]]]]:
            - float: The loss of one full epoch (normalised).
            - Tuple: Each element contains a List of sentencens -> List of Tuples -> Words, (int)
                List[List[Tuple[str, ]]]: Hypothesis
                List[List[Tuple[str, ]]]: References
    """
    model.eval()

    loss = 0.
    start_time = time.time()

    with torch.no_grad():

        hypothesis = []
        references = []
        for step, batch in enumerate(data_loader):

            sample = batch['video'].to(device, non_blocking=True)
            sample_mask = mask_pad(batch)['video'].to(device, non_blocking=True)

            label = batch[args.label_key]
            references.extend(label)
            label = binarize(label, model.decoder.vocab).to(device, non_blocking=True)

            sample = model(sample, sample_mask, label)
            hypo = sample[0].to(torch.device('cpu'), non_blocking=True)

            # -> ValueError: Expected target size (1, 42024), got torch.Size([1, 19])
            sample = criterion(sample[0].view(-1, sample[0].size(-1)), label.view(-1))

            loss += sample.item()

            if step % args.print_every == 0 and not step == 0:
                LOGGER.info(
                    f'  Batch {step:6}/{len(data_loader):6}    Loss: {sample.item():7.4f}    Remaining: {utils.time.remaining(start_time, (step/len(data_loader)))}'
                )

            # return debinarized hypothesis
            # get best class idx (predicted token)
            # topk(1)[1] gets only the top 1 class & idx of class
            hypo = hypo[0].topk(1)[1].squeeze()
            hypothesis.extend(debinarize(hypo.view(1, -1), model.decoder.vocab))

    LOGGER.info(f'  Evaluation took: {utils.time.asHMS(time.time() - start_time)}')
    LOGGER.info(f'  Validation loss: {loss / len(data_loader):7.4f}')
    return (loss / len(data_loader), (hypothesis, references))


def main(args: argparse.Namespace):
    """Rositas' main.

    Args:
        args (argparse.Namespace): Rosita cli arguments.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set seed for reproducability
    torch.manual_seed(args.seed)

    # create output dir
    create_dir(os.path.join(args.output_dir, 'model'))

    transforms = Compose([
        tr.Resize(args.image_size),
        tr.WindowDepth(args.window_size),
        tr.Tokenize(),
        tr.ApplyBPE(os.path.join(args.bpe_path, 'bpecodes'), os.path.join(args.bpe_path, 'dict.de.txt')),
    ])

    if args.preload_gpu:
        transforms.transforms.append(tr.ToGPU(device))

    loader_params = {
        'batch_size': args.batch_size,
        'collate_fn': pad_collate,
        'shuffle': args.not_shuffle,
        'num_workers': args.num_workers,
        'pin_memory': not args.preload_gpu,
    }

    # define dataset
    train_set = Phoenix2014TDataset(args.dataset_path, 'train', transforms)
    eval_set = Phoenix2014TDataset(args.dataset_path, args.eval_on, transforms)
    train_loader = DataLoader(train_set, **loader_params)
    eval_loader = DataLoader(eval_set, **loader_params)

    # define model
    encoder = TranslationEncoder('transformer.wmt19.en-de.single_model')
    decoder = TranslationDecoder('transformer.wmt19.en-de.single_model')
    feature_extractor = FeatureExtractor(args.image_size, args.window_size, encoder.embedding_dim)

    model = Sign2Text(feature_extractor, encoder, decoder).to(device)

    # init scorers
    if args.scorers:
        scorers = Scorers(args.scorers, 0)
    else:
        scorers = None

    # define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    LOGGER.info(f'Using {args.optimizer} with learning_rate: {args.learning_rate}')

    criterion = nn.CrossEntropyLoss(ignore_index=decoder.vocab.pad_index)

    # init apex
    if args.enable_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    best_loss = float('inf')

    for epoch in range(args.epochs):

        LOGGER.info(f'------------ Epoch {epoch + 1} / {args.epochs} ------------')
        LOGGER.info('Training...')

        training_loss = train(model, train_loader, optimizer, criterion, args, device)

        LOGGER.info('Evaluation...')
        evaluation_loss, sentences = evaluate(model, eval_loader, criterion, args, device)

        if scorers:
            LOGGER.info('Computing metrics...')
            metric_scores = compute_metrics(scorers, sentences[0], sentences[1])
            for scorer_name, score in metric_scores.items():
                LOGGER.info(f'  {scorer_name.upper():>6} score: {score.corpus_score:6.2f}')
        else:
            metric_scores = None

        # save model if
        if evaluation_loss < best_loss:
            best_loss = evaluation_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'evaluation_loss': evaluation_loss,
                    'training_loss': training_loss,
                    'metric_scores': metric_scores,
                }, os.path.join(args.output_dir, 'model', 'rosita_' + str(epoch) + '.pth'))


if __name__ == "__main__":
    ARGS = utils.cli.parse_arguments()
    if ARGS.enable_apex and not APEX_AVAILABLE:
        ARGS.enable_apex = False
        LOGGER.warning("Apex is not installed! Install instructions on https://github.com/NVIDIA/apex . Continue not using mixed precision training.")

    utils.logger.init_logger(ARGS.output_dir)
    LOGGER.debug(ARGS)
    main(ARGS)
