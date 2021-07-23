"""Contains the main class for the Neural Sign Language Translation (NSLT) project."""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision.transforms import Compose

from data.phoenix2014Tdataset import Pad, Phoenix2014TDataset
from data.transforms import Mean, Resize, ToTensor
from models.alexnet import load_alexnet
from models.model import AttnDecoderRNN, EncoderRNN
from models.resnext_101_kinetics import load_resnext101_3d_kinetics
from test import evaluate
from train import backward, forward
from utils.logger import create_logger
from utils.onehotencoding import OneHotEncoding
from utils.plot import save_plot
from utils.time import startTime, timeSince


def parse_arguments():
    """This method parses the given command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    arg_parser = argparse.ArgumentParser(description='SignLanguageTranslation')

    # print & plot settings
    arg_parser.add_argument('--print_every', type=int, default=1, help='Print loss every n batchs')
    arg_parser.add_argument('--plot_every', type=int, default=100, help='Plot loss every n batchs')

    # training hyperparameter
    arg_parser.add_argument('--batch_size', type=int, default=1, help='Mini-Batch size')
    arg_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    arg_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning Rate')
    arg_parser.add_argument('--hidden_size', type=int, default=1000, help='Hidden Size')
    arg_parser.add_argument('--window_size', type=int, default=16, help='Feature Extractor Window Size')
    arg_parser.add_argument('--clip_gradients', type=int, default=5, help='Gradient clipping value')
    arg_parser.add_argument('--ignore_pad_idx', action='store_true', help='Whether to ignore padding token at loss computation')

    # model settings
    arg_parser.add_argument('--feature_extractor', type=str, default='alexnet', help='Which feature extractor to use (alexnet or resnext101)')

    # machine settings
    arg_parser.add_argument('--torch_seed', type=int, default=42, help='Manual Seed to ensure reproducibility')
    arg_parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the DataLoader')
    arg_parser.add_argument('--use_multi_gpu', action='store_true', help='Whether to use more than one GPU')

    # dataset settings
    arg_parser.add_argument('--image_size', type=int, default=227, help='Size of input images to resizing')
    arg_parser.add_argument('--mean_img_path', type=str, default='data/mean/FulFrame_Mean_Image_227x227.npy', help='Path to dataset mean image')
    arg_parser.add_argument('--not_shuffle', action='store_false', help='Do not shuffle Dataset')
    arg_parser.add_argument('--dataset_path', type=str, default='data/PHOENIX-2014-T-release-v3', help='Path to the PHOENIX-2014-T-release-v3 dataset')
    arg_parser.add_argument('--max_frame_count', type=int, default=475, help='Maximum sequence length for input frames')
    arg_parser.add_argument('--flip_sequence', action='store_true', help='Flip input sequence for encoder')

    # encoding settings
    arg_parser.add_argument('--one_hot_encoding', action='store_true', help='Compute new one-hot encoding based on the dataset')
    arg_parser.add_argument('--one_hot_encoding_path',
                            type=str,
                            default='data/one-hot-encoding/one-hot-encoding_word2index.txt',
                            help='Path to existing one-hot encoding')

    # evaluation settings
    arg_parser.add_argument('--eval_on', type=str, default='dev', help='Which dataset should be evaluated (dev|test)')
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help='Mini-Batch size')

    # output settings
    arg_parser.add_argument('--output_dir', type=str, default='./output', help='Directory where to save the training output')

    return arg_parser.parse_args()


def main(args):
    """The main method of the nslt training."""

    # get logging instance
    logger = logging.getLogger('SignLanguageTranslation')

    logger.debug(f"Arguments: {args}")

    # set manual seed for reproducibility
    torch.manual_seed(args.torch_seed)

    # PyTorch Settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### SAVING ###
    model_output_dir = os.path.join(args.output_dir, 'model')

    if not os.path.exists(model_output_dir):
        try:
            os.makedirs(model_output_dir)
        except:
            logger.error("Error while try to create directory for saving torch model.")
            raise OSError

    ### DATA ###

    # load mean np array
    mean_array = np.load(args.mean_img_path)

    # Image Transformations
    if args.batch_size == 1:
        transforms = Compose([Resize(args.image_size), ToTensor(), Mean(mean_array)])
    else:
        #! Mean must come before padding. Otherwise the mean will be subtracted from the zero arrays -> not "padded" anymore
        transforms = Compose([Resize(args.image_size), ToTensor(), Mean(mean_array),
                              Pad(frame_count=args.max_frame_count)])  # if batch_size > 1 we need to add padding

    # DataLoader Parameters
    params = {'batch_size': args.batch_size, 'shuffle': args.not_shuffle, 'num_workers': args.num_workers, 'drop_last': True, 'pin_memory': True}

    if args.one_hot_encoding:
        training_set = Phoenix2014TDataset(root_dir=args.dataset_path, split_type='train', transform=transforms)
        dev_set = Phoenix2014TDataset(root_dir=args.dataset_path, split_type='dev', transform=transforms)
        test_set = Phoenix2014TDataset(root_dir=args.dataset_path, split_type='test', transform=transforms)

        ### WORD EMBEDDINGS ###
        # iterate over all samples and create list of glosses and translation words
        gloss_words = []
        translation_words = []
        for idx in range(len(dev_set)):
            glosses, translation = dev_set.get_targets(idx)
            gloss_words.extend(glosses)
            translation_words.extend(translation)
        for idx in range(len(training_set)):
            glosses, translation = training_set.get_targets(idx)
            gloss_words.extend(glosses)
            translation_words.extend(translation)
        for idx in range(len(test_set)):
            glosses, translation = test_set.get_targets(idx)
            gloss_words.extend(glosses)
            translation_words.extend(translation)

        # create one hot encodings for glosses and translations
        one_hot = OneHotEncoding(translation_words, file_path=args.output_dir)

        # clear variables to save memory
        dev_set = None
        test_set = None

    else:
        # load existing one-hot encoding
        one_hot = OneHotEncoding(file_path=args.one_hot_encoding_path)

    training_set = Phoenix2014TDataset(root_dir=args.dataset_path, split_type='train', transform=transforms)
    training_generator = data.DataLoader(training_set, **params)

    ### MODEL ###

    # Define feature extractor model
    logger.debug(f'Using model: {args.feature_extractor.lower()}')
    if args.feature_extractor.lower() == 'alexnet':
        feature_extractor = load_alexnet(remove_last_layer=True)
        enc_input_size = feature_extractor.classifier[-1].out_features  # get number of out features of last layer of classifier
    else:  # elif args.feature_extractor.lower() == 'resnext101'
        feature_extractor = load_resnext101_3d_kinetics(args.image_size, args.window_size, remove_last_layer=True)
        # TODO: remove magic number!
        enc_input_size = 2048
    feature_extractor.to(device)

    # define encoder and decoder input, hidden and output sizes
    hidden_size = args.hidden_size
    output_size = len(one_hot.word_space)  # dimension of one-hot encoding

    # init encoder and decoder
    encoder = EncoderRNN(enc_input_size, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_size)

    if torch.cuda.device_count() > 1 and args.use_multi_gpu:
        logger.debug(f"Multi GPU Count: {torch.cuda.device_count()}")
        encoder = nn.DataParallel(encoder, dim=1)  # dim=1 for batch_size
        decoder = nn.DataParallel(decoder, dim=1)

    encoder.to(device)
    decoder.to(device)

    # optimizers
    if torch.cuda.device_count() > 1 and args.use_multi_gpu:
        encoder_optimizer = optim.Adam(encoder.module.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.module.parameters(), lr=args.learning_rate)
    else:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # criterion
    if args.ignore_pad_idx:
        criterion = nn.CrossEntropyLoss(ignore_index=one_hot.word2index['<UNK>'], reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')

    start = startTime()
    best_epoch_loss = float('inf')
    epoch_batch_count = len(training_generator)
    total_batch_count = epoch_batch_count * args.epochs
    plot_losses = []

    ### TRAINING ###

    if args.epochs != 0 and args.batch_size != 0:
        logger.info(
            f"Start training with following parameters: Epochs: {args.epochs}\t Batch_Size: {args.batch_size}\t Learning_Rate: {args.learning_rate}\tNum_Workers: {args.num_workers}\tGPUs: {torch.cuda.device_count()}"
        )

    # Loop over epochs
    for epoch in range(args.epochs):

        epoch_loss = 0

        # Training
        # for i_batch, sample_batched in enumerate(dev_generator):
        for i_batch, sample_batched in enumerate(training_generator):

            # set gradients for optimizer to zero
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            if args.flip_sequence:
                # TODO: still correct dim, after 5-dim update?
                sample_batched['video'] = torch.flip(sample_batched['video'], dims=[1])

            # forward and loss
            loss_tensor, loss = forward(sample_batched, loss, args.batch_size, feature_extractor, encoder, decoder, criterion, one_hot, device,
                                        args.use_multi_gpu)
            # backprop and update
            backward(loss_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, args.clip_gradients, len(sample_batched['translation']))
            epoch_loss += loss

            current_batch = epoch * epoch_batch_count + (i_batch + 1)
            current_percentage = current_batch / total_batch_count

            if current_batch % args.print_every == 0:
                logger.info(
                    f"Epoch: {epoch+1}/{args.epochs}\tBatch: {current_batch}/{total_batch_count}\tMini-Batch: {i_batch+1}/{epoch_batch_count}\tLoss: {loss:.4f}\t{timeSince(start, current_percentage)}"
                )
            else:
                logger.debug(
                    f"Epoch: {epoch+1}/{args.epochs}\tBatch: {current_batch}/{total_batch_count}\tMini-Batch: {i_batch+1}/{epoch_batch_count}\tLoss: {loss:.4f}\t{timeSince(start, current_percentage)}"
                )

            ### PLOT LOSS ###
            if current_batch % args.plot_every == 0:
                plot_losses.append(loss)
                save_plot(plot_losses, output_path=args.output_dir)

        epoch_loss = epoch_loss / epoch_batch_count
        logger.info(f"Epoch: {epoch+1}\tLoss: {epoch_loss}")

        ### SAVING ###
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            logger.info(f"Saving new better model")

            if torch.cuda.device_count() > 1 and args.use_multi_gpu:
                torch.save(encoder.module.state_dict(), os.path.join(model_output_dir, 'encoder.pth'))
                torch.save(decoder.module.state_dict(), os.path.join(model_output_dir, 'decoder.pth'))
            else:
                torch.save(encoder.state_dict(), os.path.join(model_output_dir, 'encoder.pth'))
                torch.save(decoder.state_dict(), os.path.join(model_output_dir, 'decoder.pth'))

    # ### TEST ###

    # # DataLoader Parameters
    # params = {'batch_size': args.eval_batch_size, 'num_workers': args.num_workers, 'pin_memory': True}

    # if args.eval_batch_size == 1:
    #     transforms = Compose([Resize(args.image_size), ToTensor(), Mean(mean_array)])
    # else:
    #     #! Mean must come before padding. Otherwise the mean will be subtracted from the zero arrays -> not "padded" anymore
    #     transforms = Compose([Resize(args.image_size), ToTensor(), Mean(mean_array),
    #                           Pad(frame_count=args.max_frame_count)])  # if batch_size > 1 we need to add padding

    # eval_set = Phoenix2014TDataset(root_dir=args.dataset_path, split_type=args.eval_on, transform=transforms)
    # eval_generator = data.DataLoader(eval_set, **params)

    # logger.info(f"Start evaluation with following parameters: Eval on: {args.eval_on}\tNum_Workers: {args.num_workers}\tGPUs: {torch.cuda.device_count()}")

    # evaluate(feature_extractor, encoder, decoder, eval_generator, one_hot, device, 5, args.use_multi_gpu, args.flip_sequence)

    logger.info(f"Training completed!")


if __name__ == "__main__":
    ARGUMENTS = parse_arguments()
    create_logger(ARGUMENTS.output_dir)
    main(ARGUMENTS)
