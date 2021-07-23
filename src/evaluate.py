import argparse
import logging
import os
from test import evaluate

import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import Compose

from data.phoenix2014Tdataset import Phoenix2014TDataset
from data.transforms import Mean, Resize, ToTensor
from models.alexnet import load_alexnet
from models.model import AttnDecoderRNN, EncoderRNN
from utils.logger import create_logger
from utils.onehotencoding import OneHotEncoding


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='SignLanguageTranslation')

    # training hyperparameter
    arg_parser.add_argument('--hidden_size', type=int, default=1000, help='Hidden Size')

    # machine settings
    arg_parser.add_argument('--torch_seed', type=int, default=42, help='Manual Seed to ensure reproducibility')
    arg_parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the DataLoader')
    arg_parser.add_argument('--use_multi_gpu', action='store_true', help='Whether to use more than one GPU')

    # dataset settings
    arg_parser.add_argument('--image_size', type=int, default=227, help='Size of input images to resizing')
    arg_parser.add_argument('--mean_img_path', type=str, default='data/mean/FulFrame_Mean_Image_227x227.npy', help='Path to dataset mean image')
    # TODO: use dataset path
    arg_parser.add_argument('--dataset_path', type=str, default='data/PHOENIX-2014-T-release-v3', help='Path to the PHOENIX-2014-T-release-v3 dataset')
    arg_parser.add_argument('--flip_sequence', action='store_true', help='Flip input sequence for encoder')

    # model settings
    arg_parser.add_argument('--model_path', type=str, default='model', help='Path to the pre-trained encoder and decoder model')

    # encoding settings
    arg_parser.add_argument('--one_hot_encoding_path',
                            type=str,
                            default='data/one-hot-encoding/one-hot-encoding_word2index.txt',
                            help='Path to existing one-hot encoding')

    # evaluation settings
    arg_parser.add_argument('--eval_on', type=str, default='dev', help='Which dataset should be evaluated (dev|test)')

    # output settings
    arg_parser.add_argument('--output_dir', type=str, default='./output', help='Directory where to save the training output')

    return arg_parser.parse_args()


def main(args):
    # get logging instance
    logger = logging.getLogger('SignLanguageTranslation')

    # set manual seed for reproducibility
    torch.manual_seed(args.torch_seed)

    # PyTorch Settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### DATA ###

    # load mean np array
    mean_array = np.load(args.mean_img_path)

    # Image Transformations
    transforms = Compose([Resize(args.image_size), ToTensor(), Mean(mean_array)])

    # DataLoader Parameters
    params = {'num_workers': args.num_workers, 'pin_memory': True}

    # load existing one-hot encoding
    one_hot = OneHotEncoding(file_path=args.one_hot_encoding_path)

    ### MODEL ###

    # Define Image-Model
    alexnet = load_alexnet(remove_last_layer=True)
    alexnet.to(device)

    # define encoder and decoder input, hidden and output sizes
    enc_input_size = alexnet.classifier[-1].out_features  # get number of out features of last layer of classifier
    hidden_size = args.hidden_size
    output_size = len(one_hot.word_space)  # dimension of one-hot encoding

    # init encoder and decoder
    encoder = EncoderRNN(enc_input_size, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_size)

    # load saved model
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pth'), map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder.pth'), map_location=lambda storage, loc: storage))

    encoder.to(device)
    decoder.to(device)

    ### TEST ###

    eval_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type=args.eval_on, transform=transforms)
    eval_generator = data.DataLoader(eval_set, **params)

    logger.info(f"Start evaluation with following parameters: Eval on: {args.eval_on}\tNum_Workers: {args.num_workers}\tGPUs: {torch.cuda.device_count()}")

    evaluate(alexnet, encoder, decoder, eval_generator, one_hot, device, args.use_multi_gpu, args.flip_sequence)

    logger.info(f"Evaluation completed!")


if __name__ == "__main__":
    args = parse_arguments()
    create_logger(args.output_dir)
    main(args)
