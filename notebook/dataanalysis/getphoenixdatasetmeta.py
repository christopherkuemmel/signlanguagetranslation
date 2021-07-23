#%%
# Load dataset meta information from csv and file count

import os, os.path
import pandas as pd

train_base_dir = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
dev_base_dir = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
test_base_dir = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'

train = pd.read_csv(train_base_dir, sep='|')
dev = pd.read_csv(dev_base_dir, sep='|')
test = pd.read_csv(test_base_dir, sep='|')

max_words = 0
max_gloss = 0
max_frames = 0

for i in range(len(train)):
    if len(train.loc[i, 'translation'].split()) > max_words:
        max_words = len(train.loc[i, 'translation'].split())
    if len(train.loc[i, 'orth'].split()) > max_gloss:
        max_gloss = len(train.loc[i, 'orth'].split())

    _dir = os.path.join('data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train', train.loc[i, 'name'])
    if len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))]) > max_frames:
        max_frames = len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))])

print(f"train-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")

max_words = 0
max_gloss = 0
max_frames = 0

for i in range(len(dev)):
    if len(dev.loc[i, 'translation'].split()) > max_words:
        max_words = len(dev.loc[i, 'translation'].split())
    if len(dev.loc[i, 'orth'].split()) > max_gloss:
        max_gloss = len(dev.loc[i, 'orth'].split())

    _dir = os.path.join('data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev', dev.loc[i, 'name'])
    if len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))]) > max_frames:
        max_frames = len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))])

print(f"dev-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")

max_words = 0
max_gloss = 0
max_frames = 0

for i in range(len(test)):
    if len(test.loc[i, 'translation'].split()) > max_words:
        max_words = len(test.loc[i, 'translation'].split())
    if len(test.loc[i, 'orth'].split()) > max_gloss:
        max_gloss = len(test.loc[i, 'orth'].split())

    _dir = os.path.join('data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test', test.loc[i, 'name'])
    if len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))]) > max_frames:
        max_frames = len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))])

print(f"test-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")
    

### OUTPUT

# train-set max words: 52, max glosses: 30, max frames: 475
# dev-set max words: 32, max glosses: 18, max frames: 272
# test-set max words: 30, max glosses: 18, max frames: 242



#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

#  Load dataset meta information from dataset

# from src.data.phoenix2014Tdataset import Phoenix2014TDataset, ToTensor
# from torchvision.transforms import Compose

# # Image Transformations
# transforms = Compose([ToTensor()])

# # define dataset
# dev_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='dev', transform=transforms)
# training_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='train', transform=transforms)
# validation_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='test', transform=transforms)

# max_words = 0
# max_gloss = 0
# max_frames = 0

# for i,x in enumerate(training_set):
#     if x['video'].size(0) > max_frames:
#         max_frames = x['video'].size(0)
#     if len(x['glosses']) > max_gloss:
#         max_gloss = len(x['glosses'])
#     if len(x['translation']) > max_words:
#         max_words = len(x['translation'])
#     if i % 10 == 0:
#         print(f"Train-Processed: {i}/{len(training_set)}")

# print(f"train-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")

# max_words = 0
# max_gloss = 0
# max_frames = 0

# for x in dev_set:
#     if x['video'].size(0) > max_frames:
#         max_frames = x['video'].size(0)
#     if len(x['glosses']) > max_gloss:
#         max_gloss = len(x['glosses'])
#     if len(x['translation']) > max_words:
#         max_words = len(x['translation'])
#     if i % 10 == 0:
#         print(f"Dev-Processed: {i}/{len(training_set)}")

# print(f"dev-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")

# max_words = 0
# max_gloss = 0
# max_frames = 0

# for x in validation_set:
#     if x['video'].size(0) > max_frames:
#         max_frames = x['video'].size(0)
#     if len(x['glosses']) > max_gloss:
#         max_gloss = len(x['glosses'])
#     if len(x['translation']) > max_words:
#         max_words = len(x['translation'])
#     if i % 10 == 0:
#         print(f"Test-Processed: {i}/{len(training_set)}")

# print(f"test-set max words: {max_words}, max glosses: {max_gloss}, max frames: {max_frames}")

### OUTPUT ###
# train-set max words: 52, max glosses: 30, max frames: 475
# dev-set max words: 32, max glosses: 18, max frames: 272
# test-set max words: 30, max glosses: 18, max frames: 242
