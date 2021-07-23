import json
import os
import re

import pandas as pd

from data.phoenix2014Tdataset import Phoenix2014TDataset
from utils.onehotencoding import OneHotEncoding


def create_dir(output_dir: str) -> None:
    """Creating a new directory.

    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def filter_chars(input_str: str) -> str:
    """Filters german umlauts and special chars out of a string.

    Args:
        input_str (str): The string to be replaced.

    Returns:
        str: The replaced string.
    """
    x = input_str
    input_str = input_str.lower()

    #? replace with regex
    replace_words = [
        '.', ':', ',', ';', '-', '_', '#', '+', '*', '?', '=', ')', '(', '/', '&', '%', '$', '§', '"', '!', '<', '>', '°', '^', '„', '“', '|', '~', '{', '}',
        '[', ']', '\\', '@', '€'
    ]
    replace_words_space = ['\'', '´', '`']

    # replace german umlauts
    if 'ä' in input_str:
        input_str = input_str.replace('ä', 'ae')
    if 'ü' in input_str:
        input_str = input_str.replace('ü', 'ue')
    if 'ö' in input_str:
        input_str = input_str.replace('ö', 'oe')
    if 'ß' in input_str:
        input_str = input_str.replace('ß', 'ss')

    # replace special characters
    for re in replace_words:
        input_str = input_str.replace(re, '')
    for re in replace_words_space:
        # input_str = input_str.replace(re, '')
        input_str = input_str.replace(re, ' ')

    # remove double spaces
    if '  ' in input_str:
        input_str = input_str.replace('  ', ' ')
        input_str = input_str.strip()

    return input_str


def compare_datasets(output_path: str) -> None:
    """This method loads and compares the different vocabularies for the gebaerdenlernen, RWTH Phoenix and DGS Corpus datasets."""

    ### gebaerdenlernen
    dataset_dir = 'data/gebaerdenlernen/annotations/gebaerdenlernen.mp4.csv'

    gebaerdenlernen_df = pd.read_csv(dataset_dir, sep='|')

    gloss_words = []

    # filter glosses
    for gloss in gebaerdenlernen_df.loc[:, 'gloss'].tolist():
        gloss_words.extend(filter_chars(gloss).split(' '))

    # create one hot encoding for glosses
    gloss_one_hot_gebaerdenlernen = OneHotEncoding(gebaerdenlernen_df.loc[:, 'gloss'].tolist(), file_path=output_path + "gloss_one_hot_gebaerdenlernen")

    ### RWTH Phoenix
    training_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='train')
    dev_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='dev')
    test_set = Phoenix2014TDataset(root_dir='data/PHOENIX-2014-T-release-v3', split_type='test')

    gloss_words = []
    translation_words = []

    # add vocabular from each set and filter glosses and translations
    for idx in range(len(dev_set)):
        glosses, translations = dev_set.get_targets(idx)
        gloss_words.extend([filter_chars(gloss) for gloss in glosses])
        translation_words.extend([filter_chars(translation) for translation in translations])
    for idx in range(len(training_set)):
        glosses, translations = training_set.get_targets(idx)
        gloss_words.extend([filter_chars(gloss) for gloss in glosses])
        translation_words.extend([filter_chars(translation) for translation in translations])
    for idx in range(len(test_set)):
        glosses, translations = test_set.get_targets(idx)
        gloss_words.extend([filter_chars(gloss) for gloss in glosses])
        translation_words.extend([filter_chars(translation) for translation in translations])

    # create one hot encodings for glosses and translations
    gloss_one_hot_rwth = OneHotEncoding(gloss_words, file_path=output_path + "gloss_one_hot_rwth")
    translation_one_hot_rwth = OneHotEncoding(translation_words, file_path=output_path + "translation_one_hot_rwth")

    ### DGS Corpus
    with open('output/DGSCorpus/DGSCorpus.de.json', 'r') as dgs_file:
        dgs_dataset = json.load(dgs_file)

    gloss_words = []
    translation_words = []

    for item in dgs_dataset:
        translation = item['translation']['text'].lower()
        #! combine gloss and lexeme to one list
        gloss = ' '.join([gloss['text_simplified'].lower() for gloss in item['signs']])
        lexeme = ' '.join([gloss['text_simplified'].lower() for gloss in item['lexeme']])

        # filter translations, glosses and lexeme
        translation = filter_chars(translation)
        gloss = filter_chars(gloss)
        lexeme = filter_chars(lexeme)

        translation_words.extend(translation.split(' '))

        gloss_words.extend(gloss.split(' '))
        gloss_words.extend(lexeme.split(' '))

    # create one hot encodings for glosses and translations
    gloss_one_hot_dgs = OneHotEncoding(gloss_words, file_path=output_path + "gloss_one_hot_dgs")
    translation_one_hot_dgs = OneHotEncoding(translation_words, file_path=output_path + "translation_one_hot_dgs")

    # create output directories
    create_dir(output_path + 'gloss/intersection')
    create_dir(output_path + 'gloss/difference')
    create_dir(output_path + 'translation/intersection')
    create_dir(output_path + 'translation/difference')

    ## Evaluation
    print("Vocabular analysis:")
    print("---------------------------------")
    print(f"Gloss wordspace gebaerdenlernen: {len(gloss_one_hot_gebaerdenlernen)}")
    print(f"Gloss wordspace rwth: {len(gloss_one_hot_rwth)}")
    print(f"Gloss wordspace dgs: {len(gloss_one_hot_dgs)}")
    print("---------------------------------")
    print(f"Translation wordspace rwth: {len(translation_one_hot_rwth)}")
    print(f"Translation wordspace dgs: {len(translation_one_hot_dgs)}")
    print("---------------------------------")

    with open(output_path + 'gloss/intersection/gebaerdenlernen-rwth', 'w') as f:
        intersec = gloss_one_hot_gebaerdenlernen.word_space.intersection(gloss_one_hot_rwth.word_space)
        for item in intersec:
            f.write(f"{item}\n")
    print(f"Gloss intersection of gebaerdenlernen and rwth: {len(intersec)}")
    with open(output_path + 'gloss/intersection/rwth-dgs', 'w') as f:
        intersec = gloss_one_hot_rwth.word_space.intersection(gloss_one_hot_dgs.word_space)
        for item in intersec:
            f.write(f"{item}\n")
    print(f"Gloss intersection of rwth and dgs: {len(intersec)}")
    with open(output_path + 'gloss/intersection/dgs-gebaerdenlernen', 'w') as f:
        intersec = gloss_one_hot_dgs.word_space.intersection(gloss_one_hot_gebaerdenlernen.word_space)
        for item in intersec:
            f.write(f"{item}\n")
    print(f"Gloss intersection of dgs and gebaerdenlernen: {len(intersec)}")
    print("---------------------------------")

    with open(output_path + 'gloss/difference/gebaerden-rwth', 'w') as f:
        diff = gloss_one_hot_gebaerdenlernen.word_space.difference(gloss_one_hot_rwth.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of gebaerdenlernen and rwth: {len(diff)}")
    with open(output_path + 'gloss/difference/rwth-gebaerden', 'w') as f:
        diff = gloss_one_hot_rwth.word_space.difference(gloss_one_hot_gebaerdenlernen.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of rwth and gebaerdenlernen: {len(diff)}")
    with open(output_path + 'gloss/difference/rwth-dgs', 'w') as f:
        diff = gloss_one_hot_rwth.word_space.difference(gloss_one_hot_dgs.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of rwth and dgs: {len(diff)}")
    with open(output_path + 'gloss/difference/dgs-rwth', 'w') as f:
        diff = gloss_one_hot_dgs.word_space.difference(gloss_one_hot_rwth.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of dgs and rwth: {len(diff)}")
    with open(output_path + 'gloss/difference/dgs-gebaerdenlernen', 'w') as f:
        diff = gloss_one_hot_dgs.word_space.difference(gloss_one_hot_gebaerdenlernen.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of dgs and gebaerdenlernen: {len(diff)}")
    with open(output_path + 'gloss/difference/gebaerdenlernen-dgs', 'w') as f:
        diff = gloss_one_hot_gebaerdenlernen.word_space.difference(gloss_one_hot_dgs.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Gloss Difference of gebaerdenlernen and dgs: {len(diff)}")
    print("---------------------------------")

    with open(output_path + 'translation/intersection/rwth-dgs', 'w') as f:
        intersec = translation_one_hot_rwth.word_space.intersection(translation_one_hot_dgs.word_space)
        for item in intersec:
            f.write(f"{item}\n")
    print(f"Translation intersection of rwth and dgs: {len(intersec)}")
    print("---------------------------------")

    with open(output_path + 'translation/difference/rwth-dgs', 'w') as f:
        diff = translation_one_hot_rwth.word_space.difference(translation_one_hot_dgs.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Translation difference of rwth and dgs: {len(diff)}")
    with open(output_path + 'translation/difference/dgs-rwth', 'w') as f:
        diff = translation_one_hot_dgs.word_space.difference(translation_one_hot_rwth.word_space)
        for item in diff:
            f.write(f"{item}\n")
    print(f"Translation difference of dgs and rwth: {len(diff)}")
    print("---------------------------------")


if __name__ == "__main__":
    compare_datasets('output/compare_datasets/')

### TEST RESULTS ###

# Vocabular analysis:
# ---------------------------------
# Gloss wordspace gebaerdenlernen: 3387
# Gloss wordspace rwth: 1114
# Gloss wordspace dgs: 4612
# ---------------------------------
# Translation wordspace rwth: 3003
# Translation wordspace dgs: 22760
# ---------------------------------
# Gloss intersection of gebaerdenlernen and rwth: 501
# Gloss intersection of rwth and dgs: 627
# Gloss intersection of dgs and gebaerdenlernen: 1726
# ---------------------------------
# Gloss Difference of gebaerdenlernen and rwth: 2886
# Gloss Difference of rwth and gebaerdenlernen: 613
# Gloss Difference of rwth and dgs: 487
# Gloss Difference of dgs and rwth: 3985
# Gloss Difference of dgs and gebaerdenlernen: 2886
# Gloss Difference of gebaerdenlernen and dgs: 1661
# ---------------------------------
# Translation intersection of rwth and dgs: 1690
# ---------------------------------
# Translation difference of rwth and dgs: 1313
# Translation difference of dgs and rwth: 21070
# ---------------------------------
