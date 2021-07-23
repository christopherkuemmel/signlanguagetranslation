"""Contains the code for post-processing the DGSCorpus, splitting into subvideos and creating an annotation file."""
import ast
import gzip
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


def create_dir(output_dir: str) -> None:
    """Creating a new directory.

    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def trim_video(input_path: str, ouput_path: str, frame_range: tuple, fps: int) -> None:
    """Trims the given video by the frame range and saves it locally with ffmpeg.

    Args:
        input_path (str): The path to the video which should be trimmed.
        output_path (str): The path where the new video should be saved.
        frame_range (tuple): Tuple of start and end frame (start_frame, end_frame). The values are absolutes of the input video.
        fps (int): The fps of the input video.
    """
    # ffmpeg needs timestamps for start as input rather than frames
    start_timestamp = min(frame_range) / fps
    # relative frame range between start and end frame
    frame_range = max(frame_range) - min(frame_range)
    process = 'ffmpeg -ss ' + str(start_timestamp) + ' -i ' + input_path + ' -c:v libx264 -c:a aac -frames:v ' + str(frame_range) + ' ' + ouput_path
    completed_process = subprocess.run(process, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    if completed_process.returncode == 1:
        print(f"Failed to save video: {ouput_path}")


def save_video(video_path: str, video: np.array, fps: int) -> None:
    """Saves the given video in the path as .mp4 file.

    Args:
        video_path (str): The path to save the video including filename (with ending).
        video (np.ndarray): The video to save.
        fps (int): How many fps the encoded video should have.
    """
    # get video dimensions
    heigth, width, _ = video[0].shape

    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, heigth), True)
    for frame in video:
        writer.write(frame.astype('uint8'))
    writer.release()


def load_video(video_path: str) -> np.array:
    """Loads and returns the video as np.ndarray.

    Args:
        video_path (str): The path to load the video.

    Returns:
        np.ndarray: The loaded video.
    """
    video = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
    return np.array(video)


def load_eaf_file(path: str) -> dict:
    """Loads a eaf file by its path and returns it as dictionary.

    Args:
        path (str): The path to the eaf file.

    Returns:
        dict: A dictionary with the keys - timestamps, translations, signs, lexeme, mouths
    """
    # eaf(xml) processing
    tree = ET.parse(path)
    root = tree.getroot()

    timestamps = []
    translations = []
    signs = []
    lexeme = []
    mouths = []

    for elem in root:
        if elem.tag == 'TIME_ORDER':
            for time_slot in elem:
                timestamps.append(int(time_slot.attrib['TIME_VALUE']))

        if elem.tag == 'TIER':

            video_angle = ''
            translation_type = ''
            language = ''
            side = ''

            if elem.attrib['TIER_ID'] == 'Deutsche_Übersetzung_A':
                video_angle, translation_type, language = '1a1', 'TRANSLATION', 'DE'
            if elem.attrib['TIER_ID'] == 'Englische_Übersetzung_A':
                video_angle, translation_type, language = '1a1', 'TRANSLATION', 'EN'
            if elem.attrib['TIER_ID'] == 'Lexem_Gebärde_r_A':
                video_angle, translation_type, language, side = '1a1', 'LEXEM', 'DE', 'right'
            if elem.attrib['TIER_ID'] == 'Lexem_Gebärde_l_A':
                video_angle, translation_type, language, side = '1a1', 'LEXEM', 'DE', 'left'
            if elem.attrib['TIER_ID'] == 'Lexeme_Sign_r_A':
                video_angle, translation_type, language, side = '1a1', 'LEXEM', 'EN', 'right'
            if elem.attrib['TIER_ID'] == 'Lexeme_Sign_l_A':
                video_angle, translation_type, language, side = '1a1', 'LEXEM', 'EN', 'left'
            if elem.attrib['TIER_ID'] == 'Gebärde_r_A':
                video_angle, translation_type, language, side = '1a1', 'GLOSS', 'DE', 'right'
            if elem.attrib['TIER_ID'] == 'Gebärde_l_A':
                video_angle, translation_type, language, side = '1a1', 'GLOSS', 'DE', 'left'
            if elem.attrib['TIER_ID'] == 'Sign_r_A':
                video_angle, translation_type, language, side = '1a1', 'GLOSS', 'EN', 'right'
            if elem.attrib['TIER_ID'] == 'Sign_l_A':
                video_angle, translation_type, language, side = '1a1', 'GLOSS', 'EN', 'left'
            if elem.attrib['TIER_ID'] == 'Mundbild_Mundgestik_A':
                video_angle, translation_type, language = '1a1', 'MOUTH', 'DE'

            if elem.attrib['TIER_ID'] == 'Deutsche_Übersetzung_B':
                video_angle, translation_type, language = '1b1', 'TRANSLATION', 'DE'
            if elem.attrib['TIER_ID'] == 'Englische_Übersetzung_B':
                video_angle, translation_type, language = '1b1', 'TRANSLATION', 'EN'
            if elem.attrib['TIER_ID'] == 'Lexem_Gebärde_r_B':
                video_angle, translation_type, language, side = '1b1', 'LEXEM', 'DE', 'right'
            if elem.attrib['TIER_ID'] == 'Lexem_Gebärde_l_B':
                video_angle, translation_type, language, side = '1b1', 'LEXEM', 'DE', 'left'
            if elem.attrib['TIER_ID'] == 'Lexeme_Sign_r_B':
                video_angle, translation_type, language, side = '1b1', 'LEXEM', 'EN', 'right'
            if elem.attrib['TIER_ID'] == 'Lexeme_Sign_l_B':
                video_angle, translation_type, language, side = '1b1', 'LEXEM', 'EN', 'left'
            if elem.attrib['TIER_ID'] == 'Gebärde_r_B':
                video_angle, translation_type, language, side = '1b1', 'GLOSS', 'DE', 'right'
            if elem.attrib['TIER_ID'] == 'Gebärde_l_B':
                video_angle, translation_type, language, side = '1b1', 'GLOSS', 'DE', 'left'
            if elem.attrib['TIER_ID'] == 'Sign_r_B':
                video_angle, translation_type, language, side = '1b1', 'GLOSS', 'EN', 'right'
            if elem.attrib['TIER_ID'] == 'Sign_l_B':
                video_angle, translation_type, language, side = '1b1', 'GLOSS', 'EN', 'left'
            if elem.attrib['TIER_ID'] == 'Mundbild_Mundgestik_B':
                video_angle, translation_type, language = '1b1', 'MOUTH', 'DE'

            #! skip moderator video?!
            # if elem.attrib['TIER_ID' == 'Moderator']:
            # if elem.attrib['TIER_ID' == 'Englische_Übersetzung_Mod']:

            for annotation in elem:
                start_time_ref, end_time_ref = int(annotation[0].attrib['TIME_SLOT_REF1'].replace('ts', '')) - 1, \
                                            int(annotation[0].attrib['TIME_SLOT_REF2'].replace('ts', '')) - 1
                text = annotation[0][0].text

                if side != '':
                    item = (video_angle, start_time_ref, end_time_ref, language, text, side)
                else:
                    item = (video_angle, start_time_ref, end_time_ref, language, text)

                if translation_type == 'TRANSLATION':
                    translations.append(item)
                if translation_type == 'LEXEM':
                    lexeme.append(item)
                if translation_type == 'GLOSS':
                    signs.append(item)
                if translation_type == 'MOUTH':
                    mouths.append(item)

    return {"timestamps": timestamps, "translations": translations, "signs": signs, "lexeme": lexeme, "mouths": mouths}


def create_files(video_df: pd.Series, video_path: str, openpose: dict, timestamps: list, translations: list, signs: list, lexeme: list, mouths: list,
                 language: str, annotations: list, output_path: str) -> list:
    """Creates the annotation files, subvideo and subopenpose files.

    Args:
        video_df (pandas.core.series.Series): The Series (out of DataFrame) of the single video.
        video_path (str): The path to the full video.
        openpose (dict): The corresponding openpose information.
        timestamps (list): List of timestamps given in microseconds.
        translation (list): List of translated sentences with timeboundaries.
        signs (list): List of signs corresponding to the translation with timeboundaries.
        lexeme (list): List of lexeme corresponding to the translation with timeboundaries.
        mouths (list): List of mouth gestures corresponding to the translation with timeboundaries.
        language (str): (EN|DE) The language of the translation.
        annotations (list): The list of all annotated videos.
        output_path (str): The path to save all files.
    """
    language = language.lower()

    # german and english translations do not neccessarly have the same frametimes/ timesteps
    for subvideo_idx, translation in enumerate(translations):
        start_idx = translation[1]
        end_idx = translation[2]

        filtered_timestamps = timestamps[start_idx:end_idx + 1]

        range_filter = lambda x: x[1] >= start_idx and x[2] <= end_idx
        filtered_signs = list(filter(range_filter, signs))
        filtered_lexeme = list(filter(range_filter, lexeme))
        filtered_mouths = list(filter(range_filter, mouths))

        # english filter
        range_filter = lambda x: x[1] >= start_idx and x[2] <= end_idx and x[3] == 'ENGLISH'

        # create new output dir for each video
        video_name = str(video_df['dgs_id']) + "_" + translation[0] + "_" + str(subvideo_idx)
        output_dir = os.path.join(output_path, language, video_name)
        create_dir(output_dir)

        re_pattern = '[a-zA-Z\-öäüÖÄÜ]+'

        # create json object for each video
        annotation = {
            "name":
            video_name,
            "dgs_id":
            video_df['dgs_id'],
            "transcript":
            video_df['transcript'],
            "age_group":
            video_df['age_group'],
            "format":
            video_df['format'],
            "topics":
            ast.literal_eval(video_df['topics']),
            "timestamps": [timestamp - filtered_timestamps[0] for timestamp in filtered_timestamps],
            "translation": {
                "text": translation[-1],
                "start_ref": translation[1] - start_idx,
                "end_ref": translation[2] - start_idx
            },
            "signs": [{
                "text": sign[4],
                "text_simplified": re.search(re_pattern, sign[4]).group(0),
                "hand": sign[5],
                "start_ref": sign[1] - start_idx,
                "end_ref": sign[2] - start_idx
            } for sign in filtered_signs],
            "lexeme": [{
                "text": lexem[4],
                "text_simplified": re.search(re_pattern, lexem[4]).group(0),
                "hand": lexem[5],
                "start_ref": lexem[1] - start_idx,
                "end_ref": lexem[2] - start_idx
            } for lexem in filtered_lexeme],
            "mouth": [{
                "text": mouth[4],
                "start_ref": mouth[1] - start_idx,
                "end_ref": mouth[2] - start_idx
            } for mouth in filtered_mouths],
        }

        # append video annotation to all annotations
        annotations.append(annotation)

        # write all annotations to file
        with open(os.path.join(output_path, 'DGSCorpus.' + language + '.json'), 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(annotations, ensure_ascii=False))

        # create subvideo from full video
        # compute frame duration in ms
        fps = 50
        frame_duration = int(1 / fps * 1000)

        start_frame = filtered_timestamps[0] // frame_duration
        end_frame = filtered_timestamps[-1] // frame_duration

        # save sub video
        trim_video(video_path, os.path.join(output_dir, video_name + '.mp4'), (start_frame, end_frame), fps)

        # create sub-file from openpose
        openpose_json = {
            "id": video_name,
            "camera": openpose['camera'],
            "width": openpose['width'],
            "height": openpose['height'],
        }
        openpose_json['frames'] = [openpose['frames'][str(key)] for key in range(start_frame, end_frame)]
        with open(os.path.join(output_dir, video_name + '_openpose.json'), 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(openpose_json))

    return annotations


def split_videos_and_create_annotations(dgs_corpus_de_csv_path: str, dgs_corpus_en_csv_path: str, output_path: str) -> list:
    """Splits the DGSCorpus videos to subvideos, saves them and creates a annotation.json all based on the references.csv from the download/scraper script.

    Args:
        dgs_corpus_de_csv_path (str): The path to the german references.csv file (from the scraper script).
        dgs_corpus_en_csv_path (str): The path to the english references.csv file (from the scraper script).
        output_path (str): The path to store the newly created videos and annotation file.

    Returns:
        list: The list (json) with the splitted videos (pathes) and annotations.
    """
    dgs_corpus_de_df = pd.read_csv(dgs_corpus_de_csv_path, sep=';')
    dgs_corpus_en_df = pd.read_csv(dgs_corpus_en_csv_path, sep=';')

    annotations_de = []
    annotations_en = []

    for video_idx in range(len(dgs_corpus_de_df)):

        start_time = time.time()
        print(f"Start Processing video {video_idx + 1}/{len(dgs_corpus_de_df)}")

        video_df = dgs_corpus_de_df.iloc[video_idx]
        eaf_path = os.path.join('data/DGSCorpus/features', video_df['dgs_id'], video_df['dgs_id'] + '.eaf')

        # skip video if there is no eaf file - this is true for all joke videos
        if not os.path.exists(eaf_path):
            print(f"No translation for video {video_idx + 1}. Continue with next video.")
            continue

        eaf_dict = load_eaf_file(eaf_path)

        # load full openpose file
        openpose_path = os.path.join('data/DGSCorpus/features', video_df['dgs_id'], video_df['dgs_id'] + '_openpose.json.gz')
        with gzip.GzipFile(openpose_path, 'r') as fin:
            json_bytes = fin.read()
        full_openpose = json.loads(json_bytes.decode('utf-8'))

        # filter language and video angle and create files seperatly
        for language in ['DE', 'EN']:

            for video_angle in ['1a1', '1b1', '1c']:
                range_filter = lambda x: x[3] == language and x[0] == video_angle
                filtered_translations = list(filter(range_filter, eaf_dict["translations"]))

                if filtered_translations:
                    filtered_signs = list(filter(range_filter, eaf_dict["signs"]))
                    filtered_lexeme = list(filter(range_filter, eaf_dict["lexeme"]))
                    filtered_mouths = list(filter(range_filter, eaf_dict["mouths"]))

                    # input video name
                    video_path = os.path.join('data/DGSCorpus/features', video_df['dgs_id'], video_df['dgs_id'] + '_' + video_angle + '.mp4')

                    # select only the correct openpose features corresponding to the video angle (a, b or c)
                    openpose_video = dict(list(filter(lambda x: x['camera'] in video_angle, full_openpose))[0])

                    # create files
                    if language == 'EN':
                        video_df = dgs_corpus_en_df.iloc[video_idx]
                        create_files(video_df, video_path, openpose_video, eaf_dict["timestamps"], filtered_translations, filtered_signs, filtered_lexeme,
                                     filtered_mouths, language, annotations_en, output_path)
                    else:
                        video_df = dgs_corpus_de_df.iloc[video_idx]
                        create_files(video_df, video_path, openpose_video, eaf_dict["timestamps"], filtered_translations, filtered_signs, filtered_lexeme,
                                     filtered_mouths, language, annotations_de, output_path)
        print(f"Loading video {video_idx + 1} took {time.time() - start_time}")
    print("DONE")
    return annotations_de


def sanity_check_annotations(dgs_corpus_csv_path: str, annotation_path: str) -> None:
    """Sanity check whether all sentences are splitted correctly.

    Args:
        dgs_corpus_de_csv_path (str): The path to the german references.csv file (from the scraper script).
        dgs_corpus_en_csv_path (str): The path to the english references.csv file (from the scraper script).
        annotation_path (str):
    """
    dgs_corpus_df = pd.read_csv(dgs_corpus_csv_path, sep=';')
    with open(annotation_path, 'r') as annotation_file:
        annotation_dict = json.load(annotation_file)

    not_found = []
    for video_idx in range(len(dgs_corpus_df)):

        video_df = dgs_corpus_df.iloc[video_idx]
        eaf_path = os.path.join('data/DGSCorpus/features', video_df['dgs_id'], video_df['dgs_id'] + '.eaf')

        # skip video if there is no eaf file - this is true for all joke videos
        if not os.path.exists(eaf_path):
            print(f"No translation for video {video_idx + 1}. Continue with next video.")
            continue

        eaf_dict = load_eaf_file(eaf_path)

        filter_language = lambda x: x[3] == annotation_path.split('.')[2].upper()  #! get language from file - only works if file ends with .de.json || .en.json
        for translation in list(filter(filter_language, eaf_dict['translations'])):
            # find translation from eaf file in annotations file
            filtered = list(filter(lambda x: x['translation']['text'] == translation[4], annotation_dict))

            if not filtered:
                not_found.append(f"Translation: {translation[4]} from video: {dgs_corpus_df.iloc[video_idx]['dgs_id']} missing.")
            # print(f"EAF: {translation[4]} - DICT: {filtered[0]['translation']['text']}")

        print(f"Processed video {video_idx + 1}/{len(dgs_corpus_df) + 1} - not_found count: {len(not_found)}")

    if not_found:
        print(f"Video chunking failed. Missing {len(not_found)} items.")
        with open("not_found.txt", "w") as not_found_file:
            not_found_file.writelines(not_found)
    else:
        print("Chunking was successful! Sanity check run without problems.")


if __name__ == "__main__":
    # split_videos_and_create_annotations('data/DGSCorpus/references_de.csv', 'data/DGSCorpus/references_en.csv', 'output/DGSCorpus')
    sanity_check_annotations('data/DGSCorpus/references_de.csv', './output/DGSCorpus-done/DGSCorpus.de.json')
    sanity_check_annotations('data/DGSCorpus/references_en.csv', './output/DGSCorpus-done/DGSCorpus.en.json')
