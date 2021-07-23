"""Contains the code for post-processing the gebaerdenlernen meta data, creating annotation files and save as numpy arrays."""
import os
from typing import Union

import cv2
import numpy as np
import pandas as pd


def create_annotations_csv(gebaerdenlernen_csv_path: str) -> pd.DataFrame:
    """Create a new annotation csv file based on the meta.csv from the download/scraper script.

    Args:
        gebaerdenlernen_csv_path (str): The path to the meta.csv file (from the scraper script).

    Returns:
        pandas.DataFrame: The newly created annotation DataFrame.
    """
    gebaerdenlernen_df = pd.read_csv(gebaerdenlernen_csv_path, sep='|')

    hints = []
    synonyms = []
    for idx, gloss in enumerate(gebaerdenlernen_df.loc[:, 'gloss'].tolist()):
        x = gloss.lower()

        # filter german special chars
        if 'ä' in gloss.lower():
            x = x.replace('ä', 'ae')
        if 'ü' in gloss.lower():
            x = x.replace('ü', 'ue')
        if 'ö' in gloss.lower():
            x = x.replace('ö', 'oe')
        if 'ß' in gloss.lower():
            x = x.replace('ß', 'ss')

        # split gloss hint from translation and add it to another list
        if '(' in gloss.lower() and ')' in gloss.lower():
            start = x.index('(')
            end = x.index(')')
            hints.append(x[start + 1:end])  # start + 1 removes the '(', end will exclude ')'
            x = x[:start - 1]
        else:
            hints.append('None')

        # if there are two or more descriptions (synonyms) for one gloss split them and create a new entry for the same video
        if '/' in gloss.lower():
            glosses = x.split('/')
            x = glosses[0]
            for _idx in range(1, len(glosses)):
                gloss_df = gebaerdenlernen_df.loc[idx].copy()
                gloss_df['gloss'] = glosses[_idx]
                gloss_df['hint'] = hints[idx]
                synonyms.append(gloss_df.tolist())

        gebaerdenlernen_df.loc[idx, 'gloss'] = x

    gebaerdenlernen_df['hint'] = hints

    # create df based on synonyms and concat them to the base df
    synonyms = pd.DataFrame(synonyms, columns=['gloss', 'url', 'downloaded', 'video', 'hint'])
    gebaerdenlernen_df = pd.concat([gebaerdenlernen_df, synonyms])

    gebaerdenlernen_df = gebaerdenlernen_df.drop('downloaded', axis=1)

    column_order = ['gloss', 'hint', 'video', 'url']
    gebaerdenlernen_df = gebaerdenlernen_df.reindex(columns=column_order)

    gebaerdenlernen_df = gebaerdenlernen_df.sort_values(by=['gloss'])

    gebaerdenlernen_df.to_csv("data/gebaerdenlernen/gebaerdenlernen.csv", index=False, sep='|')
    return gebaerdenlernen_df


def create_numpy_arrays(references: Union[str, pd.DataFrame]):
    """Creates numpy files foreach frame in each video of the gebaerdenlernen dataset.

    Args:
        references (Union[str, pandas.DataFrame]): The reference file containing all videos to process. Either as path to a .csv file (str) or directly the DataFrame.
    """
    if isinstance(references, str):
        references = pd.read_csv(references, sep='|')

    for idx in range(len(references)):
        video_dir = os.path.join('data/gebaerdenlernen', references.loc[idx, 'video'])
        capture = cv2.VideoCapture(video_dir)

        pathes = video_dir.split('/')
        video_name = pathes[-1][:-4]  # get the same name for the video
        npy_video_path = os.path.join('data/gebaerdenlernen/features/npy', video_name)
        png_video_path = os.path.join('data/gebaerdenlernen/features/png', video_name)

        try:
            os.mkdir(npy_video_path)
            os.mkdir(png_video_path)
        except:
            print(f"Directory already created! Skipping following video: {video_name}")
            continue

        frame_counter = 1
        npy_video = []
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            npy_frame_path = os.path.join(npy_video_path, str(frame_counter).zfill(3))
            np.save(npy_frame_path, frame)
            npy_video.append(frame)

            png_frame_path = os.path.join(png_video_path, str(frame_counter).zfill(3))
            cv2.imwrite(png_frame_path + '.png', frame)

            frame_counter += 1

        npy_video = np.array(npy_video)

        np.save(npy_video_path, npy_video)
        np.savez(npy_video_path + '_npz', npy_video)
        np.savez_compressed(npy_video_path + '_compressed_npz', npy_video)

        print(f"Processing videos - {idx/len(references)*100}% done.")


if __name__ == "__main__":
    GEBAERDENLERNEN_DF = create_annotations_csv('data/gebaerdenlernen/gebaerden_meta.csv')
    create_numpy_arrays(GEBAERDENLERNEN_DF)
    # create_numpy_arrays('data/gebaerdenlernen/annotations/gebaerdenlernen.mp4.csv')
