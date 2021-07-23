# -*- coding: utf-8 -*-
"""Contains code to download large files from google drive.

This is a modified and slightly extended version of 
    https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
Originally taken from this StackOverflow answer: 
    https://stackoverflow.com/a/39225039
"""

import logging
from typing import Union

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def download_file_from_google_drive(id, destination) -> None:
    """Downloads and saves files from google drive, based on an ID of a sharable link.

    Args:
        id (str): The id of the google drive file.
        destination (str): The path (including file name) to save the downloaded file.
    """

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        headers = {'Range': 'bytes=0-'}
        response = session.get(URL, params=params, headers=headers, stream=True)

    LOGGER.info(f"Downloading file with id {id} from google drive..")
    save_response_content(response, destination)


def get_confirm_token(response: requests.Response) -> Union[str, None]:
    """Get the google drive specific confirmation token.

    Args:
        response (requests.Response): The initial response from google drive.

    Returns:
        Union[str, None]: Either the confirmation token or None.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response: requests.Response, destination: str) -> None:
    """Saves the response as file.

    Args:
        response (request.Response): The second response with the file as content.
        destination (str): The path where to save the file (including file name).
    """
    CHUNK_SIZE = 32768

    content_length = int(response.headers.get('Content-Range', 0).partition('/')[-1])

    with open(destination, "wb") as f:
        # TODO: enable tqdm - hower it brakes the docker logs output
        # for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=content_length / CHUNK_SIZE):
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
