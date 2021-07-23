"""Contains the code to augment the RWTH Phoenix 2014T Weather Dataset with temporal gloss information."""
import glob
import json
import os
import sys
from queue import PriorityQueue

import cv2
import dtw
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from sklearn.neighbors import KDTree


def create_dir(output_dir: str) -> None:
    """Creating a new directory.

    Args:
        output_dir (str): The path to the directory which should be created.
    """
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_array_from_keypoints(pathes_to_keypoints: list) -> np.ndarray:
    """This method loads all 2D features from the given keypoint files and returns them as one numpy array.

    Args:
        pathes_to_keypoints (list): A list of strings containing pathes to keypoint json files, which should be considered.

    Returns:
        np.ndarray: A numpy array which contains all 2d information from the given keypoint files. (concatenated 2d array - (#files, features))
    """
    video = []

    # make sure order of keypoints is correct
    pathes_to_keypoints = sorted(pathes_to_keypoints)

    for keypoint_file in pathes_to_keypoints:
        with open(keypoint_file) as file:
            joint_frame = json.loads(file.read())['people'][0]
            joints = []
            joints.extend(joint_frame['face_keypoints_2d'])
            joints.extend(joint_frame['hand_left_keypoints_2d'])
            joints.extend(joint_frame['hand_right_keypoints_2d'])
            joints.extend(joint_frame['pose_keypoints_2d'])

            joints = np.array(joints)

            # Every 3rd element in the frame keypoints is the confidence of the preceding two. Therefore, we need to remove every 3rd element.
            joints_mask = np.ones(joints.size, dtype=bool)
            joints_mask[2::3] = 0  # start with 3rd element, then do 3er steps
            joints = joints[joints_mask]

            video.append(joints)

    return np.array(video)


def load_array_from_keypoints_as_bodyparts(pathes_to_keypoints: list) -> np.ndarray:
    """This method loads all 2D features from the given keypoint files and returns them as a dictonary of bodyparts (numpy array).

    Args:
        pathes_to_keypoints (list): A list of strings containing pathes to keypoint json files, which should be considered.

    Returns:
        dict: A dictonariy which contains all bodyparts with their information as numpy arrays. ("bodypart": [[frame_keypoints], ..])
    """
    # make sure order of keypoints is correct
    pathes_to_keypoints = sorted(pathes_to_keypoints)

    face_keypoints = []
    hand_left_keypoints = []
    hand_right_keypoints = []
    pose_keypoints = []

    for keypoint_file in pathes_to_keypoints:
        with open(keypoint_file) as f:
            joint_frame = json.loads(f.read())['people'][0]

            face = []
            hand_l = []
            hand_r = []
            pose = []

            face.extend(joint_frame['face_keypoints_2d'])
            hand_l.extend(joint_frame['hand_left_keypoints_2d'])
            hand_r.extend(joint_frame['hand_right_keypoints_2d'])
            pose.extend(joint_frame['pose_keypoints_2d'])

            face = np.array(face)
            hand_l = np.array(hand_l)
            hand_r = np.array(hand_r)
            pose = np.array(pose)

            # Every 3rd element in the frame keypoints is the confidence of the preceding two. Therefore, we need to remove every 3rd element.
            joints_mask = np.ones(face.size, dtype=bool)
            joints_mask[2::3] = 0  # start with 3rd element, then do 3er steps
            face = face[joints_mask]
            face_keypoints.append(face)

            joints_mask = np.ones(hand_l.size, dtype=bool)
            joints_mask[2::3] = 0  # start with 3rd element, then do 3er steps
            hand_l = hand_l[joints_mask]
            hand_left_keypoints.append(hand_l)

            joints_mask = np.ones(hand_r.size, dtype=bool)
            joints_mask[2::3] = 0  # start with 3rd element, then do 3er steps
            hand_r = hand_r[joints_mask]
            hand_right_keypoints.append(hand_r)

            joints_mask = np.ones(pose.size, dtype=bool)
            joints_mask[2::3] = 0  # start with 3rd element, then do 3er steps
            pose = pose[joints_mask]
            pose_keypoints.append(pose)

    video = {"face": face_keypoints, "hand_left": hand_left_keypoints, "hand_right": hand_right_keypoints, "pose": pose_keypoints}

    return video


def keypoints_to_bodyparts(sequence) -> dict:
    """Transforms a numpy array of 2d keypoints to a bodypart dictionary.

    The numpy array must be in shape of (#frames, bodyparts_keypoints). 
    Where bodyparts_keypoints is a concatenated array of (face, hand_left, hand_right, pose).
    The correct order matters!

    Args:
        sequence (): The sequence to transform

    Returns:
        dict: A dictonariy which contains all bodyparts with their information as numpy arrays. ("bodypart": [[frame_keypoints], ..])
    """
    face_keypoints = sequence[:, :140]
    hand_left_keypoints = sequence[:, 140:182]
    hand_right_keypoints = sequence[:, 182:224]
    pose_keypoints = sequence[:, 224:]
    # pose_keypoints2 = sequence[:, 224:]

    video = {"face": face_keypoints, "hand_left": hand_left_keypoints, "hand_right": hand_right_keypoints, "pose": pose_keypoints}
    return video


def visualise(sequence, axis_limit: int = 1, fps: int = 30) -> None:
    """Visualises the given sequence of keypoints.

    Args:
        sequence (): The sequence of keypoints to visualise.
        axis_limit (int): The limit of the x and y axis.
        fps (int): The fps of the displayed video.
    """
    n_frames = len(sequence)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=2)

    def update(ifrm, xa, ya):
        sct.set_data(xa[ifrm], ya[ifrm])

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ani = animation.FuncAnimation(fig, update, n_frames, fargs=(sequence[:, 0::2], sequence[:, 1::2]), interval=1000 / fps, repeat=True)

    plt.show()


def trim_keypoint_sequence(sequence, adist=18000):
    """Trims frames within the sequence, where is no movement.

    Args:
        sequence (): The sequence of keypoints to trim.
        adist (): Absolut distance to first frame (which frames should be trimmed).

    Returns:
        The trimmed sequence.
    """
    old_frame = None
    first_frame = None
    trimmed_sequence = []
    for idx, frame in enumerate(sequence):
        ssd = sys.maxsize
        if old_frame is None:
            first_frame = frame
        else:
            ssd = ((frame - first_frame)**2).sum()
            idx += 1
            if ssd <= adist:
                continue
        old_frame = frame
        trimmed_sequence.append(frame)
    return np.array(trimmed_sequence)


def interpolate_zero_frames(sequence):
    """Interpolates a sequence of keypoints and fills the zero frames with interpolated values.

    Args:
        sequence (): The sequence of keypoints to interpolate.

    Returns:
        The trimmed sequence.
    """
    # for each keypoint value (x, y or z) series (over the whole sequence)
    for keypoint_1d_idx in range(sequence.shape[1]):
        # if all values are zero or if no zero -> continue
        if not np.any(sequence[:, keypoint_1d_idx] == 0.0) or np.all(sequence[:, keypoint_1d_idx] == 0.0):
            continue
        time_series = np.nonzero(sequence[:, keypoint_1d_idx])[0]
        zero_series = np.where(sequence[:, keypoint_1d_idx] == 0.0)[0]
        time_series_values = np.take(sequence[:, keypoint_1d_idx], time_series)

        # interpolate zero
        interpolated_zeroes = np.interp(zero_series, time_series, time_series_values)

        # put interpolated values into sequence
        np.put(sequence[:, keypoint_1d_idx], zero_series, interpolated_zeroes)
    return sequence


def normalize(sequence):
    """Normalizes a sequence of keypoints.

    Args:
        sequence (): The sequence of keypoints to normalize.

    Returns:
        The normalized sequence.
    """
    # return sequence / np.linalg.norm(sequence)
    sequence *= 1 / sequence.max()
    return sequence


def subtract_mean(sequence):
    """Computes and subtracts the mean of a sequence of keypoints.

    Args:
        sequence (): The sequence of keypoints to subtract the mean.

    Returns:
        The mean subtracted sequence.
    """
    # if sequence consists of all zeros return the sequence
    if sequence.sum() == 0.0:
        return sequence

    x = sequence[:, ::2]
    y = sequence[:, 1::2]

    x = x - x[x != 0.0].mean()  # ignore zero keypoints for mean computation
    y = y - y[y != 0.0].mean()

    sequence[:, ::2] = x
    sequence[:, 1::2] = y
    return sequence
    # return sequence - sequence[sequence != 0.0].mean()  # ignore zero keypoints for mean computation


def reduce_video():
    """This method reduces a GebaerdenLernen video by calculating differences to the first frame and omit those with a value below a threshold."""
    video = []
    cap = cv2.VideoCapture('data/gebaerdenlernen/features/mp4/regen.mp4')

    old_frame = None
    first_frame = None
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ssd = sys.maxsize
        if old_frame is None:
            first_frame = frame
        else:
            ssd = ((frame - first_frame)**2).sum()
            # ssd = ((frame - old_frame)**2).sum()
            print(f"Idx: {idx:3}, Diff: {ssd:10}")
            idx += 1
            # if ssd <= 2300000:
            if ssd <= 15000000:
                continue

        old_frame = frame

        video.append(frame)
    video = np.array(video)
    # print(f"Length of video: {x.shape[0]}")

    writer = cv2.VideoWriter('regen.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (640, 480), True)
    for frame in video:
        writer.write(frame.astype('uint8'))
    writer.release()


def augment_phoenix_spatial_embeddings(output_dir: str = './output/phoenix', split_type: str = 'train') -> None:
    """Augments the RWTH Phoenix 2014T Weather dataset with temporal gloss annotations, based on the spatial embedding from ResNext for each frame.

    Args:
        output_dir (str): The directory where the augmented files should be saved.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """

    # Phoenix video
    # NACHMITTAG LAND BISSCHEN REGEN
    video = torch.squeeze(torch.load('./data/spatialembeddings/phoenix/train/18April_2010_Sunday_tagesschau-6658.pt',
                                     map_location=lambda storage, loc: storage))

    # GebaerdenLernen glosses
    # nachmittag.mp4 land_staat.mp4 || land_boden.mp4 bisschen.mp4 regen.mp4
    nachmittag_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/nachmittag.pt', map_location=lambda storage, loc: storage))
    land_staat_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/land_staat.pt', map_location=lambda storage, loc: storage))
    land_boden_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/land_boden.pt', map_location=lambda storage, loc: storage))
    bisschen_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/bisschen.pt', map_location=lambda storage, loc: storage))
    regen_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/regen.pt', map_location=lambda storage, loc: storage))
    abitur_gloss = torch.squeeze(torch.load('./data/spatialembeddings/gebaerdenlernen/abitur.pt', map_location=lambda storage, loc: storage))

    video_length = video.numpy().shape[0]
    video_splitted_length = int(video_length / 4)

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[:video_splitted_length * 2, :], nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag (1/2) dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[video_splitted_length * 2:, :], nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag (2/2) dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[:video_splitted_length, :], nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag (1/4) dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[video_splitted_length:video_splitted_length * 2, :], nachmittag_gloss.numpy(),
                                                                   'euclidean')
    print(f"Nachmittag (2/4) dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[video_splitted_length * 2:video_splitted_length * 3, :],
                                                                   nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag (3/4) dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy()[video_splitted_length * 3:, :], nachmittag_gloss.numpy(), 'euclidean')
    print(f"Nachmittag (4/4) dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), land_staat_gloss.numpy(), 'euclidean')
    print(f"Land staat dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), land_boden_gloss.numpy(), 'euclidean')
    print(f"Land boden dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), bisschen_gloss.numpy(), 'euclidean')
    print(f"Bisschen dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), regen_gloss.numpy(), 'euclidean')
    print(f"Regen dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video.numpy(), abitur_gloss.numpy(), 'euclidean')
    print(f"Abitur dist: {dist}")


def augment_phoenix_openpose(output_dir: str = './output/phoenix', split_type: str = 'train') -> None:
    """Augments the RWTH Phoenix 2014T Weather dataset with temporal gloss annotations, based on the openpose keyjoints for each frame.

    Args:
        output_dir (str): The directory where the augmented files should be saved.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """
    # video directories
    video_dir = glob.glob('./data/openpose/normal/18April_2010_Sunday_tagesschau-6658/*.json')
    nachmittag_dir = glob.glob('./data/openpose/normal/nachmittag/*.json')
    cropped_1_nachmittag_dir = glob.glob('./data/openpose/normal/1/*.json')
    cropped_3_nachmittag_dir = glob.glob('./data/openpose/normal/3/*.json')

    land_staat_dir = glob.glob('./data/openpose/normal/land_staat/*.json')
    land_boden_dir = glob.glob('./data/openpose/normal/land_boden/*.json')
    bisschen_dir = glob.glob('./data/openpose/normal/bisschen/*.json')
    regen_dir = glob.glob('./data/openpose/normal/regen/*.json')
    abitur_dir = glob.glob('./data/openpose/normal/abitur/*.json')

    # load numpy arrays
    video = load_array_from_keypoints(video_dir)
    nachmittag = load_array_from_keypoints(nachmittag_dir)
    cropped_1_nachmittag = load_array_from_keypoints(cropped_1_nachmittag_dir)
    cropped_3_nachmittag = load_array_from_keypoints(cropped_3_nachmittag_dir)

    land_staat = load_array_from_keypoints(land_staat_dir)
    land_boden = load_array_from_keypoints(land_boden_dir)
    bisschen = load_array_from_keypoints(bisschen_dir)
    regen = load_array_from_keypoints(regen_dir)
    abitur = load_array_from_keypoints(abitur_dir)

    # compute DTW
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, nachmittag, 'euclidean')
    print(f"Nachmittag Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video[:26, :], nachmittag, 'euclidean')
    print(f"Nachmittag - Video(:26) Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video[:26, :], nachmittag[60:-60, :], 'euclidean')
    print(f"Nachmittag(60:-60) - Video(:26) Dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, cropped_1_nachmittag, 'euclidean')
    print(f"Nachmittag 1 Cropped Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video[:26, :], cropped_1_nachmittag, 'euclidean')
    print(f"Nachmittag 1 Cropped - Video(:26) Dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, cropped_3_nachmittag, 'euclidean')
    print(f"Nachmittag 3 Cropped - Video Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video[:26, :], cropped_3_nachmittag, 'euclidean')
    print(f"Nachmittag 3 Cropped - Video(:26) Dist: {dist}")

    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, land_staat, 'euclidean')
    print(f"Land Staat Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, land_boden, 'euclidean')
    print(f"Land Boden Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, bisschen, 'euclidean')
    print(f"Bisschen Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, regen, 'euclidean')
    print(f"Regen Dist: {dist}")
    dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(video, abitur, 'euclidean')
    print(f"Abitur Dist: {dist}")

    # brute force compute best fitting window
    best_dist = sys.maxsize
    for subsequence_length in range(1, len(video)):
        for window in range(len(video) - subsequence_length):
            dist, _, _, _ = dtw.accelerated_dtw(video[window:window + subsequence_length, :], cropped_3_nachmittag, 'euclidean')
            if dist < best_dist:
                best_dist = dist
                print(f"S-length: {subsequence_length} - Idx: {window} - Dist: {dist}")
    # Smaller windows -> better results (lesser points to fit)

    # brute force compute best fitting window
    best_dist = sys.maxsize
    for subsequence_length in range(30, len(video)):
        for window in range(len(video) - subsequence_length):
            dist, _, _, _ = dtw.accelerated_dtw(video[window:window + subsequence_length, :], regen, 'euclidean')
            if dist < best_dist:
                best_dist = dist
                print(f"S-length: {subsequence_length} - Idx: {window} - Dist: {dist}")

    # concatenate single glosses to longer sequence -> "ideal" video representation
    video_replica = np.concatenate((nachmittag, land_boden, bisschen, regen), axis=0)
    dist, _, _, _ = dtw.accelerated_dtw(video, video_replica, 'euclidean')
    print(f"Dist: {dist}")
    video_replica = np.concatenate((nachmittag, land_staat, bisschen, regen), axis=0)
    dist, _, _, _ = dtw.accelerated_dtw(video, video_replica, 'euclidean')
    print(f"Dist: {dist}")
    # even worse results..


def augment_phoenix_openpose_dtw_chunks(output_dir: str = './output/phoenix', split_type: str = 'train') -> None:
    """Augments the RWTH Phoenix 2014T Weather dataset with temporal gloss annotations, based on the openpose keyjoints for each frame; computed with dtw and frame chunks.

    Args:
        output_dir (str): The directory where the augmented files should be saved.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """
    # video directories
    video_dir = glob.glob('./data/openpose/normal/18April_2010_Sunday_tagesschau-6658/*.json')
    nachmittag_dir = glob.glob('./data/openpose/normal/nachmittag/*.json')
    cropped_1_nachmittag_dir = glob.glob('./data/openpose/normal/1/*.json')
    cropped_3_nachmittag_dir = glob.glob('./data/openpose/normal/3/*.json')

    land_staat_dir = glob.glob('./data/openpose/normal/land_staat/*.json')
    land_boden_dir = glob.glob('./data/openpose/normal/land_boden/*.json')
    bisschen_dir = glob.glob('./data/openpose/normal/bisschen/*.json')
    regen_dir = glob.glob('./data/openpose/normal/regen/*.json')
    abitur_dir = glob.glob('./data/openpose/normal/abitur/*.json')

    # load numpy arrays
    video = load_array_from_keypoints(video_dir)
    nachmittag = load_array_from_keypoints(nachmittag_dir)
    cropped_1_nachmittag = load_array_from_keypoints(cropped_1_nachmittag_dir)
    cropped_3_nachmittag = load_array_from_keypoints(cropped_3_nachmittag_dir)

    land_staat = load_array_from_keypoints(land_staat_dir)
    land_boden = load_array_from_keypoints(land_boden_dir)
    bisschen = load_array_from_keypoints(bisschen_dir)
    regen = load_array_from_keypoints(regen_dir)
    abitur = load_array_from_keypoints(abitur_dir)

    # interpolate and fill zero frames
    video = interpolate_zero_frames(video)
    # visualise(video)
    nachmittag = interpolate_zero_frames(nachmittag)
    land_boden = interpolate_zero_frames(land_boden)
    bisschen = interpolate_zero_frames(bisschen)
    regen = interpolate_zero_frames(regen)

    # apply savgol filter to all videos in order to smooth them
    video = savgol_filter(video, 11, 3, axis=0)
    # visualise(video)
    nachmittag = savgol_filter(nachmittag, 11, 3, axis=0)
    land_boden = savgol_filter(land_boden, 11, 3, axis=0)
    bisschen = savgol_filter(bisschen, 11, 3, axis=0)
    regen = savgol_filter(regen, 11, 3, axis=0)

    # trim gloss videos where is no movement
    #! different gloss needs different dist?
    nachmittag = trim_keypoint_sequence(nachmittag)
    land_boden = trim_keypoint_sequence(land_boden)
    bisschen = trim_keypoint_sequence(bisschen)
    regen = trim_keypoint_sequence(regen)

    # normalize and subtract mean
    video = normalize(video)
    video = subtract_mean(video)

    nachmittag = normalize(nachmittag)
    nachmittag = subtract_mean(nachmittag)
    land_boden = normalize(land_boden)
    land_boden = subtract_mean(land_boden)
    bisschen = normalize(bisschen)
    bisschen = subtract_mean(bisschen)
    regen = normalize(regen)
    regen = subtract_mean(regen)

    # visualise(nachmittag)
    # visualise(land_boden)
    # visualise(bisschen)
    # visualise(regen)

    # separate weather video into 8 frame chunks
    chunk_size = 8
    video = [video[i * chunk_size:(i + 1) * chunk_size] for i in range(int((len(video) + chunk_size - 1) // chunk_size))]

    # select just first, middle and last frame in each 8 frame chunk
    video = np.array([[item[0], item[len(item) // 2], item[-1]] for item in video])

    # separate gloss videos into 10 frame chunks
    chunk_size = 10
    nachmittag = [nachmittag[i * chunk_size:(i + 1) * chunk_size] for i in range(int((len(nachmittag) + chunk_size - 1) // chunk_size))]
    land_boden = [land_boden[i * chunk_size:(i + 1) * chunk_size] for i in range(int((len(land_boden) + chunk_size - 1) // chunk_size))]
    bisschen = [bisschen[i * chunk_size:(i + 1) * chunk_size] for i in range(int((len(bisschen) + chunk_size - 1) // chunk_size))]
    regen = [regen[i * chunk_size:(i + 1) * chunk_size] for i in range(int((len(regen) + chunk_size - 1) // chunk_size))]

    # select just first, middle and last frame in each 10 frame chunk
    nachmittag = np.array([[item[0], item[len(item) // 2], item[-1]] for item in nachmittag])
    land_boden = np.array([[item[0], item[len(item) // 2], item[-1]] for item in land_boden])
    bisschen = np.array([[item[0], item[len(item) // 2], item[-1]] for item in bisschen])
    regen = np.array([[item[0], item[len(item) // 2], item[-1]] for item in regen])

    # concatenate all gloss chunks to one array
    gloss_chunks = []
    gloss_chunks.append((nachmittag, 'nachmittag'))
    gloss_chunks.append((land_boden, 'land_boden'))
    gloss_chunks.append((bisschen, 'bisschen'))
    gloss_chunks.append((regen, 'regen'))

    # DTW for all video and gloss neighbor sequences
    p_queue = PriorityQueue()
    # dtw_results = []
    for v_idx, video_chunk in enumerate(video):
        for gloss_array, gloss_name in gloss_chunks:
            for g_idx, gloss_chunk in enumerate(gloss_array):
                dist, _, _, _ = dtw.accelerated_dtw(video_chunk, gloss_chunk, 'euclidean')
                # dtw_results.append({'gloss_name': gloss_name, 'video_chunk': v_idx, 'gloss_chunk': g_idx, 'dist': dist})
                p_queue.put((v_idx, dist, g_idx, gloss_name))

    # pick top k results foreach video chunk
    ranked_results = []
    idx = 0
    counter = 0
    top_k = 10
    while not p_queue.empty():
        v_idx, dist, g_idx, gloss_name = p_queue.get()
        if v_idx != idx:
            counter = 0
            idx = v_idx
        if counter < top_k:
            ranked_results.append((v_idx, dist, g_idx, gloss_name))
        counter += 1

    print("")

    # # find best fit for one gloss in the video
    # b_queue = PriorityQueue()
    # for g_idx, gloss_chunk in enumerate(regen):
    #     for v_idx, video_chunk in enumerate(video):
    #         dist, _, _, _ = dtw.accelerated_dtw(video_chunk, gloss_chunk, 'euclidean')
    #         b_queue.put((g_idx, dist, v_idx))

    # b_ranked_results = []
    # idx = 0
    # counter = 0
    # top_k = 10
    # while not b_queue.empty():
    #     g_idx, dist, v_idx = b_queue.get()
    #     if g_idx != idx:
    #         counter = 0
    #         idx = g_idx
    #     # if counter < top_k:
    #     b_ranked_results.append((g_idx, dist, v_idx))
    #     counter += 1

    # print("")


def augment_phoenix_openpose_kd(output_dir: str = './output/phoenix', split_type: str = 'train') -> None:
    """Augments the RWTH Phoenix 2014T Weather dataset with temporal gloss annotations, based on the openpose keyjoints for each frame; computed with a kd tree.

    Args:
        output_dir (str): The directory where the augmented files should be saved.
        split_type (str): The dataset split you want to process (e.g. train, dev, test).
    """
    # video directories
    # NACHMITTAG LAND BISSCHEN REGEN
    video_dir = glob.glob('./data/openpose/normal/18April_2010_Sunday_tagesschau-6658/*.json')
    nachmittag_dir = glob.glob('./data/openpose/normal/nachmittag/*.json')
    land_boden_dir = glob.glob('./data/openpose/normal/land_boden/*.json')
    bisschen_dir = glob.glob('./data/openpose/normal/bisschen/*.json')
    regen_dir = glob.glob('./data/openpose/normal/regen/*.json')

    # load numpy arrays
    video = load_array_from_keypoints(video_dir)
    nachmittag = load_array_from_keypoints(nachmittag_dir)
    land_boden = load_array_from_keypoints(land_boden_dir)
    bisschen = load_array_from_keypoints(bisschen_dir)
    regen = load_array_from_keypoints(regen_dir)

    ### pre-process ###

    # interpolate and fill zero frames
    video = interpolate_zero_frames(video)
    nachmittag = interpolate_zero_frames(nachmittag)
    land_boden = interpolate_zero_frames(land_boden)
    bisschen = interpolate_zero_frames(bisschen)
    regen = interpolate_zero_frames(regen)

    # apply savgol filter to all videos in order to smooth them
    video = savgol_filter(video, 11, 3, axis=0)
    nachmittag = savgol_filter(nachmittag, 11, 3, axis=0)
    land_boden = savgol_filter(land_boden, 11, 3, axis=0)
    bisschen = savgol_filter(bisschen, 11, 3, axis=0)
    regen = savgol_filter(regen, 11, 3, axis=0)

    # # trim gloss videos where is no movement
    # #! different gloss needs different dist?
    # nachmittag = trim_keypoint_sequence(nachmittag)
    # land_boden = trim_keypoint_sequence(land_boden)
    # bisschen = trim_keypoint_sequence(bisschen)
    # regen = trim_keypoint_sequence(regen)

    ### normalization ###

    # parse keypoint videos to bodyparts
    video = keypoints_to_bodyparts(video)  # video2 = load_array_from_keypoints_as_bodyparts(video_dir)
    # visualise(video2['face'])
    # visualise(video2['hand_left'])
    # visualise(video2['hand_right'])
    # visualise(video2['pose'])
    nachmittag = keypoints_to_bodyparts(nachmittag)
    land_boden = keypoints_to_bodyparts(land_boden)
    bisschen = keypoints_to_bodyparts(bisschen)
    regen = keypoints_to_bodyparts(regen)

    # normalize and subtract mean for each bodypart in each video
    for vid in (video, nachmittag, land_boden, bisschen, regen):
        for key in vid:
            subtract_mean(vid[key])
            normalize(vid[key])

    ### KD Tree ###
    #   0- 45 nachmittag
    #  46- 84 land_boden
    #  85-115 bisschen
    # 116-167 regen
    # gloss_db = np.concatenate((nachmittag, land_boden, bisschen, regen), axis=0)
    # kd_tree = KDTree(gloss_db)
    # nearest_dist, nearest_ind = kd_tree.query(gloss_db[100].reshape(1, -1), k=3)

    # TODO: feature selection? frame-wise or chunks?

    # TODO: db bodypart mask -> which index belongs to which bodypart?

    face_db = []
    face_db_mask = []
    hand_left_db = []
    hand_left_db_mask = []
    hand_right_db = []
    hand_right_db_mask = []
    pose_db = []
    pose_db_mask = []
    for name, vid in (('nachmittag', nachmittag), ('land_boden', land_boden), ('bisschen', bisschen), ('regen', regen)):
        if not np.isnan(vid['face']).all():
            face_db.extend(vid['face'])
            face_db_mask.extend([name] * len(vid['face']))
        if not np.isnan(vid['hand_left']).all():
            hand_left_db.extend(vid['hand_left'])
            hand_left_db_mask.extend([name] * len(vid['hand_left']))
        if not np.isnan(vid['hand_right']).all():
            hand_right_db.extend(vid['hand_right'])
            hand_right_db_mask.extend([name] * len(vid['hand_right']))
        if not np.isnan(vid['pose']).all():
            pose_db.extend(vid['pose'])
            pose_db_mask.extend([name] * len(vid['pose']))

    # KD Tree for all bodyparts

    face_kd = KDTree(np.array(face_db))
    hand_left_kd = KDTree(np.array(hand_left_db))
    hand_right_kd = KDTree(np.array(hand_right_db))
    pose_kd = KDTree(np.array(pose_db))

    # KD query for each video sub sequence

    _, nearest_ind_face = face_kd.query(video['face'], k=10)
    _, nearest_ind_hand_left = hand_left_kd.query(video['hand_left'], k=10)
    _, nearest_ind_hand_right = hand_right_kd.query(video['hand_right'], k=10)
    _, nearest_ind_pose = pose_kd.query(video['pose'], k=10)

    nearest_ind_face_string = []
    for row in nearest_ind_face:
        _row = []
        for idx in row:
            _row.append(face_db_mask[idx])
        nearest_ind_face_string.append(_row)

    nearest_ind_hand_left_string = []
    for row in nearest_ind_hand_left:
        _row = []
        for idx in row:
            _row.append(hand_left_db_mask[idx])
        nearest_ind_hand_left_string.append(_row)

    nearest_ind_hand_right_string = []
    for row in nearest_ind_hand_right:
        _row = []
        for idx in row:
            _row.append(hand_right_db_mask[idx])
        nearest_ind_hand_right_string.append(_row)
        
    nearest_ind_pose_string = []
    for row in nearest_ind_pose:
        _row = []
        for idx in row:
            _row.append(pose_db_mask[idx])
        nearest_ind_pose_string.append(_row)

    # TODO: process nearest indexes (with a graph?) to find the best fitting bodypart to each sequence queue
    print("")


if __name__ == "__main__":
    # augment_phoenix_spatial_embeddings('./output/phoenix/spatialembeddings/train', 'train')
    # augment_phoenix_spatial_embeddings('./output/phoenix/spatialembeddings/test', 'test')
    # augment_phoenix_spatial_embeddings('./output/phoenix/spatialembeddings/dev', 'dev')
    # augment_phoenix_openpose('./output/phoenix/openpose/train', 'train')
    # augment_phoenix_openpose('./output/phoenix/openpose/test', 'test')
    # augment_phoenix_openpose('./output/phoenix/openpose/dev', 'dev')
    # augment_phoenix_openpose_dtw_chunks('./output/phoenix/openpose/train', 'train')
    augment_phoenix_openpose_kd('./output/phoenix/openpose/train', 'train')
