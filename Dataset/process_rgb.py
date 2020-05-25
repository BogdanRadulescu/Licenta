import yaml
import cv2
import numpy as np
import glob
from os import mkdir

cfg = yaml.safe_load(open('config.yml', 'r'))

def get_meta(filename, as_int=True):
    A = filename[-11:-8]
    R = filename[-15:-12]
    P = filename[-19:-16]
    C = filename[-23:-20]
    S = filename[-27:-24]
    if as_int:
        return [int(A), int(R), int(P), int(C), int(S)]
    return [A, R, P, C, S]

def scale_image(frames):
    scale = cfg['scale_percent']
    return cv2.resize(frames, (int(frames.shape[1] * scale), int(frames.shape[0] * scale)), interpolation=cv2.INTER_AREA)

def resize_rgb(name: str):
    capture = cv2.VideoCapture(name)
    capture.open(name)
    images = []
    scale = cfg['scale_percent']
    while True:
        ret, img = capture.read()
        if ret:
            images.append(scale_image(img))
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    index = np.linspace(0, len(images) - 1, cfg['max_frames'])
    images = [images[int(i)] for i in index]
    return images

dataset_path = cfg["dataset_path"]
preprocessed_path = cfg["preprocessed_path_2"]
all_folders = glob.glob(f'{dataset_path}/*')
all_preprocessed = glob.glob(f'{preprocessed_path}/*')
all_unprocessed_names = [f[-28:] for f in glob.glob(f'{dataset_path}/*/*')]
to_process = [f for f in all_folders if f not in all_preprocessed]
all_preprocessed_folders = [f[-4:] for f in all_preprocessed]
all_preprocessed_names = [f[-28:] for f in glob.glob(f'{preprocessed_path}/*/*')]

def process_all_unprocessed():
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    for folder in to_process:
        print(folder)
        # uncomment for preprocessing all files, useful when changing preprocessing params
        # all_videos = [f[-28:] for f in glob.glob(f'{folder}/*')]
        all_videos = [f[-28:] for f in glob.glob(f'{folder}/*') if f[-28:] not in all_preprocessed_names]
        for video in all_videos:
            print(f'\t{video}')
            _, _, _, _, S = get_meta(video, as_int=False)
            S = f'S{S}'
            scaled_video = resize_rgb(f'{folder}\{video}')
            path = f'{preprocessed_path}\{S}'
            size = (len(scaled_video[0][0]), len(scaled_video[0]))
            if S not in all_preprocessed_folders:
                mkdir(path)
                all_preprocessed_folders.append(S)
            vw = cv2.VideoWriter(f'{path}\{video}', fourcc, 2.0, size)
            for f in scaled_video:
                vw.write(f)
            vw.release()

process_all_unprocessed()

# print(all_folders[:2])
# print(all_preprocessed[:2])
# print(all_unprocessed_names[:2])
# print(to_process[:2])
# print(all_preprocessed_folders[:2])
# print(all_preprocessed_names[:2])
    