""" preprocess.py

Preprocessing for UCF101 dataset.

This modules provides functions to extract spatials frames representations
of the UCF101 dataset. Representations can be extracted using ResNet50 or 
InceptionV3 models and outputs are placed at ${DATASETS} directory.

Example
-------
To run from scratch:

  python preprocess.py run FRAMES_PER_BATCH

where FRAMES_PER_BATCH is the number of frames per batch, 
it controls the memory used.
"""


import glob
import os
import pathlib
from os.path import join

import fire
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv
from tqdm import tqdm

import common
from common.utils import extract_video_frames


load_dotenv()
DATASETS_DIR = os.getenv('DATASETS_DIR')
URL = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
DS_DIR = join(DATASETS_DIR, 'ucf101')
FILENAME = 'UCF101.rar'
VIDEOS_DIR = join(DS_DIR, 'videos')
FRAMES_DIR = join(DS_DIR, 'frames')
FPS = 10
NAME_PADDING = 3


def download():
  """Downloads and extracts at ${DATASETS_DIR}/ucf101/videos."""
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')
  tmp_dir = os.path.join(DS_DIR, 'UCF-101')
  if os.path.exists(tmp_dir):
    os.rename(tmp_dir, VIDEOS_DIR)
  else:
    print(
      f"Error, extracted 'UCF-101' directory not found, "
      "could not rename to 'videos' directory"
    )


def extract_frames():
  """Extract frames at ${DATASETS_DIR}/ucf101/frames."""
  print('extract_frames() running ...')
  videos_dir = pathlib.Path(VIDEOS_DIR)
  frames_dir = pathlib.Path(FRAMES_DIR)
  frames_dir.mkdir(exist_ok=True)
  videos_paths = sorted(list(videos_dir.glob('*/*.avi')))
  result = []
  for video_path in videos_paths:
    rel_path = video_path.relative_to(videos_dir)
    video_frames_dir = frames_dir / rel_path.parent / rel_path.stem
    result.append(delayed(extract_video_frames)(video_path, 
      video_frames_dir, FPS, 'fps', NAME_PADDING))
  with ProgressBar():
    compute(result)


def run(frames_per_batch=25):
  download()
  extract_frames()


if __name__ == '__main__':
  fire.Fire()
