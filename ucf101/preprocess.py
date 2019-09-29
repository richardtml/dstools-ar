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
from dotenv import load_dotenv
from tqdm import tqdm

import common


load_dotenv()
DATASETS_DIR = os.getenv('DATASETS_DIR')
URL = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
DS_DIR = join(DATASETS_DIR, 'ucf101')
FILENAME = 'UCF101.rar'
VIDEOS_DIR = join(DS_DIR, 'videos')


def download():
  """Downloads and extracts the UCF101 dataset.

  Video are extracted at ${DATASETS_DIR}/ucf101/videos.
  """
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')
  os.rename(os.path.join(DS_DIR, 'UCF-101'), VIDEOS_DIR)


def run(frames_per_batch=25):
  download()


if __name__ == '__main__':
  fire.Fire()
