""" preprocess.py

Preprocessing for UCF101 dataset.

This modules provides functions to extract spatials frames representations
of the UCF101 dataset. Representations can be extracted using ResNet50 model,
outputs are saved at ${DATASETS}/ucf101 directory.

Example
-------
To run from scratch:

  python preprocess.py run --frames_per_batch=FRAMES_PER_BATCH

where FRAMES_PER_BATCH is the number of frames per batch,
it controls the CPU/GPU memory used.
"""


import csv
import glob
import os
import pathlib
from os.path import join

import fire
import numcodecs
import numpy as np
import torch
import zarr
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import common
from common.utils import extract_video_frames, load_config, load_cnn


load_config()
URL = 'https://cloud.xibalba.com.mx/s/swZKJGnSFqdBXj4/download'
SPLITS_URL = 'https://cloud.xibalba.com.mx/s/X248be2WBPwsbzM/download'
FILENAME = 'ucf101-videos.tar.gz'
SPLITS_FILENAME = 'ucf101-splits.tar.gz'
DATASETS_DIR = os.getenv('DATASETS_DIR')
DS_DIR = join(DATASETS_DIR, 'ucf101')
SPLITS_DIR = join(DS_DIR, 'splits')
VIDEOS_DIR = join(DS_DIR, 'videos')
FRAMES_DIR = join(DS_DIR, 'frames')
FPS = 16
NAME_PADDING = 3
RESNET_INPUT_SIZE = (224, 224)
REPS_2048_DIR = join(DS_DIR, 'ucf101_resnet50_2048.zarr')
REPS_1024_DIR = join(DS_DIR, 'ucf101_resnet50_1024.zarr')
REPS_0512_DIR = join(DS_DIR, 'ucf101_resnet50_0512.zarr')
SPLITS_FINAL_DIR = join(DS_DIR, 'splits.zarr')
FRAMES_PER_BATCH = 5
PCA_SIZE = 1024


class VFramesDataset(Dataset):
  """Video frames dataset."""

  def __init__(self, frames_dir, frame_size):
    self.frames_paths = sorted([str(f) for f in frames_dir.iterdir()])
    self.frame_size = frame_size

  def __len__(self):
    return len(self.frames_paths)

  def __getitem__(self, i):
    img = imread(self.frames_paths[i])
    img = resize(img, self.frame_size, anti_aliasing=True)
    img = np.array(img).transpose((2, 0, 1)).astype(np.float32)
    return img


def save2zarr(data, dir):
  f = zarr.open(dir, 'w')
  for video_name, video_reps, class_idx in data:
    g = f.create_group(video_name)
    g.create_dataset('x', data=video_reps, dtype=np.float32)
    g.create_dataset('y', data=class_idx, dtype=np.int32)


def load_split(splits_dir, split, subset):
  """Loads examples names in split."""
  path = join(splits_dir, f'{subset}list0{split}.txt')
  names = []
  with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
      name = row[0].split('/')[1].split('.')[0]
      names.append(name)
  return names


def download():
  """Downloads and extracts to ${DATASETS_DIR}/ucf101/videos."""
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')
  print(f'Videos saved to {VIDEOS_DIR}')


def download_splits():
  """Downloads and extracts splits to ${DATASETS_DIR}/ucf101/splits."""
  print('download_splits() running ...')
  common.utils.download(SPLITS_URL, DS_DIR, SPLITS_FILENAME, extract='auto')
  print(f'Splits saved to {SPLITS_DIR}')


def extract_frames():
  """Extracts frames to ${DATASETS_DIR}/ucf101/frames."""
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
  print(f'Frames saved to {FRAMES_DIR}')


def extract_reps(arch='resnet50', frames_per_batch=FRAMES_PER_BATCH):
  """Extracts representations to ${DATASETS_DIR}/ucf101/ucf101_resnet50_2048.zarr."""
  print('extract_reps() running ...')
  print('Computing representations')
  frames_dir = pathlib.Path(FRAMES_DIR)
  frames_dirs = sorted(list(frames_dir.glob('*/*/')))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_cnn(arch)
  model.to(device)
  reps, lens = [], []
  with torch.no_grad():
    #frames_dirs = frames_dirs[:10]
    for vframes_dir in tqdm(frames_dirs):
      ds = VFramesDataset(vframes_dir, RESNET_INPUT_SIZE)
      dl = DataLoader(ds, batch_size=frames_per_batch, num_workers=2)
      vreps = [model(frames.to(device)) for frames in dl]
      vreps = [r.cpu().numpy() for r in vreps]
      vreps = np.concatenate(vreps, 0)
      reps.append(vreps)
      lens.append(len(vreps))
  reps = np.concatenate(reps, 0)
  print('Applying z-score norm')
  reps = StandardScaler().fit_transform(reps)
  print('Saving')
  classes_names = sorted([d.name for d in frames_dir.iterdir()
      if not d.name.startswith('.')])
  classes_indices = {c: i for i, c in enumerate(classes_names)}
  indices = np.cumsum(lens)[:-1].tolist()
  reps = np.vsplit(reps, indices)
  data = []
  for vframes_dir, vreps in zip(frames_dirs, reps):
      rel_path = vframes_dir.relative_to(frames_dir)
      video_name = str(rel_path.name)
      class_name = str(rel_path.parent)
      class_idx = classes_indices[class_name]
      data.append((video_name, vreps, class_idx))
  save2zarr(data, REPS_2048_DIR)
  print(f'Representations saved to {REPS_2048_DIR}')


def reduce_reps(size=PCA_SIZE):
  """Applies PCA reduction."""
  print('reduce_reps() running ...')
  if size == 1024:
    zarr_dir = REPS_1024_DIR
  elif size == 512:
    zarr_dir = REPS_0512_DIR
  else:
    raise ValueError('Unsupported `size` representations')
  names, reps, labels, lens = [], [], [], []
  f = zarr.open(REPS_2048_DIR, 'r')
  for name in sorted(f.keys()):
    names.append(name)
    reps.append(np.array(f[name]['x']))
    labels.append(f[name]['y'][()])
    lens.append(f[name]['x'].shape[0])
  reps = np.concatenate(reps, 0)
  print('Applying PCA')
  reps = PCA(n_components=size).fit_transform(reps)
  print('Applying z-score norm')
  reps = StandardScaler().fit_transform(reps)
  indices = np.cumsum(lens)[:-1].tolist()
  reps = np.vsplit(reps, indices)
  print('Saving')
  f = zarr.open(zarr_dir, 'w')
  for name, x, y in zip(names, reps, labels):
    g = f.create_group(name)
    g.create_dataset('x', data=x, dtype=np.float32)
    g.create_dataset('y', data=y, dtype=np.int32)
  print(f'Representations saved to {zarr_dir}')


def extract_splits():
  """Extracts splits to ${DATASETS_DIR}/ucf101/splits.zarr."""
  f = zarr.open(SPLITS_FINAL_DIR, 'w')
  for split in (1, 2, 3):
    g = f.create_group(str(split))
    for subset in ('train', 'test'):
      names = load_split(SPLITS_DIR, split, subset)
      g.create_dataset(subset, data=names,
          dtype=object, object_codec=numcodecs.VLenUTF8())
  print(f'Splits saved to {SPLITS_FINAL_DIR}')


def run(arch='resnet50', frames_per_batch=FRAMES_PER_BATCH):
  download()
  extract_frames()
  extract_reps(arch, frames_per_batch)
  reduce_reps(1024)
  reduce_reps(512)
  extract_splits()


if __name__ == '__main__':
  fire.Fire()
