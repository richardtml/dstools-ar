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


import glob
import os
import pathlib
from os.path import join

import fire
import h5py
import numpy as np
import torch
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
DATASETS_DIR = os.getenv('DATASETS_DIR')
URL = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
DS_DIR = join(DATASETS_DIR, 'ucf101')
FILENAME = 'UCF101.rar'
VIDEOS_DIR = join(DS_DIR, 'videos')
FRAMES_DIR = join(DS_DIR, 'frames')
FPS = 10
NAME_PADDING = 3
RESNET_INPUT_SIZE = (224, 224)
REPS_2048_PATH = join(DS_DIR, 'ucf101_resnet50_2048.h5')
REPS_1024_PATH = join(DS_DIR, 'ucf101_resnet50_1024.h5')
REPS_0512_PATH = join(DS_DIR, 'ucf101_resnet50_0512.h5')
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


def save2h5(data, filepath):
  with h5py.File(filepath, 'w') as f:
    for video_name, video_reps, class_idx in data:
      g = f.create_group(video_name)
      g.create_dataset('x', data=video_reps)
      g.create_dataset('y', data=class_idx)


def download():
  """Downloads and extracts at ${DATASETS_DIR}/ucf101/videos."""
  print('download() running ...')
  common.utils.download(URL, DS_DIR, FILENAME, extract='auto')
  tmp_dir = os.path.join(DS_DIR, 'UCF-101')
  if os.path.exists(tmp_dir):
    os.rename(tmp_dir, VIDEOS_DIR)
    print(f'Videos saved at {VIDEOS_DIR}')
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
  print(f'Frames saved at {FRAMES_DIR}')


def extract_reps(arch='resnet50', frames_per_batch=FRAMES_PER_BATCH):
  """Extract representations at ${DATASETS_DIR}/ucf101/ucf101_resnet50_2048.h5."""
  print('extract_reps() running ...')  
  print('Computing representations')
  frames_dir = pathlib.Path(FRAMES_DIR)
  frames_dirs = sorted(list(frames_dir.glob('*/*/')))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_cnn(arch)
  model.to(device)
  reps = []
  lens = []
  # frames_dirs = frames_dirs[:10]
  with torch.no_grad(): 
    for vframes_dir in tqdm(frames_dirs):
      ds = VFramesDataset(vframes_dir, RESNET_INPUT_SIZE)
      dl = DataLoader(ds, batch_size=frames_per_batch, num_workers=2)
      vreps = [model(frames.to(device)) for frames in tqdm(dl, leave=False)]
      vreps = [r.cpu().numpy() for r in vreps]
      lens.append(len(vreps))
      vreps = np.concatenate(vreps, 0)
      reps.append(vreps)
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
  save2h5(data, REPS_2048_PATH)
  print(f'Representations saved at {REPS_2048_PATH}')


def reduce_reps(size=PCA_SIZE):
  """Applies PCA reduction."""
  if size == 1024:
    filename = REPS_1024_PATH
  elif size == 512:
    filename = REPS_0512_PATH
  else:
    raise ValueError('Unsupported `size` representations')
  names, reps, labels, lens = [], [], [], []
  with h5py.File(REPS_2048_PATH, 'r') as f:
    for name in sorted(f.keys()):
      names.append(name)
      reps.append(np.array(f[name]['x']))
      labels.append(f[name]['y'][()])
      lens.append(f[name]['x'].shape[0])
  reps = np.concatenate(reps, 0)
  reps = PCA(n_components=size).fit_transform(reps)
  indices = np.cumsum(lens)[:-1].tolist()
  reps = np.vsplit(reps, indices)
  with h5py.File(filename, 'w') as f:
    for name, x, y in zip(names, reps, labels):
      g = f.create_group(name)
      g.create_dataset('x', data=x)
      g.create_dataset('y', data=y)
  print(f'Representations saved at {filename}')


def run(arch='resnet50', frames_per_batch=FRAMES_PER_BATCH, size=PCA_SIZE):
  download()
  extract_frames()
  extract_reps(arch, frames_per_batch)
  reduce_reps(size)


if __name__ == '__main__':
  fire.Fire()
