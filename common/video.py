""" video.py

Video utils.
""" 

import subprocess
import os


def count_video_frames(video_path, verbose=False):
  """Counts number of frames in video, if `verbose` prints ffprobe command."""
  cmd = (
    f'ffprobe -v error -count_frames -select_streams v:0 '
    f'-show_entries stream=nb_read_frames '
    f'-of default=nokey=1:noprint_wrappers=1 {video_path}'
  )
  if verbose:
    print(f'CMD {cmd}')
  try:
    res = subprocess.check_output(cmd, shell=True)
    res = int(res.decode('utf-8'))
    return res
  except subprocess.CalledProcessError as e:
    print(f"CalledProcessError: {video_path} {e.returncode} {e.output}")
    return -1
  except ValueError as e:
    print(f"ValueError: {video_path}")
    if not verbose:
        print(cmd)
    return 0


def extract_video_frames(video_path, frames_dir,
    n, mode='fps', pad=0, verbose=False):
  """Extracts video frames using ffmpeg.
  
  Parameters
    ----------
    video_path : str
      Video path.
    frames_dir : str
      Frames output dir, created if does not exist.
    n: int
      Number of frames to extract.
    mode : str
      If 'fps' it extracts `n` frames per second,
      if 'fpv' it extracts `n` frames per video,
      otherwise it raises ValueError.
    pad : int
      Defualt 0, number of leading zeros to pad 
      in the frames files name.
    verbose : bool
      If True, prints the ffmpeg command.
  Returns
    -------
    bool
      True if successful, False otherwise.
  """
  if mode == 'fps':
    cmd = (
      f'ffmpeg -loglevel panic '
      f'-i "{video_path}" '
      f'-vf fps={n} '
      f'-start_number 0 '
      f'"{frames_dir}/%0{pad}d.jpg"'
    )
  elif mode == 'fpv':
    cmd = (
      f'ffmpeg -loglevel panic '
      f'-i "{video_path}" '
      f'-vframes {n} '
      f'-start_number 0 '
      f'"{frames_dir}/%0{pad}d.jpg"'
    )
  else:
    raise ValueError(f"Mode {mode} unknown")
  if verbose:
    print(f'CMD {cmd}')
  os.makedirs(frames_dir, exist_ok=True)
  try:
    subprocess.check_output(cmd, shell=True)
    return True
  except subprocess.CalledProcessError as e:
    print(f"Error: {video_path} {e.returncode} {e.output}.")
    return False


def get_video_length(video_path, verbose=False):
  """Returns video length."""
  cmd = (
    f'ffprobe -v error -select_streams v:0 '
    f'-show_entries stream=duration '
    f'-of default=noprint_wrappers=1:nokey=1 {video_path}'
  )
  if verbose:
    print(f'CMD {cmd}')
  try:
    res = subprocess.check_output(cmd, shell=True)
    res = float(res.decode('utf-8'))
    return res
  except subprocess.CalledProcessError as e:
    print(f"Error: {video_path} {e.returncode} {e.output}.")
    return -1
