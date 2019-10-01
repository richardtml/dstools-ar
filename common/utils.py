""" utils.py

Common preprocessing utils.
""" 

from dotenv import load_dotenv

from common.download import *
from common.video import *
from common.cnn import *


def load_config():
  if 'DATASETS_DIR' not in os.environ:
    load_dotenv()
  if 'DATASETS_DIR' not in os.environ:
    raise ValueError(
        'DATASETS_DIR environment variable not defined, see README.md.'
    )
