""" image.py

Image utils.
"""

from PIL import Image

def resize_image(path, min_size, interpolation='lanczos'):
  """Resizes an image respecting aspect ratio with `min_size` per side."""
  with Image.open(path) as image:
    w, h = image.size
    ratio = max(min_size / w, min_size / h)
    w, h = int(w * ratio), int(h * ratio)
    resample = get_PIL_interpolation(interpolation)
    image = image.resize((w, h), resample=resample)
    image.save(path)

def get_PIL_interpolation(interpolation):
  if interpolation == 'nearest':
    return Image.NEAREST
  elif interpolation == 'lanczos':
    return Image.LANCZOS
  elif interpolation == 'bilinear':
    return Image.BILINEAR
  elif interpolation == 'bicubic':
    return Image.BICUBIC
  elif interpolation == 'cubic':
    return Image.CUBIC
