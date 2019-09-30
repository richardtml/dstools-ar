""" cnn.py

CNN models utils.
"""  


import torch.nn as nn
from torchvision import models


def load_cnn(arch='resnet50'):
  """Loads CNN model pretrained with Imagenet without top layers.
 
  See https://pytorch.org/docs/stable/torchvision/models.html

  Parameters
  ----------
  arch : str
    Architecture name: 'resnet50' (default).

  Raises
  ------
  ValueError
    If `arch` is unknown.

  """
  if arch == 'inceptionv3':
    model = models.inception_v3(pretrained=True)
    # in_features = model.fc.in_features
    # print(model.fc.in_features)
    # model.fc = nn.AvgPool2d(8)
    model.fc = nn.AdaptiveAvgPool2d((1,1))
    #model.fc = nn.Identity()
    # nn.Linear(in_features, 2)
  elif arch == 'resnet50':
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
  else:
    raise ValueError('Unknown `arch` {arch}')
  model.eval()
  return model
