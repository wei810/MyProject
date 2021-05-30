import sys; sys.path.append('../')
from models import *
import torch
from torchinfo import summary
gen = Unet(UnetEncoder, UnetDecoder)
summary(gen)