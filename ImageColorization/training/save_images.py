import torch
torch.cuda.device_count()
import os
import sys; sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
import pickle
from data_wrapper import Data
from utils import *
from models import *
import cv2
from tqdm import tqdm
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from IPython.display import display
from fid_score.fid_score import FidScore
import subprocess
with open('../coco_fileDict.p', 'rb') as f:
    fileDict = pickle.load(f)
device = 'cuda:0'
state_dicts = torch.load('./state_dicts/result.pt', map_location=device)
models = {}
generator = Unet(UnetEncoder, UnetDecoder)
models['gen'] = generator
models['gen'].load_state_dict(state_dicts['gen'])
for k in models.keys():
    for x in models[k].parameters():
        x.requires_grad = False
    models[k].eval()
models['gen'].to(device)
subprocess.run(['rm', '-rf', 'source_images_coco/', ' target_images_coco/'])
def display_result_real_fake(real, fake):
    if torch.is_tensor(real) and torch.is_tensor(fake):
        fig, axs = plt.subplots(1, 2, figsize=(20, 15))
        axs[0].imshow(np.array(ToPILImage()(make_grid(fake))))
        axs[0].set_title('fake')
        axs[1].imshow(np.array(ToPILImage()(make_grid(real))))
        axs[1].set_title('real')
        plt.show()
for name, module in models['gen'].named_modules():
    if 'fusion' in name:
        try:
            getattr(module, 'apply_noise')
            module.apply_noise = False
        except:
            pass
subprocess.run(['mkdir', 'source_images_coco'])
subprocess.run(['mkdir', 'target_images_coco'])
size = (128, 128)
bs = 32
ds = Data(fileDict['train'], 'val', size)
train = DataLoader(ds, batch_size=bs)
ds = Data(fileDict['val'], 'val', size)
val = DataLoader(ds, batch_size=bs)
ds = Data(fileDict['test'], 'test', size)
test = DataLoader(ds, batch_size=bs)
dl = {'Train': train, 'Val': val, 'Test': test}
c_range_scale_factor = 1.
for phase in ['Train', 'Val', 'Test']:
    print(phase)
    source_path = f'./source_images_coco/{size[0]}'
    p = source_path + f'/{phase}'
    subprocess.run(['mkdir', source_path])
    subprocess.run(['mkdir', p])
    target_path = f'./target_images_coco/{size[0]}'
    p = target_path + f'/{phase}'
    subprocess.run(['mkdir', target_path])
    subprocess.run(['mkdir', p])
    for i, d in tqdm(enumerate(dl[phase])):
        with torch.no_grad():
            x = d['bw']
            y = d['rgb']
            x = x*c_range_scale_factor
            fake = models['gen'](x.to(device))
            x = x.cpu()*0.5 + 0.5
            y = y.cpu()*0.5 + 0.5
            fake = fake.cpu()*0.5 + 0.5
            # save images
            real_images = np.array((y*255.).permute(0, 2, 3, 1)).astype(np.uint8)
            fake_images = np.array((fake*255.).permute(0, 2, 3, 1)).astype(np.uint8)
            for j in range(len(real_images)):
                Image.fromarray(fake_images[j]).save(f'{source_path}/{phase}/{i*bs + j}.jpg')
                Image.fromarray(real_images[j]).save(f'{target_path}/{phase}/{i*bs + j}.jpg')
            #display_result_real_fake(y.cpu(), fake.cpu())