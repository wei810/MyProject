import torch
torch.cuda.device_count()
import os
import sys; sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision.utils import make_grid
from torchvision.transforms import *
from ignite.engine import Engine, Events
from ignite.metrics import Average, RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.contrib.handlers import ProgressBar
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from data_wrapper import Data
from functools import partial
from models import *
from torchinfo import summary
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary_writer_path', type=str)
    parser.add_argument('state_dict_path', type=str)
    parser.add_argument('--epoch', default=25, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--tensorboard_freq', default=200, type=int)
    parser.add_argument('--generator_lr', default=0.0002, type=float)
    parser.add_argument('--critic_lr', default=0.0002, type=float)
    parser.add_argument('--l1_constant', default=100., type=float)
    args = parser.parse_args()
    print(args)
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    SIZE = (args.size, args.size)
    TENSORBOARD_FREQ = args.tensorboard_freq
    GENERATOR_LR = args.generator_lr
    CRITIC_LR = args.critic_lr
    L1_CONSTANT = args.l1_constant
    SUMMARY_WRITER_PATH = args.summary_writer_path
    STATE_DICT_PATH = args.state_dict_path
    FILE_DICT_PATH = '../coco_fileDict.p'
    DEVICE = 'cuda:0'
    with open(FILE_DICT_PATH, 'rb') as f:
        fileDict = pickle.load(f)
    gen = Unet(UnetEncoder, UnetDecoder)
    print('GENERATOR ARCHITECTURE')
    summary(gen)
    gen.to(DEVICE)
    critic = PatchCritic(ResidualLayer, [1, 1, 1, 1], nn.ReLU, nn.BatchNorm2d)
    print('CRITIC ARCHITECTURE')
    summary(critic)
    critic.to(DEVICE)
    opts = {
        'gen': optim.Adam(gen.parameters(), lr=GENERATOR_LR),
        'critic': optim.Adam(critic.parameters(), lr=CRITIC_LR)
    }
    trainDS = Data(fileDict['train'], 'train', SIZE)
    valDS = Data(fileDict['val'], 'val', SIZE)
    trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE)
    valDL = DataLoader(valDS, batch_size=BATCH_SIZE)
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionGAN = GANLoss(label_smoothing=1., device=DEVICE)
    logging_freq = 30
    def train_step(engine, batch):
        gen.train()
        critic.train()
        opts['gen'].zero_grad()
        opts['critic'].zero_grad()
        x, real = batch['bw'].to(DEVICE), batch['rgb'].to(DEVICE)
        fake = gen(x)
        fake_critic_input = torch.cat([x, fake], dim=1).detach()
        fake_outs = critic(fake_critic_input)
        real_critic_input = torch.cat([x, real], dim=1).detach()
        real_outs = critic(real_critic_input)
        d_loss = criterionGAN(fake_outs, False)*0.5 + criterionGAN(real_outs, True)*0.5
        # critic step
        d_loss.backward()
        opts['critic'].step()
        #psnr
        psnr = 10*torch.log10(4 / criterionMSE(fake, real))
        critic.eval()
        fake_critic_input = torch.cat([x, fake], dim=1)
        fake_outs = critic(fake_critic_input)
        l1_loss = criterionL1(fake, real)
        # generator steps
        gan_loss = criterionGAN(fake_outs, True)
        g_loss = L1_CONSTANT*l1_loss + gan_loss
        g_loss.backward()
        opts['gen'].step()
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gan_loss': gan_loss.item(),
            'l1_loss': l1_loss.item(),
            'psnr': psnr.item(),
        }
    tb = tb_logger.TensorboardLogger(log_dir=SUMMARY_WRITER_PATH)
    trainEngine = Engine(train_step)
    
    names = ['g_loss', 'd_loss', 'gan_loss', 'l1_loss', 'psnr']
    def ot_func(output, name):
        return output[name]
    [RunningAverage(output_transform=partial(ot_func, name=name)).attach(trainEngine, name) for name in names]
    ProgressBar().attach(trainEngine)
    def val_step(engine, batch):
        with torch.no_grad():
            x, real = batch['bw'].to(DEVICE), batch['rgb'].to(DEVICE)
            fake = gen(x).detach()
            fake_critic_input = torch.cat([x, fake], dim=1).detach()
            fake_outs = critic(fake_critic_input).detach()
            real_critic_input = torch.cat([x, real], dim=1).detach()
            real_outs = critic(real_critic_input).detach()
            d_loss = criterionGAN(fake_outs, False)*0.5 + criterionGAN(real_outs, True)*0.5
            l1_loss = criterionL1(fake, real)
            gan_loss = criterionGAN(fake_outs, True)
            g_loss = L1_CONSTANT*l1_loss + gan_loss
            #psnr
            psnr = 10*torch.log10(4 / criterionMSE(fake, real))
            return {
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'gan_loss': gan_loss.item(),
                'l1_loss': l1_loss.item(),
                'psnr': psnr.item(),
            }
    valEngine = Engine(val_step)
    [Average(output_transform=partial(ot_func, name=name)).attach(valEngine, name) for name in names]
    @trainEngine.on(Events.ITERATION_COMPLETED)
    def log(engine):
        if engine.state.iteration % logging_freq == 0:
            names = ['g_loss', 'd_loss', 'gan_loss', 'l1_loss', 'psnr']
            print(f'<iteration: {engine.state.iteration}>')
            for x in names:
                print(f'--{x}: {round(engine.state.metrics[x], 4)}')
        if engine.state.iteration % 10000 == 0:
            state_dict = {'gen': gen.state_dict(), 'critic': critic.state_dict()}
            i = STATE_DICT_PATH.find('.pt')
            path = STATE_DICT_PATH[:i] + f'_{engine.state.iteration}_.pt'
            torch.save(state_dict, path)
        if engine.state.iteration % TENSORBOARD_FREQ == 0:
            gen.eval()
            print('Logging images to Tensorboard ...')
            np.random.seed(engine.state.iteration)
            num = 32
            train_picked = np.random.choice(fileDict['train'], num, replace=False)
            ds = Data(train_picked, 'val', SIZE)
            train = DataLoader(ds, batch_size=num)
            val_picked = np.random.choice(fileDict['val'], num, replace=False)
            ds = Data(val_picked, 'val', SIZE)
            val = DataLoader(ds, batch_size=num)
            dl = {'Train': train, 'Val': val}
            for phase in ['Train', 'Val']:
                for d in dl[phase]:
                    with torch.no_grad():
                        x, real = d['bw'].to(DEVICE), d['rgb']
                        fake = gen(x)
                        x = x*0.5 + 0.5
                        real = real*0.5 + 0.5
                        fake = fake*0.5 + 0.5
                tb.writer.add_image(f'{phase}/X', make_grid(x.cpu()), engine.state.iteration)
                tb.writer.add_image(f'{phase}/Real', make_grid(real.cpu()), engine.state.iteration)
                tb.writer.add_image(f'{phase}/Fake', make_grid(fake.cpu()), engine.state.iteration)
        tb.writer.add_scalar('G_Loss/Train', engine.state.metrics['g_loss'], engine.state.iteration)
        tb.writer.add_scalar('D_Loss/Train', engine.state.metrics['d_loss'], engine.state.iteration)
        tb.writer.add_scalar('G_Gan_Loss/Train', engine.state.metrics['gan_loss'], engine.state.iteration)
        tb.writer.add_scalar('G_L1_Loss/Train', engine.state.metrics['l1_loss'], engine.state.iteration)
        tb.writer.add_scalar('G_PSNR/Train', engine.state.metrics['psnr'], engine.state.iteration)
    @trainEngine.on(Events.EPOCH_COMPLETED)
    def complete(engine):
        gen.eval()
        critic.eval()
        names = ['g_loss', 'd_loss', 'gan_loss', 'l1_loss', 'psnr']
        print('Training Results - Epoch[%d]' % (engine.state.epoch))
        for x in names:
            print(f'{x}={round(engine.state.metrics[x], 4)}')
        valEngine.run(valDL)
        print('Validating Results - Epoch[%d]' % (engine.state.epoch))
        for x in names:
            print(f'{x}={round(valEngine.state.metrics[x], 4)}')
        tb.writer.add_scalar('D_Loss/Val', valEngine.state.metrics['d_loss'], engine.state.epoch)
        tb.writer.add_scalar('G_Loss/Val', valEngine.state.metrics['g_loss'], engine.state.epoch)
        tb.writer.add_scalar('G_L1_Loss/Val', valEngine.state.metrics['l1_loss'], engine.state.epoch)
        tb.writer.add_scalar('G_Gan_Loss/Val', valEngine.state.metrics['gan_loss'], engine.state.epoch)
        tb.writer.add_scalar('G_PSNR/Val', valEngine.state.metrics['psnr'], engine.state.epoch)
    trainEngine.run(trainDL,  max_epochs=EPOCH)
    state_dict = {'gen': gen.state_dict(), 'critic': critic.state_dict()}
    torch.save(state_dict, STATE_DICT_PATH)
if __name__ == '__main__':
    main()