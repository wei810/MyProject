import torch
torch.cuda.is_available()
from torch import nn
from torch.nn.utils import spectral_norm 
from torchvision import models
from utils import *
from typing import Type, Any, Callable, Union, List, Optional
from functools import partial

# Residual Network Implemetation
# Reproduced from Pytorch (2021) resnet.py [source code] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv1x1(inplanes: int, planes: int, stride: int = 1):
    return nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
def conv3x3(inplanes: int, planes: int, stride: int, padding: int = 1):
    return nn.Conv2d(inplanes, planes, 3, stride=stride, padding=padding, bias=False)
class ResidualBasicBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int,
        activation_function: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        dropout_rate: Union[None, float] = None,
    ):
        super(ResidualBasicBlock, self).__init__()
        self.stride = stride
        self.process = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, padding=(kernel_size - 1)//2, stride=1, bias=False),
            conv3x3(planes, planes, 1),
            norm_layer(planes),
            activation_function(),
        )   
        self.merge = nn.Sequential(
            conv1x1(inplanes + planes, planes),
            norm_layer(planes)
        )
        if dropout_rate is None:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout2d(dropout_rate)
        self.downscale = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size, padding=(kernel_size - 1)//2, stride=stride, bias=False),
            norm_layer(planes),
            activation_function(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.process(x)
        out = self.merge(torch.cat([x, out], dim=1))
        out = self.dropout(out)
        out = self.downscale(out)
        return out
class UnetEncoder(nn.Module):
    def __init__(
        self,
        block: Union[ResidualBasicBlock],
        layers: List[int],
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
        inplane_base: Optional[int] = 64,
        dropout_rate: Union[None, float] = 0.15,
    ):
        super(UnetEncoder, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer
        self.block = nn.Sequential(
            nn.Conv2d(3, inplane_base, 7, stride=1, padding=3),
            norm_layer(inplane_base),
            self.activ(),
        )
        self.layer1 = self.__make_layer(block, 5, inplane_base, layers[0], dropout_rate=dropout_rate)
        self.bottle1 = conv1x1(inplane_base, inplane_base)
        self.down1 = block(5, inplane_base, inplane_base*2, 2, self.activ, self.norm_layer)
        
        self.layer2 = self.__make_layer(block, 3, inplane_base*2, layers[1], dropout_rate=dropout_rate)
        self.bottle2 = conv1x1(inplane_base*2, inplane_base*2)
        self.down2 = block(3, inplane_base*2, inplane_base*4, 2, self.activ, self.norm_layer)
        
        self.layer3 = self.__make_layer(block, 3, inplane_base*4, layers[2], dropout_rate=dropout_rate)
        self.bottle3 = conv1x1(inplane_base*4, inplane_base*4)
        self.down3 = block(3, inplane_base*4, inplane_base*8, 2, self.activ, self.norm_layer)
        
        self.layer4 = self.__make_layer(block, 3, inplane_base*8, layers[3], dropout_rate=dropout_rate)
        self.bottle4 = conv1x1(inplane_base*8, inplane_base*8)
        self.down4 = block(3, inplane_base*8, inplane_base*8, 2, self.activ, self.norm_layer)
    def __make_layer(
        self,
        block: Union[ResidualBasicBlock],
        kernel_size: int,
        planes: int,
        blocks: int,
        dropout_rate: Union[None, float] = None,
    ):
        layers = []
        for i in range(blocks):
            layers.append(block(kernel_size, planes, planes, 1, self.activ, self.norm_layer, dropout_rate=dropout_rate))
        return nn.ModuleList(layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = {'x': x}
        
        out = self.block(x)
        outs['block'] = out
        
        out = self.__merge(out, self.layer1, self.bottle1)
        out = self.down1(out)
        outs['down1'] = out
        
        out = self.__merge(out, self.layer2, self.bottle2)
        out = self.down2(out)
        outs['down2'] = out
        
        out = self.__merge(out, self.layer3, self.bottle3)
        out = self.down3(out)
        outs['down3'] = out
        
        out = self.__merge(out, self.layer4, self.bottle4)
        out = self.down4(out)
        outs['down4'] = out
        
        return outs
    def __merge(self, x: torch.Tensor, layers: nn.ModuleList, bottle: nn.Module):
        out = 0.
        for l in layers:
            x = l(x)
            out = out + x
        out = bottle(out)
        return out
class Fusion(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
    ):
        super(Fusion, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.norm_x = norm_layer(inplanes)
        self.norm_y = norm_layer(inplanes)
        self.bottle = conv1x1(inplanes, planes)
        self.activ = activation_function()
        self.apply_noise = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, :self.inplanes, ...]
        b = x[:, self.inplanes:, ...]
        out = self.norm_x(a) + self.norm_y(b)
        out = self.bottle(out)
        out = self.activ(out)
        return out
class UnetDecoder(nn.Module):
    def __init__(
        self,
        block: Union[ResidualBasicBlock],
        layers: List[int],
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
        inplane_base: Optional[int] = 64,
        dropout_rate: Optional[Union[None, float]] = 0.15,
    ):
        super(UnetDecoder, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer
        
        self.up1 = self.__make_upsample_block(block, 8, inplane_base, scale=1)
        self.layer1 = self.__make_layer(block, 3, inplane_base*8, layers[0])
        self.bottle1 = conv1x1(inplane_base*8, inplane_base*8)
        self.fusion1 = Fusion(inplane_base*8, inplane_base*8, norm_layer, activation_function)
        
        self.up2 = self.__make_upsample_block(block, 8, inplane_base)
        self.layer2 = self.__make_layer(block, 3, inplane_base*4, layers[1])
        self.bottle2 = conv1x1(inplane_base*4, inplane_base*4)
        self.fusion2 = Fusion(inplane_base*4, inplane_base*4, norm_layer, activation_function)
        
        self.up3 = self.__make_upsample_block(block, 4, inplane_base)
        self.layer3 = self.__make_layer(block, 3, inplane_base*2, layers[2])
        self.bottle3 = conv1x1(inplane_base*2, inplane_base*2)
        self.fusion3 = Fusion(inplane_base*2, inplane_base*2, norm_layer, activation_function)

        self.up4 = self.__make_upsample_block(block, 2, inplane_base)
        self.layer4 = self.__make_layer(block, 5, inplane_base, layers[3])
        self.bottle4 = conv1x1(inplane_base, inplane_base)
        self.fusion4 = Fusion(inplane_base, inplane_base, norm_layer, activation_function)
        
        self.merged_block = nn.ModuleDict({
            'unet_layer': block(3, inplane_base, inplane_base, 1, self.activ, self.norm_layer),
            'merged_layer': block(3, inplane_base, inplane_base, 1, self.activ, self.norm_layer),
        })
        
        self.output_block = nn.Sequential(
            conv1x1(inplane_base, inplane_base),
            conv1x1(inplane_base, 3),
            nn.Tanh(),
        )
    def forward(self, inp) -> torch.Tensor:
        x, block, down1, down2, down3, down4 = inp['x'], inp['block'], inp['down1'], inp['down2'], inp['down3'], inp['down4']
        down_output = self.__merge(self.up1(down4), self.layer1, self.bottle1)

        out = self.fusion1(torch.cat([down3, down_output], dim=1))
        
        down_output = self.__merge(self.up2(out), self.layer2, self.bottle2)
        out = self.fusion2(torch.cat([down2, down_output], dim=1))
        
        down_output = self.__merge(self.up3(out), self.layer3, self.bottle3)
        out = self.fusion3(torch.cat([down1, down_output], dim=1))
        
        down_output = self.__merge(self.up4(out), self.layer4, self.bottle4)
        out = self.fusion4(torch.cat([block, down_output], dim=1))
        
        out = self.merged_block['unet_layer'](out)
        out = self.merged_block['merged_layer'](out)
        out = self.output_block(out)
        return out
    def __merge(self, x: torch.Tensor, layers: nn.ModuleList, bottle: nn.Module):
        out = 0.
        for l in layers:
            x = l(x)
            out = out + x
        out = bottle(out)
        return out
    def __make_upsample_block(
        self,
        block,
        f: int,
        inplane_base: int,
        dropout_rate: Union[None, float] = None,
        scale: int = 2,
        scale_factor: int = 2,
    ):
        inplanes = f*inplane_base
        if scale_factor > 1:
            return nn.Sequential(
                block(3, inplanes, inplanes//scale, 1, self.activ, self.norm_layer, dropout_rate=dropout_rate),
                nn.Upsample(scale_factor=scale_factor),
            )
        else:
            return nn.Sequential(
                block(3, inplanes, inplanes//scale, 1, self.activ, self.norm_layer, dropout_rate=dropout_rate),
            )
    def __make_layer(
        self,
        block: Union[ResidualBasicBlock],
        kernel_size: int,
        planes: int,
        blocks: int,
        dropout_rate: Union[None, float] = None,
    ):
        layers = []
        for i in range(blocks):
            layers.append(block(kernel_size, planes, planes, 1, self.activ, self.norm_layer, dropout_rate=dropout_rate))
        return nn.ModuleList(layers)
def custom_leaky_relu(rate: float = 0.):
    return partial(nn.LeakyReLU, negative_slope=rate)
class Unet(nn.Module):
    def __init__(
        self,
        encoder: Callable[..., nn.Module],
        decoder: Callable[..., nn.Module],
        **kwargs,
    ):
        super(Unet, self).__init__()
        
        self.args = {
            'inplane_base': 64,
            'encoder_block': ResidualBasicBlock,
            'encoder_layers': [4, 3, 2, 1],
            'encoder_norm_layer': nn.BatchNorm2d,
            'encoder_activation_function': nn.ReLU,
            'encoder_dropout_rate': 0.15, 
            'decoder_block': ResidualBasicBlock,
            'decoder_layers': [1, 2, 3, 4],
            'decoder_norm_layer': nn.BatchNorm2d,
            'decoder_activation_function': nn.ReLU,
            'decoder_dropout_rate': 0.15, 
        }
        self.args.update(kwargs)
        self.encoder = encoder(
            self.args['encoder_block'], 
            self.args['encoder_layers'], 
            self.args['encoder_norm_layer'], 
            self.args['encoder_activation_function'],
            inplane_base=self.args['inplane_base'],
            dropout_rate=self.args['encoder_dropout_rate'],
        )
        self.decoder = decoder(
            self.args['decoder_block'], 
            self.args['decoder_layers'], 
            self.args['decoder_norm_layer'], 
            self.args['decoder_activation_function'],
            inplane_base=self.args['inplane_base'],
            dropout_rate=self.args['decoder_dropout_rate'],
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        y = self.decoder(out)
        return y
class ResidualLayer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        planes: int,
        depth: int,
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
        dropout_rate: float,
    ):
        super(ResidualLayer, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer
        self.dropout_rate = dropout_rate
        self.__make_layer(kernel_size, planes, depth)
    def __make_layer(
        self,
        kernel_size: int,
        planes: int,
        depth: int,
    ):
        self.inner_layers = nn.ModuleList()
        for i in range(depth):
            self.inner_layers.append(
                ResidualBasicBlock(
                    kernel_size,
                    planes, 
                    planes,
                    1,
                    self.activ,
                    self.norm_layer,
                    dropout_rate=self.dropout_rate,
                )
            )
    def forward(self, x: torch.Tensor):
        out = x
        for module in self.inner_layers:
            out = module(out)
        return out
class PatchCritic(nn.Module):
    def __init__(
        self,
        layer_block,
        layer_depth: List[int],
        activation_function: Callable[..., nn.Module],
        norm_layer: nn.BatchNorm2d,
        inplane_base: Optional[int] = 8, 
        out_inplanes: Optional[int] = 1, 
        dropout_rate: Optional[Union[None, float]] = 0.15,
        noise_params={'mean': 0., 'stddev': 0.15},
    ):
        super(PatchCritic, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer
        self.noise_params = noise_params
        self.block = nn.Sequential(
            nn.Conv2d(6, inplane_base, 7, stride=2, padding=3),
            norm_layer(inplane_base),
            self.activ(),
        )
        
        self.layer1 = layer_block(5, inplane_base, layer_depth[0], norm_layer, activation_function, dropout_rate)
        self.down1 = nn.Sequential(
            nn.Conv2d(inplane_base, inplane_base*2, 5, stride=2, padding=2, bias=False),
            norm_layer(inplane_base*2),
            self.activ(),
        )
        
        self.layer2 = layer_block(5, inplane_base*2, layer_depth[1], norm_layer, activation_function, dropout_rate)
        self.down2 = nn.Sequential(
            nn.Conv2d(inplane_base*2, inplane_base*4, 5, stride=2, padding=2, bias=False),
            norm_layer(inplane_base*4),
            self.activ(),
        )
        
        self.layer3 = layer_block(3, inplane_base*4, layer_depth[2], norm_layer, activation_function, dropout_rate)
        self.down3 = nn.Sequential(
            nn.Conv2d(inplane_base*4, inplane_base*8, 3, stride=2, padding=1, bias=False),
            norm_layer(inplane_base*8),
            self.activ(),
        )
        
        self.layer4 = layer_block(3, inplane_base*8, layer_depth[3], norm_layer, activation_function, dropout_rate)
        self.down4 = nn.Sequential(
            nn.Conv2d(inplane_base*8, inplane_base*8, 3, stride=2, padding=1, bias=False),
            norm_layer(inplane_base*8),
            conv3x3(inplane_base*8, inplane_base*8, 1),
        )
        
        self.squeeze = conv1x1(inplane_base*8, out_inplanes)
        self.layers = {
            'block': self.block,
            'layer1': self.layer1,
            'down1': self.down1,
            'layer2': self.layer2,
            'down2': self.down2,
            'layer3': self.layer3,
            'down3': self.down3,
            'layer4': self.layer4,
            'down4': self.down4,
        }
    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        out = x
        if self.training:
            noise = torch.normal(self.noise_params['mean'], self.noise_params['stddev'], out.size()).to(out.device)
            out = out + noise
        for name, layer in self.layers.items():
            out = layer(out)
        out = self.squeeze(out)
        return out