# from mmflow.apis import inference_model, init_model
import math
import os
import pickle
import random
import sys
from glob import glob

import clip
import cv2
import numpy as np
import PIL.Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (ApplyTransformToKey, RandomResizedCrop,
                                     ShortSideScale, UniformCropVideo,
                                     UniformTemporalSubsample)
from torch import Tensor, nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (CenterCropVideo,
                                                      NormalizeVideo)

from gen_gif import make_gif
from torch_utils import training_stats

print(torch.__version__)
import time
from collections import OrderedDict, namedtuple
from itertools import chain

import lpips
import torchvision.transforms.functional as TF
from torch.nn import (AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d,
                      Dropout, Linear, MaxPool2d, Module, PReLU, ReLU,
                      Sequential, Sigmoid)


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m
    
class Backbone(Module):
	def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
		super(Backbone, self).__init__()
		assert input_size in [112, 224], "input_size should be 112 or 224"
		assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
		assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
		blocks = get_blocks(num_layers)
		if mode == 'ir':
			unit_module = bottleneck_IR
		elif mode == 'ir_se':
			unit_module = bottleneck_IR_SE
		self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  BatchNorm2d(64),
									  PReLU(64))
		if input_size == 112:
			self.output_layer = Sequential(BatchNorm2d(512),
			                               Dropout(drop_ratio),
			                               Flatten(),
			                               Linear(512 * 7 * 7, 512),
			                               BatchNorm1d(512, affine=affine))
		else:
			self.output_layer = Sequential(BatchNorm2d(512),
			                               Dropout(drop_ratio),
			                               Flatten(),
			                               Linear(512 * 14 * 14, 512),
			                               BatchNorm1d(512, affine=affine))

		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = Sequential(*modules)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.body(x)
		x = self.output_layer(x)
		return l2_norm(x)

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class GradualStyleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x) # [b,512,H,W]->[b,512,1,1]
        # (H,W) in [(16,16),(32,32),(64,64)]
        x = x.view(-1, self.out_c) # [b,512,1,1]-> [b,512]
        x = self.linear(x)
        x = nn.LeakyReLU()(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, spatial, **styleblock):
        super(TransformerBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.num_encoder_layers = styleblock['num_encoder_layers']
        num_pools = int(np.log2(spatial))-4
        modules = []
#        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
#                    nn.LeakyReLU()]
        for i in range(num_pools-1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.positional_encoding = PositionalEncoding(out_c)
        self.transformer_encoder = nn.Transformer(num_encoder_layers=self.num_encoder_layers).encoder
#(out_c, out_c)

    def forward(self, x):
        x = self.convs(x) # [b,512,H,W]->[b,512,16,16]
        # (H,W) in [(16,16),(32,32),(64,64)]
        x = x.view(x.shape[0], -1, self.out_c) # [b,256,512]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)[:,0,:]
        return x


class GradualStyleEncoder(torch.nn.Module):
    def __init__(self, **styleblock):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(50) # num_layers=50
        unit_module = bottleneck_IR_SE # 'ir_se' bottleneck

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        ) # [b,3,256,256]->[b,3,64,64]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList() # feat->latent

        # TODO: 
        # need some other method for handling w[0]
            # train w[0] separately ?
        # coarse_ind, middle_ind tuning
        self.style_count = 16
        self.coarse_ind = 3
        self.middle_ind = 7
        # print(styleblock)
        # styleblock = dict(arch='transformer', num_encoder_layers=1)
        if 'arch' in styleblock:
            for i in range(self.style_count):
                if i < self.coarse_ind:
                    style = TransformerBlock(512, 512, 16, **styleblock)
                elif i < self.middle_ind:
                    style = TransformerBlock(512, 512, 32, **styleblock)
                else:
                    style = TransformerBlock(512, 512, 64, **styleblock)
                self.styles.append(style)
        else:
            for i in range(self.style_count):
                if i < self.coarse_ind:
                    style = GradualStyleBlock(512, 512, 16)
                elif i < self.middle_ind:
                    style = GradualStyleBlock(512, 512, 32)
                else:
                    style = GradualStyleBlock(512, 512, 64)
                self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x) # [b,3,256,256]->[b,64,256,256]

        latents = []
        modulelist = list(self.body._modules.values())

        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x # [b,128,64,64]
            elif i == 20:
                c2 = x # [b,256,32,32]
            elif i == 23:
                c3 = x # [b,512,16,16]

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2)) # [b,512,32,32]
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1)) # [b,512,64,64]
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
#       Adding Transformer structure in this results in gradient exploding
#        if self.neck is not None:
#            out = self.neck(out)
        return out

class Encoder(torch.nn.Module):
    """stylegan3 encoder implementation
    based on pixel2sylte2pixel GradualStyleEncoder
    (b, 3, 256, 256) -> (b, 16, 512)
    stylegan3 generator synthesis
    (b, 16, 512) -> (b, 3, 1024, 1024)
    """
    def __init__(
        self,
        pretrained=None,
        w_avg=None,
        **kwargs, 
    ):
        super(Encoder, self).__init__()
        self.encoder = GradualStyleEncoder(**kwargs) # 50, irse
        self.resume_step = 0
        self.w_avg = w_avg

        # load weight
        if pretrained is not None:
            with open(pretrained, 'rb') as f:
                dic = pickle.load(f)
            weights = dic['E']
            weights_ = dict()
            for layer in weights:
                if 'module.encoder' in layer:
                    weights_['.'.join(layer.split('.')[2:])] = weights[layer]
            self.resume_step = dic['step']
            self.encoder.load_state_dict(weights_, strict=True)
            del weights
        else:
            irse50 = torch.load("./pretrained_models/model_ir_se50.pth", map_location='cpu')
            weights = {k:v for k,v in irse50.items() if "input_layer" not in k}
            self.encoder.load_state_dict(weights, strict=False)

    def forward(self, img):
        if self.w_avg is None:
            return self.encoder(img)
        else: # train delta_w, from w_avg
            delta_w = self.encoder(img)
            w = delta_w + self.w_avg.repeat(delta_w.shape[0],1,1)
            return w

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator2D(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator2D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 2) x 64 x 64
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 4) x 32 x 32
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 8) x 16 x 16
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            
            # state size (ndf * 8) x 8 x 8
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.25),
            # state size (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output 

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32, T=5, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x T x 256 x 256
        nn.Conv3d(nc, ndf, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf) x T x 128 x 128
        nn.Conv3d(ndf, ndf * 2, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf*2) x T x 64 x 64
        nn.Conv3d(ndf * 2, ndf * 4, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf*4) x T x 32 x 32
        nn.Conv3d(ndf * 4, ndf * 8, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),


        nn.Conv3d(ndf * 8, ndf * 8, (1, 1, 1), 1, 0),

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            # output = self.fc(output)

        # return output.view(-1, 1).squeeze(1)
        return output

class SceneVideoDataset(Dataset):
    def __init__(self):
        self.video_paths = glob("./datasets/scene_video/*/*.mp4")
        self.audio_paths = glob("./datasets/curation/dataset_curation/*.npy")
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 6
        self.sampling_rate = 2
        self.frames_per_second = 1
        self.alpha = 4
        self.time_length = 864
        self.n_mels = 128 * 2
        self.width_resolution = 512 // 2
        self.frame_per_audio = self.time_length // self.num_frames
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    
                    # NormalizeVideo(self.mean, std),
                    ShortSideScale(
                    size=self.side_size
                    ),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
                    Lambda(lambda x: x / 127.5 - 1.0),
                    # RandomResizedCrop(target_height=1024, target_width=1024, scale=0.25, aspect_ratio=1.0)
                ]
            ),
        )

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            
            video = EncodedVideo.from_path(video_path)
            clip_start_sec = 0.0 # secs
            clip_duration = 2.0 # secs
            video_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration)
            video_data = self.transform(video_data)
            '''audio data'''
            file_name = video_path.split("/")[-1].split(".")[0]
            file_name = os.path.join('./datasets/curation/dataset_curation', file_name)
            file_name = file_name + '.npy'
            index = self.audio_paths.index(file_name)
            npy_name = self.audio_paths[index]
            audio_inputs = np.load(npy_name, allow_pickle=True)
            c, h, w = audio_inputs.shape
            if w >= self.time_length:
                j = random.randint(0, w-self.time_length)
                full_audio = audio_inputs[:,:,j:j+self.time_length]
            elif w < self.time_length:
                zero = np.zeros((1, self.n_mels, self.time_length))
                j = random.randint(0, self.time_length - w - 1)
                zero[:,:,j:j+w] = audio_inputs[:,:,:w]
                full_audio = zero
            full_audio = cv2.resize(full_audio[0], (self.n_mels, self.width_resolution))
            full_audio = torch.from_numpy(full_audio).float()

            text = video_path.split("/")[3]


            return video_data["video"], full_audio, text

        except Exception as e:
            print("Wo ", e)
            return self.__getitem__((idx+1) % len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)

class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x    

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)  

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

class CLIP(object):
  def __init__(self, device):
    clip_model = "ViT-B/32"
    self.model, _ = clip.load(clip_model)
    self.model = self.model.requires_grad_(False).to(device)
    self.device = device

    self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                          std=[0.26862954, 0.26130258, 0.27577711])

  @torch.no_grad()
  def embed_text(self, prompt):
      "Normalized clip text embedding."
      return norm1(self.model.encode_text(clip.tokenize(prompt).to(self.device)).float())

  def embed_cutout(self, image):
      "Normalized clip image embedding."
      # return norm1(self.model.encode_image(self.normalize(image)))
      return norm1(self.model.encode_image(image))

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class CLIPLoss(torch.nn.Module):

    def __init__(self, device):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=256 // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity ** 2

class LatentRNN(nn.Module):
    def __init__(self, gpu, type = "coarse", hidden_size=512, num_layers=5, dropout=0.0, bidrectional=True):
        super(LatentRNN, self).__init__()
        #self.device = device
        self.type = type
        self.num_layers = num_layers
        
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, dropout=0.3, num_layers=self.num_layers, batch_first=True)
        if self.type=="coarse":
            input_size = 4 * 512
        if self.type=="mid":
            input_size = 4 * 512
        if self.type=="fine":
            input_size = 8 * 512

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, "GRU")(input_size, self.hidden_size, self.num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, input_size)
        self.args_gpu = gpu

        
    def forward(self, x, h):
        out, h1 = self.rnn(x, h.view(2 * self.num_layers, -1, self.hidden_size))
        out = x + self.fc(out)
        return out, h1

def subprocess_fn(rank, num_gpus, timestring):
    torch.distributed.init_process_group(backend='nccl',  rank=rank, world_size=num_gpus)

    # Init torch_utils
    torch.cuda.set_device(rank)
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    make_cutouts = MakeCutouts(224, 32, 0.5)
    dataset = SceneVideoDataset()

    print(len(dataset))

    device = torch.device('cuda', rank)
    print('Using device:', device, file=sys.stderr)

    model_url = "./pretrained_models/lhq-256.pkl"
    # model_url = "./pretrained_models/stylegan3-r-afhqv2-512x512.pkl"
    # model_url = "./pretrained_models/stylegan3-r-urmp-256x256.pkl"
    # model_url = "./pretrained_models/stylegan3-r-ffhq-1024x1024.pkl"
    # model_url = "./pretrained_models/stylegan3-r-ffhqu-256x256.pkl"
    
    # mapper = DDP(LevelsMapper().to(device), device_ids=[rank])

    with open(model_url, 'rb') as fp:
        G = pickle.load(fp)['G_ema'].to(device)

    styleblock = dict(arch='transformer', num_encoder_layers=2)
    sound_inversion = DDP(Encoder(w_avg=G.mapping.w_avg, **styleblock).to(device), device_ids=[rank])
    sound_inversion.train()

    def embed_image(image):
        n = image.shape[0]
        cutouts = make_cutouts(image)
        embeds = clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds


    # clip_model = CLIP()
    
    print("Model Load!")
    # torch.set_num_threads(8)
    print(torch.get_num_threads())
    # 1 x 512
    z = torch.randn([5, G.z_dim]).to(device)
    c = None
    w = G.mapping(z, c, truncation_psi=0.7)
    print(w.size())

    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    w_stds = G.mapping(zs, None).std(0)

    num_epochs = 40000
    sequence_length = 6
    batch_size = 1

    iteration = 40000
    cnt_iteration = 0
    train_size = int(0.95 * len(dataset))

    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0)

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    # lr = 1e-5
    lr = 0.0001
    lr = 1e-4
    # lr = 1e-2
    betas=(0.5, 0.999)


    criterion_l2 = nn.MSELoss()
    criterion_video = nn.MSELoss()
    criterion_dis = nn.MSELoss()
    criterion_g = nn.MSELoss()
    criterion_2g = nn.MSELoss()
    criterion_cos = nn.CosineEmbeddingLoss()
    label = torch.FloatTensor().to(device)
    clip_loss = CLIPLoss(device)

    clip_model = CLIP(device)


    G.eval()

    former_coarse = DDP(LatentRNN(device, type="coarse").to(device), device_ids=[rank])
    former_mid = DDP(LatentRNN(device, type="mid").to(device), device_ids=[rank])
    former_fine = DDP(LatentRNN(device, type="fine").to(device), device_ids=[rank])


    discriminator_3d = Discriminator(T=sequence_length).to(device)
    discriminator_2d = Discriminator2D().to(device)

    discriminator_2d.train()
    discriminator_3d.train()
    former_coarse.train()
    former_mid.train()
    former_fine.train()
    sound_inversion.train()

    optimizer_3d = optim.AdamW(discriminator_3d.parameters(), lr=lr)
    optimizer_2d = optim.AdamW(discriminator_2d.parameters(), lr=lr)
    optimizer = optim.AdamW(chain(former_coarse.parameters(), former_fine.parameters(), former_mid.parameters()), lr=lr)
    optimizer_inver = optim.AdamW(sound_inversion.parameters(), lr=lr * 5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

    shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
    G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
    G.synthesis.input.affine.weight.data.zero_()
    
    os.makedirs(f'samples/{timestring}', exist_ok=True)
    loader = iter(traindataloader)
    for epoch in range(num_epochs):

        for batchidx, (batchvideo, batchaudio, batchtext) in enumerate(traindataloader):

            batchrecon, batchrecon_s = [], []
            batchvideo = batchvideo.to(device)
            reg_loss = 0
            av_loss = 0
            l2_loss = 0
            video_l2_loss = 0
            text_contrastive_loss = 0
            
            
            audio_input = torch.cat([batchaudio.unsqueeze(0), batchaudio.unsqueeze(0), batchaudio.unsqueeze(0)], axis=1).to(device)
            for i in range(batch_size):
                
                w_s = sound_inversion(audio_input[i].unsqueeze(0))

                texts = [frase.strip() for frase in batchtext[i].split("|") if frase]
                targets = [clip_model.embed_text(text) for text in texts]

                
                w0 = w.clone().detach()
                w_list, w_s_list = [], []

                reg_loss += criterion_l2(w_s, w0)
            
                with torch.no_grad():
                    targets = torch.cat([clip.tokenize(batchtext)]).to(device)

                h_s_coarse = torch.zeros(1, 5 * 2, 512).to(device).requires_grad_()
                h_s_mid = torch.zeros(1, 5 * 2, 512).to(device).requires_grad_()
                h_s_fine = torch.zeros(1, 5 * 2, 512).to(device).requires_grad_()

                for s in range(sequence_length):
                    w_s_list.append(w_s)

                    w_s_coarse, h_s_coarse = former_coarse(w_s.view(-1, 16, 512)[:,:4,:].view(-1, 1, 4 * 512), h_s_coarse)
                    w_s_mid, h_s_mid = former_mid(w_s.view(-1, 16, 512)[:,4:8,:].view(-1, 1, 4 * 512), h_s_mid)
                    w_s_fine, h_s_fine = former_fine(w_s.view(-1, 16, 512)[:,8:,:].view(-1, 1, 8 * 512), h_s_fine)
                
                    w_s_next = torch.cat([w_s_coarse.view(1, -1, 512), w_s_mid.view(1, -1, 512), w_s_fine.view(1, -1, 512)], axis=1).view(-1, 16, 512) # + 0.05 * w_s # - G.mapping.w_avg
                    
                    w_s = w_s_next
                    
                    l2_loss += criterion_l2(w_s_next, w0)
                
                text_contrastive_loss += 0.001 * clip_loss(G.synthesis(w_s_next.view(-1, 16, 512), noise_mode='const', force_fp32=True).add(1).div(2), targets).mean()
                w_s_output = torch.cat(w_s_list, axis=0)
                

                sound_image = G.synthesis(w_s_output.view(-1, 16, 512), noise_mode='const', force_fp32=True).unsqueeze(0)

                batchrecon_s.append(sound_image)

            batchrecon = torch.cat(batchrecon_s, axis=0)
            batchvideo = batchvideo.permute(0, 2, 1, 3, 4)

            # 3d Discriminator
            optimizer_3d.zero_grad()

            fake_output_3d = discriminator_3d(batchrecon.clone().detach().permute(0, 2, 1, 3, 4))
            real_output_3d = discriminator_3d(batchvideo[:,:,:,:,:].permute(0, 2, 1, 3, 4))
            
            label_fake_3d = torch.zeros_like(real_output_3d, requires_grad=False)
            label_real_3d = torch.ones_like(real_output_3d, requires_grad=False)
            d_loss = 0.5 * criterion_dis(real_output_3d, label_real_3d) + 0.5 * criterion_dis(fake_output_3d, label_fake_3d) 
            d_loss.backward(retain_graph=True)
            optimizer_3d.step()

            optimizer_2d.zero_grad()
            fake_output_2d = discriminator_2d(batchrecon.clone().detach().view(-1, 3, 256, 256))
            real_output_2d = discriminator_2d(batchvideo[:,:,:,:,:].view(-1, 3, 256, 256))
            
            label_fake_2d = torch.zeros_like(real_output_2d, requires_grad=False)
            label_real_2d = torch.ones_like(real_output_2d, requires_grad=False)
            d_loss = 0.5 * criterion_dis(real_output_2d, label_real_2d) + 0.5 * criterion_dis(fake_output_2d, label_fake_2d) 
            d_loss.backward(retain_graph=True)
            optimizer_2d.step()
            
            
            optimizer.zero_grad()
            optimizer_inver.zero_grad()

            fake_output_3d = discriminator_3d(batchrecon.permute(0, 2, 1, 3, 4))

            fake_output_2d = discriminator_2d(batchrecon.view(-1, 3, 256, 256))
            g_loss = criterion_g(fake_output_3d, label_real_3d) + criterion_2g(fake_output_2d, label_real_2d)
            
            loss = g_loss + reg_loss + l2_loss + text_contrastive_loss + criterion_video(batchrecon_s, batchrecon.detach()).mean() # + criterion_video(batchrecon, batchvideo.clone().detach()).mean() # + video_l2_loss # +  + # + 1e-1 * vgg_loss

            loss.backward()

            optimizer.step()
            optimizer_inver.step()

            torch.distributed.barrier()
            
            cnt_iteration += 1
            print(f"[epoch : {epoch}] [{batchidx} / {len(traindataloader)}] d_loss : {d_loss.item():.3f} total_g_loss : {loss.item():.3f} / min : {torch.min(batchrecon): .2f}, max : {torch.max(batchrecon) : .2f}")
            if cnt_iteration % 50 == 0 and rank==0:
                print("Model Save !")
                save_path = "./pretrained_models/coarseformer_.pth"
                torch.save(former_coarse.state_dict(), save_path)
                save_path = "./pretrained_models/midformer_.pth"
                torch.save(former_mid.state_dict(), save_path)
                save_path = "./pretrained_models/fineformer_.pth"
                torch.save(former_fine.state_dict(), save_path)
                save_path = f"./pretrained_models/audio_inversion_.pth"
                torch.save(sound_inversion.state_dict(), save_path)

            if cnt_iteration % 10 == 0 and rank==0:
                save_real = batchvideo[0]
                save_fake = batchrecon[0]
                save_s = batchrecon_s[0]
                for s in range(sequence_length):

                    pil_image = TF.to_pil_image(save_real[s].add(1).div(2).clamp(0,1).cpu())
                    pil_image.save(f'samples/{timestring}/real_000{s}.jpg')

                    pil_image = TF.to_pil_image(save_fake[s].add(1).div(2).clamp(0,1).cpu())
                    pil_image.save(f'samples/{timestring}/fake_000{s}.jpg')

                    pil_image = TF.to_pil_image(save_s[s].add(1).div(2).clamp(0,1).cpu())
                    pil_image.save(f'samples/{timestring}/sound_000{s}.jpg')


        scheduler.step()

if __name__ == '__main__': 
    # timestring = time.strftime('%Y%m%d%H%M%S')
    timestring = "video-generation"
    os.makedirs(f'samples/{timestring}', exist_ok=True)

    random_seed = 32
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    num_gpus=2
    torch.multiprocessing.spawn(fn=subprocess_fn, args=(num_gpus, timestring), nprocs=num_gpus)
