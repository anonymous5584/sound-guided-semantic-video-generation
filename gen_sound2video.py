import sys
import timm
import librosa
import io
import os, time, glob
import pickle
import random
import shutil
from unittest.loader import VALID_MODULE_NAME
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
import unicodedata
import re
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from einops import rearrange
from collections import OrderedDict
import torch.nn as nn
from typing import List, Optional, Tuple
from moviepy.editor import *
import soundfile as sf
import math

# from mmpt.models import MMPTModel
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
# seed = 45
# random.seed(seed)
from collections import namedtuple
import cv2
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module

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
    
def trim_audio_data(audio_file, save_file):
    sr = 44100
    sec = 4
    y, sr = librosa.load(audio_file, sr=sr)
    ny = y[:sr*sec]
    sf.write(save_file+'.wav', ny, sr, 'PCM_24')
    # librosa.output.write_wav(save_file + '.wav', ny, sr)
    
def img2mp4(paths, pathOut , fps=30):
    print(paths)
    clips = [ImageClip(m).set_duration(1/30.0) for m in paths]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(pathOut, fps=fps)    
    print("img2mp4 finish")
    
def generate_full_video(video_path, timestring):
    videoclip = VideoFileClip(video_path)
    # audioclip = AudioFileClip("./4K_Fresh_Waterfall__Natural_White_Noise_Sounds__Flowing_Water__10_Hours__Relaxation_Sleep_Video_0.wav")
    trim_audio_data("./datasets/audio_file/thunder/Storm_Ocean_Sky_Rain_The_Waves_Wind_Dark_Clouds_7.wav", "./datasets/audio_file/thunder/Storm_Ocean_Sky_Rain_The_Waves_Wind_Dark_Clouds_7.wav_3_sec")
    # trim_audio_data("./datasets/audio_file/squishing_water/_Soothing_Sounds_of_a_Beautiful_Mountain_River__Awesome_Nature_Landscape_Screensaver_12_HOURS__1.wav", "./tmp.wav_3_sec")
    # audioclip = AudioFileClip("./1_hour_video_of_big_ocean_waves_crashing_into_sea_cliffs___HD_1080P_0_3_sec.wav")
    audioclip = AudioFileClip( "./datasets/audio_file/thunder/Storm_Ocean_Sky_Rain_The_Waves_Wind_Dark_Clouds_7.wav_3_sec.wav")
    # trim_audio_data("./4K_Fresh_Waterfall__Natural_White_Noise_Sounds__Flowing_Water__10_Hours__Relaxation_Sleep_Video_0.wav", "4K_Fresh_Waterfall__Natural_White_Noise_Sounds__Flowing_Water__10_Hours__Relaxation_Sleep_Video_0_3_sec")
    # audioclip = AudioFileClip("./4K_Fresh_Waterfall__Natural_White_Noise_Sounds__Flowing_Water__10_Hours__Relaxation_Sleep_Video_0_3_sec.wav")
    # audioclip = AudioFileClip("./1_hour_video_of_big_ocean_waves_crashing_into_sea_cliffs___HD_1080P_0_3_sec.wav")
    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile(f"./samples/{timestring}/base.mp4")
    print("generate_full_video finish")
    
def generate_gif(path, n):
    img_list = os.listdir(path)
    img_list.sort()
    img_list = [path + '/' + x for x in img_list]
    print(img_list)
    images = [Image.open(x) for x in img_list]
    
    im = images[0]
    im.save(f"./samples/{timestring}/{n}.gif", save_all=True, append_images=images[1:],loop=0xff, duration=100)
    

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


class LatentRNN(nn.Module):
    def __init__(self, gpu, type = "coarse", hidden_size=512, num_layers=5, dropout=0.0, bidrectional=True):
        super(LatentRNN, self).__init__()
        #self.device = device
        self.type = type
        # self.hidden_size = hidden_size
        
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
        self.fc = nn.Linear(self.hidden_size * 2, input_size)
        # self.fc = LevelsMapper().to(gpu)
        self.audio_noise_fc = nn.Linear(5120, 5120)   # frame_num * embedding dim
        self.tanh = nn.Tanh()
        self.args_gpu = gpu
        self.weight_noise = 1.0
        # self.alpha = torch.nn.Parameter(torch.ones(8192))
        # self.audio_encoder = audio_encoder
        # self.audio_encoder = self.audio_encoder.eval()
        
    def forward(self, x, h):
        # noise = torch.randn(self.num_layers * 2, 1, self.hidden_size).to(self.args_gpu)
        # noise = self.audio_noise_fc(noise + )
        # noise = self.weight_noise * noise 
        # print(self.alpha)
        # audio = audio.unsqueeze(0)
        # audio_noise = torch.cat((noise, audio), axis=0).view(1, -1)
        # audio_noise = self.audio_noise_fc(audio_noise).view(10, -1, 512)
        #h = self.audio_noise_fc(h.view(-1, 5120))
        # out, h1 = self.rnn(x, h.view(10, -1, 512) + audio_noise.view(10, -1, 512))
        # out, h1 = self.rnn(x, h.view(10, -1, 512) + noise)
        # print(h.size())
        h = self.audio_noise_fc(h.view(-1, 5120))
        out, h1 = self.rnn(x, h.view(10, -1, self.hidden_size))
        out = x + self.fc(out)
        #out = x + self.fc(out)
        return out, h1

class CLIP(object):
    def __init__(self):
        clip_model = "ViT-B/32"
        self.model, _ = clip.load(clip_model, device=device)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    @torch.no_grad()
    def embed_text(self, prompt):
        "Normalized clip text embedding."
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())
    def embed_cutout(self, image):
        "Normalized clip image embedding."
        # return norm1(self.model.encode_image(self.normalize(image)))
        return norm1(self.model.encode_image(image))

tf = Compose([
    Resize(224),
    lambda x: torch.clamp((x+1)/2,min=0,max=1),
])

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)

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

make_cutouts = MakeCutouts(224, 32, 0.5)

def embed_image(image):
    n = image.shape[0]
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds

def run(timestring):
    coarse_rnn = LatentRNN(device, type="coarse").to(device)
    mid_rnn = LatentRNN(device, type="mid").to(device)
    fine_rnn = LatentRNN(device, type="fine").to(device) 
    coarse_rnn.load_state_dict(copyStateDict(torch.load("./pretrained_models/coarseformer_.pth", map_location=device)))
    mid_rnn.load_state_dict(copyStateDict(torch.load("./pretrained_models/midformer_.pth", map_location=device)))
    fine_rnn.load_state_dict(copyStateDict(torch.load("./pretrained_models/fineformer_.pth", map_location=device)))

    coarse_rnn.eval()
    mid_rnn.eval()
    fine_rnn.eval()
    print("load rnn")
    
    
    sequence_length = 120

    sound_inversion = Encoder().to(device)
    sound_inversion.load_state_dict(copyStateDict(torch.load("./pretrained_models/audio_inversion_.pth", map_location=device)))
    sound_inversion.eval()

    audio_path = "./audio/ocean.wav"
    y, sr = librosa.load(audio_path, sr=44100)
    n_mels = 128 * 2
    time_length = 864
    resize_resolution = 512 // 2

    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1

    zero = np.zeros((n_mels, time_length))
    h, w = audio_inputs.shape
    if w >= time_length:
        j = (w - time_length) // 2
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        j = (time_length - w) // 2
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero
    
    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    # audio_inputs[:,:] = 0
    audio_inputs = np.array([audio_inputs, audio_inputs, audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 3, n_mels, resize_resolution))).float().to(device)
    
    shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
    G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
    G.synthesis.input.affine.weight.data.zero_()
    
    N = 1
    start_time = time.time()
    for n in range(N):
        os.makedirs(f'samples/{timestring}/{n}', exist_ok=True)
        w = sound_inversion(audio_inputs)
        beta = 0.4
        h_coarse = torch.zeros(1, 5 * 2, 512).to(device)
        h_mid = torch.zeros(1, 5 * 2, 512).to(device)
        h_fine = torch.zeros(1, 5 * 2, 512).to(device)

        for frame in range(sequence_length):

            
            image = G.synthesis(w.view(-1, 16, 512), noise_mode='const', force_fp32=True)
                
            pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1).cpu())
            pil_image.save(f'samples/{timestring}/{n}/{frame:04}.png')

            w_coarse, h_coarse = coarse_rnn(w.view(-1, 16, 512)[:,:4,:].view(-1, 1, 4 * 512), h_coarse)
            w_mid, h_mid = mid_rnn(w.view(-1, 16, 512)[:,4:8,:].view(-1, 1, 4 * 512), h_mid)
            w_fine, h_fine = fine_rnn(w.view(-1, 16, 512)[:,8:,:].view(-1, 1, 8 * 512), h_fine)
            
            w_next = torch.cat([w_coarse.view(1, -1, 512), w_mid.view(1, -1, 512), w_fine.view(1, -1, 512)], axis=1)

            w = beta * w_next + (1 - beta) * w
    
        path = f'samples/{timestring}/{n}'       
        generate_gif(path, n)
    end_time = time.time()
    print(end_time - start_time)

device = torch.device('cuda')
print('Using device:', device, file=sys.stderr)
base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
Model = 'FFHQ' #@param ["FFHQ", "MetFaces", "AFHQv2", "cosplay", "Wikiart", "Landscapes"]
model_name = {
"FFHQ": base_url + "stylegan3-t-ffhqu-1024x1024.pkl",
"MetFaces": base_url + "stylegan3-r-metfacesu-1024x1024.pkl",
"AFHQv2": base_url + "stylegan3-t-afhqv2-512x512.pkl",
"cosplay": "https://l4rz.net/cosplayface-snapshot-stylegan3t-008000.pkl",
"Wikiart": "https://drive.google.com/u/0/open?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj",
"Landscapes": "https://drive.google.com/u/0/open?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1"
}
# model_url = "./pretrained_models/stylegan3-r-afhqv2-512x512.pkl"
model_url = "./pretrained_models/lhq-256.pkl"
# model_url = "./pretrained_models/stylegan3-r-ffhqu-256x256.pkl"
# model_url = "./pretrained_models/lhq-1024.pkl"
# model_url = "pretrained_models/stylegan3-r-ffhq-1024x1024.pkl"
# model_url = "./pretrained_models/stylegan3-r-urmp-256x256.pkl"
# model_url = "./pretrained_models/wikiart.pkl"

with open(model_url, 'rb') as fp:
    G = pickle.load(fp)['G_ema'].to(device)
m = make_transform([0,0], 0)
m = np.linalg.inv(m)
G.synthesis.input.transform.copy_(torch.from_numpy(m))
zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)
texts = "Ocean Wave"

steps = 400
seed = 14
texts = [frase.strip() for frase in texts.split("|") if frase]
clip_model = CLIP()
targets = [clip_model.embed_text(text) for text in texts]
timestring = time.strftime('%Y%m%d%H%M%S')

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore


run(timestring)