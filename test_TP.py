# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from randaugment import RandAugment
import torch.nn.functional as F
import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import umap
from modules.Visual_Prompt import visual_prompt
# from modules.prompters import PadPrompter
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import glob
from tqdm import tqdm
import random
import csv
from thop import profile, clever_format
import time
from flopth import flopth
    
def calculate_Gflops_and_TP(model,device):
    from transformers import CLIPProcessor
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    image_input = torch.randn(8,3, 224, 224).to(device)
    text_input = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # text_input = processor(["a photo of a cat on a sunny day"], return_tensors="pt").to(device)
    # Number of inference passes
    num_inference_passes = 100  # Adjust as needed

    # Warm-up inference
    with torch.no_grad():
        for _ in range(10):  # Warm-up for stability
            outputs = model(image_input, text_input)
            # outputs = model(image_input)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_inference_passes):
            outputs = model(image_input, text_input)
            # outputs = model(image_input)
    end_time = time.time()
    
    # Calculate throughput in IPS
    total_time = end_time - start_time
    throughput_ips = (num_inference_passes / total_time)*8

    print(f"Number of Inference Passes: {num_inference_passes}")
    print(f"Total Time: {total_time} seconds")
    print(f"Throughput: {throughput_ips} IPS")
    
    # Calculate FLOPs
    flops, params = profile(model, inputs=(image_input, text_input))
    # flops, params = profile(model.visual, image_input)
    flops_in_gflops = flops / 1e9

    
    print(f"FLOPs: {flops_in_gflops} GFLOPs")
    print(f"Parameters: {params}")
        
    
            
            
            
    
    
    

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
    
    
    
    
    
###############################


def main():
    from pthflops import count_ops
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/configs/base_to_nove/ss_base_to_novel.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    args = parser.parse_args()
    args.log_time='base_to_novel_ucf_testing'
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/visualization', 'TSNE_plot',
                               args.log_time)
    # wandb.init(project=config['network']['type'],
    #            name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
    #                                      config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,config, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_image.eval()
    # Or use input tensors
    # dummy_inputs = torch.rand(8, 3, 224, 224)
    
    # count_ops(model_image, dummy_inputs)
    
    
    parameters = model.visual.parameters()
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('CLIP_model Trainable Parameters: %.3fM' % parameters)

    
    calculate_Gflops_and_TP(model,device)

    
if __name__ == '__main__':
    main()
   

    
    
    
    
    
    
    
# if __name__ == '__main__':
#     path = '/media/sdb_access/Data_for_Action_CLIP/UCF_101/v_FrisbeeCatch_g05_c01'
#     files = os.listdir(path)
#     files = [file for file in files if os.path.isfile(os.path.join(path, file))]
#     num_frames = num_files = len(files)
#     segment_indices = get_val_indices(num_frames)
#     print(path,num_frames)
#     print(segment_indices)
#     images = get(path, segment_indices, transform_val)
    
    # num of frames 70