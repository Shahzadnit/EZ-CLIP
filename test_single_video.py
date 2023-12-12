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
    
def calculate_Gflops_and_TP(model,device):
    from transformers import CLIPProcessor
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    image_input = torch.randn(8,3, 224, 224).to(device)
    text_input = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # text_input = processor(["a photo of a cat on a sunny day"], return_tensors="pt").to(device)
    # Number of inference passes
    num_inference_passes = 1000  # Adjust as needed

    # Warm-up inference
    with torch.no_grad():
        for _ in range(10):  # Warm-up for stability
            outputs = model(image_input, text_input)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_inference_passes):
            outputs = model(image_input, text_input)
    end_time = time.time()

    # Calculate FLOPs
    flops, params = profile(model, inputs=(image_input, text_input))
    flops_in_gflops = flops / 1e9

    # Calculate throughput in IPS
    total_time = end_time - start_time
    throughput_ips = (num_inference_passes / total_time)*8

    print(f"Number of Inference Passes: {num_inference_passes}")
    print(f"Total Time: {total_time} seconds")
    print(f"Throughput: {throughput_ips} IPS")
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
    

def interpolate_points(p1, p2, t):
    return (1 - t) * p1 + t * p2


def generate_motion_path(points, num_steps=100):
    path = []
    for i in range(len(points) - 1):
        for t in np.linspace(0, 1, num_steps):
            interpolated_point = interpolate_points(points[i], points[i + 1], t)
            path.append(interpolated_point)
    return np.array(path)  
  
######### TSNE PLOT ##########
def feature_plot(video_feature,labels,working_dir,epoch,dataset_name):
    name = 'embedding_%d.png'%(epoch)
    features = torch.cat(video_feature, dim=0)
    labels = torch.cat(labels, dim=0)
    # num_of_class = len(labels.unique())
    plot_dir = os.path.join(working_dir, dataset_name+'_plot')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=8, random_state=0, n_jobs=-1)
    tsne_result = tsne.fit_transform(features.cpu())

    min_val = np.min(tsne_result)
    max_val = np.max(tsne_result)
    # Normalize the entire data between 0 and 1
    tsne_result = (tsne_result - min_val) / (max_val - min_val)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels.cpu(), cmap=plt.get_cmap('tab20'), s=10)
    for i in range(0, tsne_result.shape[0], 8):
        points = tsne_result[i:i + 8,:]
        motion_path = generate_motion_path(points)
        plt.plot(motion_path[:, 0], motion_path[:, 1], linestyle='dashed', label='Motion Path')

    plt.colorbar()
    plt.title('t-SNE Visualization')
    plt.savefig(os.path.join(plot_dir, name))
    plt.close()
    
    
###############################

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug,working_dir,dataset_name):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    video_features = []
    labels = []
    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        # for iii, (image, class_id) in enumerate((val_loader)):
        for iii, (prompt_image, class_id) in enumerate(tqdm(val_loader)):
            # prompt_image = prompter(image)
            prompt_image = prompt_image.view((-1, config.data.num_segments, 3) + prompt_image.size()[-2:])
            b, t, c, h, w = prompt_image.size()
            class_id = class_id.to(device)
            image_input = prompt_image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            video_features.append(image_features)
            labels.append(class_id)
            
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    # wandb.log({"top1": top1})
    # wandb.log({"top5": top5})
    feature_plot(video_features,labels,working_dir,epoch,dataset_name)
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1

def load_image(directory, idx):
    image_tmpl = 'img_{:05d}.jpg'
    return [Image.open(os.path.join(directory, image_tmpl.format(idx))).convert('RGB')]


def load_video(video_path, indices):        
    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    frames_rgb =[]
    #######################################################
    # tic = time.time()
    for index in indices:
        # cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        cap.set(1,index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224,224))
            # frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_rgb.append(frame)
    # print('Time taken by selected frames: ',time.time()-tic)
    cap.release()
    #########################################################   cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
    # cap = cv2.VideoCapture(video_path)
    # tic = time.time()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.resize(frame, (224,224))
    #     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))      
    # print('Time taken by all frames: ',time.time()-tic)
    ###########################################################    
    # cap.release()
    return frames_rgb

    
def get(path, indices,transform):
    ################ To make each frame constant in video which remove the motion ##################
    # for i in range(len(indices)):
    #     indices[i]=indices[len(indices)//2]
    ################################################################################################
    images = list()
    for i, seg_ind in enumerate(indices):
        p = int(seg_ind)
        try:
            seg_imgs = load_image(path, p)
        except OSError:
            print('ERROR: Could not read image "{}"'.format(path))
            print('invalid indices: {}'.format(indices))
            raise
        images.extend(seg_imgs)
    process_data = transform(images)
    return process_data

def get_video_frame(path, indices,transform):
    ################ To make each frame constant in video which remove the motion ##################
    # for i in range(len(indices)):
    #     indices[i]=indices[len(indices)//2]
    ################################################################################################
    frames = load_video(path, indices)
    images = list()
    # print(len(frames))
    # print(indices)
    # for i, seg_ind in enumerate(indices):
    for i, seg_ind in enumerate(frames):
        try:
            seg_imgs = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).convert('RGB')]
        except OSError:
            print('ERROR: Could not read image "{}"'.format(path))
            print('invalid indices: {}'.format(indices))
            raise
        images.extend(seg_imgs)
    process_data = transform(images)
    return process_data, frames

    

    
def get_val_indices(num_frames, num_segments=8):
    offset = (num_frames / num_segments - 1) / 2.0
    return np.array([i * num_frames / num_segments + offset + j
                        for i in range(num_segments)
                        for j in range(1)], dtype=np.int) + 1


def get_num_of_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames
    
def overly_images_new(images,Attn_map):
    overlay_images = []
    heatmap_list = []
    count = 1
    for image,heatmap in zip(images,Attn_map[:,]):
        cv2.putText(image, str(count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # heatmap = torch.sum(attention_tensor, dim=-1)
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        heatmap = 1-((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = cv2.resize(np.float32(heatmap), (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_1 = heatmap.copy()
        heatmap_colormap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)
        heatmap_list.append(heatmap_colormap_1)
        # _, heatmap= cv2.threshold(heatmap, 225, 255, cv2.THRESH_BINARY)
        # heatmap[heatmap < 225] = 0
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap_colormap, 1, 0)
        overlay_images.append(overlay)
        count = count + 1
    final = np.vstack([np.hstack(overlay_images[0:4]),np.hstack(overlay_images[4:8])])
    final_heat_map = np.vstack([np.hstack(heatmap_list[0:4]),np.hstack(heatmap_list[4:8])])
    combined = np.vstack([final,final_heat_map])
    return combined
        
        
def overly_images_old(images,Attn_map):
    overlay_images = []
    simple_image =[]
    heatmap_list = []
    count = 1
    for image,attention_tensor in zip(images,Attn_map[:,]):
        simple_image.append(image)
        cv2.putText(image, str(count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # heatmap = torch.sum(attention_tensor, dim=-1)
        heatmap = F.interpolate(attention_tensor.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = cv2.resize(np.float32(heatmap), (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        # _, heatmap= cv2.threshold(heatmap, 150, 255, cv2.THRESH_BINARY)
        # heatmap[heatmap < 150] = heatmap[heatmap < 150]*.7
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_list.append(heatmap_colormap)
        overlay = cv2.addWeighted(image, 1, heatmap_colormap, 0.5, 0)
        overlay_images.append(overlay)
        count = count + 1
    final = np.vstack([np.hstack(overlay_images[0:4]),np.hstack(overlay_images[4:8])])
    final_heat_map = np.vstack([np.hstack(heatmap_list[0:4]),np.hstack(heatmap_list[4:8])])
    combined = np.vstack([final,final_heat_map])
    
    
    combined_2 = np.vstack([np.hstack(simple_image[::2]),np.hstack(overlay_images[::2])])
    return combined_2


def create_gif(images,Attn_map,gif_path):
    import imageio
    overlay_images = []
    count = 1
    for image,attention_tensor in zip(images,Attn_map[:,]):
        cv2.putText(image, str(count), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # heatmap = torch.sum(attention_tensor, dim=-1)
        heatmap = F.interpolate(attention_tensor.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = cv2.resize(np.float32(heatmap), (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        # _, heatmap= cv2.threshold(heatmap, 150, 255, cv2.THRESH_BINARY)
        # heatmap[heatmap < 150] = heatmap[heatmap < 150]*.7
        heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 1, heatmap_colormap, 0.5, 0)
        combine_img = cv2.cvtColor(np.hstack([overlay,heatmap_colormap]), cv2.COLOR_BGR2RGB)
        overlay_images.append(combine_img)
        count = count + 1    
    imageio.mimsave(gif_path, overlay_images, fps=1.5)
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # height, width, layers = images[0].shape
    # out = cv2.VideoWriter(gif_path, fourcc, 1, (width, height), True)

    # # Write each frame (image) to the GIF
    # for image in images:
    #     out.write(image)

    # # Release the VideoWriter
    # out.release()

def get_class_name_list(csv_path):
    Class_dic={}
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file)
        for i,row in enumerate(csvreader):
            if i>0:
                class_name = row[1].replace(' ','_')
                cls_id = int(row[0].replace(' ','_'))
                Class_dic.update({cls_id:class_name})
    return Class_dic
 
     
def tensor_image(tensor_image,i):
    heatmap = F.interpolate(tensor_image.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    heatmap = heatmap.cpu().detach().numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/test_img/img_%d.png'%(i),heatmap_colormap)
def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/configs/base_to_nove/ucf_base_to_novel.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    args = parser.parse_args()
    args.log_time='base_to_novel_ucf_testing_2'
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

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    # shutil.copy(args.config, working_dir)
    # shutil.copy('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,config, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    # prompter = PadPrompter(config)
    
    # calculate_Gflops_and_TP(model,device)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    # prompter = torch.nn.DataParallel(prompter).cuda()
    # wandb.watch(model)
    # wandb.watch(fusion_model)
    

    val_data = Action_DATASETS(config.data.novel_val_list, config.data.novel_label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            # prompter.load_state_dict(checkpoint['prompter_state_dict'],strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data,config.data.gpt_discription,config.data.use_chat_gpt)

    best_prec1 = 0.0
    # prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug,working_dir,'Novel')
    class_name_csv_path = '/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/Base_to_novel/ucf/ucf_base_train_class_list.csv'
    # class_name_csv_path = '/media/sdb_access/Data_for_Action_CLIP/Txts/Someting_Something/ss_base_train_class_list.csv'
    class_name_dic = get_class_name_list(class_name_csv_path)
    
    
    video_file = open('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/Base_to_novel/ucf/my_ucf_base_train.txt' , 'r')
    # video_file = open('/media/sdb_access/Data_for_Action_CLIP/Txts/Someting_Something/my_ss_base_val.txt' , 'r')
    
    # video_file = open('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/base_to_novel/hmdb/my_hmdb_base_train.txt' , 'r')
    Lines = video_file.readlines()
    video_names = []
    video_name_to_class_name = {}
    for line in Lines:
        class_number = int(line.split(' ')[-1].replace('\n',''))
        class_name = class_name_dic[class_number]
        videoname = line.split(' ')[0].split('/')[-1]+'.avi'  #ucf
        # videoname = line.split(' ')[0].split('/')[-1]  #hmdb
        # videoname = line.split(' ')[0].split('/')[-1]  # ss
        video_name_to_class_name.update({videoname:[class_name,class_number]})
        video_names.append(videoname)
    
    
    video_files = glob.glob('/media/sdb_access/UCF_101_dataset/All_video/*.avi')
    # video_files = glob.glob('/media/sdb_access/20bn-something-something-v2/*.webm')
    # video_files = glob.glob('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/visualization/test_video/ucf-101/*.avi')
    # video_files = glob.glob('/media/sda1_acces/Code/UCF_Out_of_distribution_data/*.mp4')
    
    # random.shuffle(video_files)
    # video_files = glob.glob('/media/sdb_access/HMDB_51/all_video/*.avi')
    plot_dir = '/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/visualization/UCF_new_2/'
    gif_dir = '/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/visualization/UCF_new_2_gif/'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    class_for_plot = random.sample(range(80), 30)
    cls_id_check =[]
    labels = []
    video_features = []
    for video_path in tqdm(video_files):
        vid_name = video_path.split('/')[-1]
        if vid_name in video_names:
        # if True:
            cls_name = video_name_to_class_name[vid_name][0]
            cls_id = video_name_to_class_name[vid_name][1]
            if cls_id > 6:
                continue
            # labels.append(cls_id)
            num_frames_video = get_num_of_frame(video_path)
            segment_indices = get_val_indices(num_frames_video)
            # print(path,num_frames)
            # images = get(path, segment_indices, transform_val)
            video_images_tensor,frames = get_video_frame(video_path, segment_indices,transform_val)
            img_name = cls_name+'_'+vid_name.replace('.avi','.png')
            gif_name = cls_name+'_'+vid_name.replace('.avi','.gif')
            
            # img_name = vid_name.replace('.mp4','.png')
            # gif_name = vid_name.replace('.mp4','.gif')
            # img_name = cls_name+'_'+vid_name.replace('.webm','.png')
            # gif_name = cls_name+'_'+vid_name.replace('.webm','.gif')
            #########
            model.eval()
            with torch.no_grad():
                if video_images_tensor.shape[0]<24:
                    # print(video_images_tensor.shape)
                    # print('drop this video')
                    continue
                
                prompt_image = video_images_tensor.view((-1, config.data.num_segments, 3) + video_images_tensor.size()[-2:])
                b, t, c, h, w = prompt_image.size()
                image_input = prompt_image.to(device).view(-1, c, h, w)
                
                if cls_id < 5:
                    # labels.append(torch.tensor([1,2,3,4,5,6,7,8]))
                    # # labels.append(torch.tensor([cls_id]*8))
                    # embeddings = model.encode_image(image_input).view(b, t, -1)
                    # embeddings = embeddings.mean(dim=0, keepdim=False)
                    # video_features.append(embeddings)
                    # if len(video_features)>40:
                    #     break       
                    if cls_id not in cls_id_check: # to take only one video from each class
                        cls_id_check.append(cls_id)
                        # labels.append(torch.tensor([cls_id]*8))
                        labels.append(torch.tensor([1,2,3,4,5,6,7,8]))
                        embeddings = model.encode_image(image_input).view(b, t, -1)
                        embeddings = embeddings.mean(dim=0, keepdim=False)
                        video_features.append(embeddings)
                        if len(video_features)>6:
                            break

                # attentions = model.encode_image_attention(image_input)
                # attentions =torch.mean(attentions,1)[:,1:-1].view(8,14,14)
                # attentions =torch.mean(attentions,1)[:,1:].view(8,14,14)  # temporal prompt not in use
                # print(image_features.shape)


                ######
                # final_images = overly_images_old(frames,attentions)
                # image_path = os.path.join(plot_dir, img_name)
                # gif_path = os.path.join(gif_dir, gif_name)
                # create_gif(frames,attentions,gif_path)
                # cv2.imwrite(image_path,final_images)
                #################
                
    # labels = torch.tensor(labels)
    feature_plot(video_features,labels,working_dir,1,'UCF')    
    
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