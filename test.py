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
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *

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
    
    
    
######### TSNE PLOT ##########
def feature_plot(video_feature,labels,working_dir,epoch,dataset_name):
    name = 'embedding_%d.pdf'%(epoch)
    features = torch.cat(video_feature, dim=0)
    labels = torch.cat(labels, dim=0)
    # num_of_class = len(labels.unique())
    plot_dir = os.path.join(working_dir, dataset_name+'_plot')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
    tsne_result = tsne.fit_transform(features.cpu())
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels.cpu(), cmap=plt.get_cmap('tab20'), s=6)
    plt.colorbar()
    plt.title('t-SNE Visualization')
    plt.savefig(os.path.join(plot_dir, name),format='pdf', dpi=300)
    plt.close()
    
    
###############################
def Class_analysis(cls_dic,pred,orignal, video_features,labels,working_dir,epoch,dataset_name):
    plot_dir = os.path.join(working_dir, dataset_name+'_class_analysis')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    txt_path = '{}/class_analysis_{}.txt'.format(plot_dir,epoch)
    plot_path = '{}/class_analysis_{}.pdf'.format(plot_dir,epoch)
    confusion_matrix_plot_path = '{}/confusion_class_analysis_{}.pdf'.format(plot_dir,epoch)
    cls_list = []
    percentage_list = []
    cls_persentage_dic ={}
    with open(txt_path, mode="wt") as f:
        for key in np.sort(list(cls_dic.keys())):
            f.write('\n')
            alist = cls_dic[key]
            percentage_of_ones = (sum(alist) / len(alist)) * 100
            cls_list.append(key)
            percentage_list.append(percentage_of_ones)
            cls_persentage_dic.update({key:percentage_of_ones})
            f.write('class: {}---{}\n'.format(key,percentage_of_ones))
        f.close()
    sorted_dict = {k: v for k, v in sorted(cls_persentage_dic.items(), key=lambda item: item[1], reverse=True)}
    top_30_keys = list(sorted_dict.keys())[:30]
    label_new = []
    video_features_new =[]
    for label , feature in zip(labels, video_features):
        if label.item() in top_30_keys:
            label_new.append(label)
            video_features_new.append(feature)
    print(len(label_new),len(video_features_new))
    feature_plot(video_features_new,label_new,working_dir,epoch,dataset_name)
    plt.bar(cls_list, percentage_list)
    # Set labels and title
    plt.xlabel('Class Numbers')
    plt.ylabel('Percentage (%)')
    # plt.title('Class Analysis {}'.format(epoch))
    plt.title('Class Analysis')
    plt.savefig(plot_path, format='pdf', dpi=300)
    plt.close()
    
    cm = confusion_matrix(orignal, pred)
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', cbar=True)

    # Set labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_plot_path, format='pdf', dpi=300)
    plt.close()
###############################

def validate(epoch, val_loader, classes, device, model, config, num_text_aug,working_dir,dataset_name):
    model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    cls_dic = {}
    video_features = []
    labels = []
    pred = []
    orignal = []
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
            # image_features = fusion_model(image_features)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            class_for_plot = random.sample(range(230), 160)
            for id,feature in zip(class_id,image_features):
                if dataset_name=='K600':
                    if id.item() in class_for_plot:    
                        video_features.append(feature.unsqueeze(0))
                        labels.append(torch.tensor([id.item()]))
                else:
                    video_features.append(feature.unsqueeze(0))
                    labels.append(torch.tensor([id.item()]))
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            # print(values_1, indices_1)
            pred.append(indices_1.squeeze().tolist())
            orignal.append(class_id.squeeze().tolist())
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item():[1]})
                    else:
                        cls_dic[class_id[i].item()].append(1)
                else:
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item():[0]})
                    else:
                        cls_dic[class_id[i].item()].append(0)
                        
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100

    # pred = [item for sublist in pred for item in sublist]
    # orignal = [item for sublist in orignal for item in sublist]
    # Class_analysis(cls_dic,pred,orignal, video_features,labels,working_dir,epoch,dataset_name)
    # feature_plot(video_features,labels,working_dir,epoch,dataset_name)
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1

def validate_1(epoch, val_loader, classes, device, model, config, num_text_aug,working_dir,dataset_name):
    model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    cls_dic = {}
    video_features = []
    labels = []
    pred = []
    orignal = []
    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        for iii, (prompt_image, class_id) in enumerate(tqdm(val_loader)):
            prompt_image = prompt_image.view((-1, config.data.num_segments, 3) + prompt_image.size()[-2:])
            b, t, c, h, w = prompt_image.size()
            class_id = class_id.to(device)
            image_input = prompt_image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            class_for_plot = random.sample(range(230), 160)
            for id,feature in zip(class_id,image_features):
                if dataset_name=='K600':
                    if id.item() in class_for_plot:    
                        video_features.append(feature.unsqueeze(0))
                        labels.append(torch.tensor([id.item()]))
                else:
                    video_features.append(feature.unsqueeze(0))
                    labels.append(torch.tensor([id.item()]))
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            # print(values_1, indices_1)
            pred.append(indices_1.squeeze().tolist())
            orignal.append(class_id.squeeze().tolist())
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item():[1]})
                    else:
                        cls_dic[class_id[i].item()].append(1)
                else:
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item():[0]})
                    else:
                        cls_dic[class_id[i].item()].append(0)
                        
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    pred = [item for sublist in pred for item in sublist]
    orignal = [item for sublist in orignal for item in sublist]
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    # Class_analysis(cls_dic,pred,orignal, video_features,labels, working_dir,epoch,dataset_name)
    # feature_plot(video_features,labels,working_dir,epoch,dataset_name)
    
    return top1



def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='/cluster/datastore/shahzaa/Code/EZ-CLIP/configs/ucf101/UCF_zero_shot_testing.yaml')
    parser.add_argument('--traning_name', default='')
    args = parser.parse_args()
    args.traning_name = 'SSv2_base_to_novel_testing'  # give testing name 
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join(config['weight_save_dir'], config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.traning_name)
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
    shutil.copy(args.config, working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,config, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()

    for name, p in model.named_parameters():
        if 'prompt' not in name and 'Adapter' not in name:
            p.requires_grad = False

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data,config.data.gpt_discription,config.data.use_llm)

    best_prec1 = 0.0
    # 'UCF_base' is dataset name 
    prec1 = validate(start_epoch, val_loader, classes, device, model, config, num_text_aug,working_dir,'UCF_base')

if __name__ == '__main__':
    main()