# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
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
import time


import matplotlib.pyplot as plt
import numpy as np
import math
from modules.Visual_Prompt import visual_prompt
# from modules.prompters import PadPrompter
from utils.KLLoss import *
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer,  _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)
    
def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='/media/sda1_acces/Code/ActionCLIP_without_wandb/prompt_work/ActionCLIP/configs/ucf101/ucf_train.yaml')
    # parser.add_argument('--prompt-vit-config-file', default='/media/sda1_acces/Code/VPT_new/cars.yaml',metavar="FILE")
    parser.add_argument('--log_time', default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    args = parser.parse_args()
     
    # print('##################################',args.log_time)
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    args.log_time = config['training_name']     
    # prompt_vit_model = Prompt_Vit_model(args.prompt_vit_config_file)
    working_dir = os.path.join('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/Exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    # wandb.init(project=config['network']['type'],name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))
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
    shutil.copy('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/train.py', working_dir)
    shutil.copy('/media/sda1_acces/Code/ActionCLIP_without_wandb/Prompt_Adaptor_mix_work/ActionCLIP/configs/k400/k400_train.yaml', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, config, device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))
    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    
    
    
    for name, p in fusion_model.named_parameters():
        p.requires_grad = False
        
        
    # for name, p in model.named_parameters():
    #     if 'logit_scale' not in name and 'message_' not in name and'text_projection' not in name and 'class_embedding' not in name and 'positional_embedding' not in name and 'temporal_embedding' not in name and 'ln_post' not in name and 'ln_final' not in name and 'prompt' not in name:
    #         p.requires_grad = False
    
    for name, p in model.named_parameters():
        if 'prompt' not in name and 'Adapter' not in name:
            p.requires_grad = False

    ###########################################################
    parameters = filter(lambda p: p.requires_grad, fusion_model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('fusion_model Trainable Parameters: %.3fM' % parameters)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('CLIP_model Trainable Parameters: %.3fM' % parameters)
    
    ############################################################
    # prompt_vit_model = torch.nn.DataParallel(prompt_vit_model).cuda()
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    # prompter = torch.nn.DataParallel(prompter).cuda()
    # wandb.watch(model)
    # wandb.watch(fusion_model)
    ############################## K-400 dataset  loader ###################################
    train_data = Action_DATASETS(config.data.ucf_train_list,config.data.ucf_label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.data.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    
    val_data = Action_DATASETS(config.data.ucf_val_list,config.data.ucf_label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    ###################################################################################################################################################
    
    ############################# Something dataset loader ########################
    # something_train_data = Action_DATASETS(config.data.something_train_list,config.data.something_label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.data.random_shift,
    #                    transform=transform_train)
    # something_train_loader = DataLoader(something_train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    # something_val_data = Action_DATASETS(config.data.something_val_list,config.data.something_label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
    #                    transform=transform_val)
    # something_val_loader = DataLoader(something_val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    #################################################################################    
    
   
    
    ############################## UCF-101 test loader ###################################
    ucf_val_data = Action_DATASETS(config.data.ucf_val_list,config.data.ucf_label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val)
    ucf_val_loader = DataLoader(ucf_val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    ###################################################################################################################################################
    
    ############################## HMDB51 test loader ###################################
    hmdb_val_data = Action_DATASETS(config.data.hmdb_val_list,config.data.hmdb_label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val)
    hmdb_val_loader = DataLoader(hmdb_val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    ###################################################################################################################################################
    
    ############################## K600 test loader ###################################
    # k600_val_data = Action_DATASETS(config.data.k600_val_list,config.data.k600_label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
    #                    transform=transform_val)
    # k600_val_loader = DataLoader(k600_val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)
    ###################################################################################################################################################
    

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()
    loss_motion = Motion_loss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'],strict=False)
            # prompter.load_state_dict(checkpoint['prompter_state_dict'],strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            # prompter.load_state_dict(checkpoint['prompter_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(train_data,config.data.ucf_gpt_discription, config.data.use_chat_gpt)
    
    ##### UCF ########
    ucf_classes, ucf_num_text_aug, ucf_text_dict = text_prompt(ucf_val_data,config.data.ucf_gpt_discription,config.data.use_chat_gpt)
    
    ##### hmdb########
    hmdb_classes, hmdb_num_text_aug, hmdb_text_dict = text_prompt(hmdb_val_data,config.data.hmdb_gpt_discription, config.data.use_chat_gpt)
    ##### Something ########
    # something_classes, something_num_text_aug, something_text_dict = text_prompt(something_train_data,config.data.something_gpt_discription,config.data.use_chat_gpt)
    #### k600 #######
    # k600_classes, k600_num_text_aug, k600_text_dict = text_prompt(k600_val_data,config.data.k600_gpt_discription,config.data.use_chat_gpt)
    
    
    optimizer = _optimizer(config, model, fusion_model)
    # optimizer = _optimizer_2(config, model_image, model_text, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    loss=[]
    top_1_acc=[]
    ucf_top_1_acc=[]
    hmdb_top_1_acc=[]
    something_top_1_acc=[]
    k600_top_1_acc=[]
    
    best_prec1 = 0.0
    ucf_best_prec1 = 0.0
    hmdb_best_prec1 = 0.0
    something_best_prec1 = 0.0
    k600_best_prec1 = 0.0
    
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, device,model, fusion_model, config,num_text_aug,working_dir,'k_400')
        return

    for k,v in model.named_parameters():
        if v.requires_grad:
            print('{}: {}'.format(k, v.requires_grad))

    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        print('------------------------------------------------------------------------')
        print('Epoch %d start ..'%epoch)
        # prompt_vit_model.train()
        model_image.train()
        model_text.train()
        fusion_model.train()
        # for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
        tic = time.time()
        epoch_loss=[]
        for kkk,(prompt_images,list_id) in enumerate((train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()
            # prompt_images = prompter(images)

            prompt_images = prompt_images.view((-1,config.data.num_segments,3)+prompt_images.size()[-2:])
            b,t,c,h,w = prompt_images.size()
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            prompt_images= prompt_images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            image_embedding = model_image(prompt_images) 
            image_embedding = image_embedding.view(b,t,-1)
            if config.use_motion_loss:
                loss_video_motion = loss_motion(image_embedding)
            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            if config.use_motion_loss:
                total_loss = (loss_imgs + loss_texts)/2 + loss_video_motion
            else:
                total_loss = (loss_imgs + loss_texts)/2
            # total_loss = (loss_imgs + loss_texts)/2 
            epoch_loss.append(total_loss.item())
            # wandb.log({"train_total_loss": total_loss})
            # wandb.log({"train_loss_imgs": loss_imgs})
            # wandb.log({"train_loss_texts": loss_texts})
            # wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            
            if kkk%10==0:
                if config.use_motion_loss:
                    print('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, motion loss:%f, lr:%f '%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), loss_video_motion.item(), optimizer.param_groups[0]['lr']))
                else:
                    print('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, lr:%f '%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), optimizer.param_groups[0]['lr']))
                # print('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, lr:%f '%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), optimizer.param_groups[0]['lr']))

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            print('UCF accuracy')
            ucf_prec1 = validate(epoch,ucf_val_loader, ucf_classes, device, model, fusion_model, config,ucf_num_text_aug,working_dir,'UCF')
            print('HMDB accuracy')
            hmdb_prec1 = validate(epoch,hmdb_val_loader, hmdb_classes, device, model, fusion_model, config,hmdb_num_text_aug,working_dir,'HMDB')
            # print('K600 accuracy')
            # k600_prec1 = validate(epoch,k600_val_loader, k600_classes, device, model, fusion_model, config,k600_num_text_aug,working_dir,'K600')
            # print('Something accuracy')
            # something_prec1 = validate(epoch,something_val_loader, something_classes, device, model, fusion_model, config,something_num_text_aug,working_dir,'SSv2')
            print('K-400 val accuracy')
            prec1 = validate(epoch,val_loader, classes, device, model, fusion_model, config,num_text_aug, working_dir,'K-400')
        # print('Epoch:%d, total loss:%f, image loss:%f, text loss:%f'%(epoch,total_loss.item(),loss_imgs.item(),loss_texts.item()))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('K-400 Testing: {}/{}'.format(prec1,best_prec1))
        
        ucf_is_best = ucf_prec1 > ucf_best_prec1
        ucf_best_prec1 = max(ucf_prec1, ucf_best_prec1)
        print('UCF Testing: {}/{}'.format(ucf_prec1,ucf_best_prec1))
        
        hmdb_is_best = hmdb_prec1 > hmdb_best_prec1
        hmdb_best_prec1 = max(hmdb_prec1, hmdb_best_prec1)
        print('HMDB Testing: {}/{}'.format(hmdb_prec1,hmdb_best_prec1))
        
        # something_is_best = something_prec1 > something_best_prec1
        # something_best_prec1 = max(something_prec1, something_best_prec1)
        # print('something Testing: {}/{}'.format(something_prec1,something_best_prec1))
        
        
        # k600_is_best = k600_prec1 > k600_best_prec1
        # k600_best_prec1 = max(k600_prec1, k600_best_prec1)
        # print('k600 Testing: {}/{}'.format(k600_prec1,k600_best_prec1))
        
        
        txt_path = '{}/log.txt'.format(working_dir)
        if os.path.exists(txt_path):
            with open(txt_path, 'a+') as f:
                f.write('\n')
                if config.use_motion_loss:
                    f.write('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, motion loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), loss_video_motion.item(), optimizer.param_groups[0]['lr']))
                else:
                    f.write('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), optimizer.param_groups[0]['lr']))
                f.write('K-400 Testing: {}/{}\n'.format(prec1,best_prec1))
                f.write('UCF Testing: {}/{}\n'.format(ucf_prec1,ucf_best_prec1))
                f.write('HMDB Testing: {}/{}\n'.format(hmdb_prec1,hmdb_best_prec1))
                # f.write('something Testing: {}/{}\n'.format(something_prec1,something_best_prec1))
                # f.write('k600 Testing: {}/{}\n'.format(k600_prec1,k600_best_prec1))
                f.close()
        else:
            with open(txt_path, mode="wt") as f:
                if config.use_motion_loss:
                    f.write('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, motion loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), loss_video_motion.item(), optimizer.param_groups[0]['lr']))
                else:
                    f.write('Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, text loss:%f, lr:%f \n'%(epoch,kkk,len(train_loader),total_loss.item(),loss_imgs.item(),loss_texts.item(), optimizer.param_groups[0]['lr']))
                f.write('K-400 Testing: {}/{}\n'.format(prec1,best_prec1))
                f.write('UCF Testing: {}/{}\n'.format(ucf_prec1,ucf_best_prec1))
                f.write('HMDB Testing: {}/{}\n'.format(hmdb_prec1,hmdb_best_prec1))
                f.write('HMDB Testing: {}/{}\n'.format(hmdb_prec1,hmdb_best_prec1))
                f.write('HMDB Testing: {}/{}\n'.format(hmdb_prec1,hmdb_best_prec1))
                # f.write('something Testing: {}/{}\n'.format(something_prec1,something_best_prec1))
                # f.write('k600 Testing: {}/{}\n'.format(k600_prec1,k600_best_prec1))
                f.close()
        # something_is_best = something_prec1 > something_best_prec1
        # something_best_prec1 = max(something_prec1, something_best_prec1)
        # print('Something Testing: {}/{}'.format(something_prec1,something_best_prec1))
        
        
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)
        top_1_acc.append(prec1/100)
        ucf_top_1_acc.append(ucf_prec1/100)
        hmdb_top_1_acc.append(hmdb_prec1/100)
        # k600_top_1_acc.append(k600_prec1/100)
        # something_top_1_acc.append(something_prec1/100)
        loss.append(np.mean(epoch_loss))
        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if is_best:
            print('Saving best weight based on K-400 accuracy at epoch %d'%epoch)
            best_saving(working_dir, epoch, model, fusion_model, optimizer, 'K_400')
        if ucf_is_best:
            print('Saving best weight based on UCF accuracy at epoch %d'%epoch)
            best_saving(working_dir, epoch, model, fusion_model, optimizer, 'UCF_101')
        if hmdb_is_best:
            print('Saving best weight based on HMDB accuracy at epoch %d'%epoch)
            best_saving(working_dir, epoch, model, fusion_model, optimizer, 'HMDB_51')
        
        # if k600_is_best:
        #     print('Saving best weight based on k600 accuracy at epoch %d'%epoch)
        #     best_saving(working_dir, epoch, model, fusion_model, optimizer, 'k600_51')
        
        # if something_is_best:
        #     print('Saving best weight based on Something accuracy at epoch %d'%epoch)
        #     best_saving(working_dir, epoch, model, fusion_model, optimizer, 'Something')
        print('Epoch %d end ..'%epoch)
        # print('Time taken by epoch %d:'%epoch, print_time(time.time()-tic))
        # print('------------------------------------------------------------------------')
        ##############graph_plot################
        # X = list(range(config.solver.epochs))
        X = list(range(len(loss)))
        plt.plot(X, loss, color='r', label='Training loss')
        plt.plot(X, top_1_acc, color='g', label='K-400 Accuracy')
        plt.plot(X, ucf_top_1_acc, color='b', label='UCF-101 Accuracy')
        plt.plot(X, hmdb_top_1_acc, color='m', label='HMDB-51 Accuracy')
        # plt.plot(X, k600_top_1_acc, color='m', label='k600-51 Accuracy')
        # plt.plot(X, something_top_1_acc, color='m', label='SSv2 Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Training loss and Accuracy")
        plt.title("Traing graph")
        plt.legend()
        plt.savefig('{}/Graph_plot.png'.format(working_dir))
        plt.close()
        print('Time taken by epoch %d:'%epoch, print_time(time.time()-tic))
        print('------------------------------------------------------------------------')
if __name__ == '__main__':
    main()
