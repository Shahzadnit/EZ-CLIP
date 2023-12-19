import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
from tqdm import tqdm

import argparse
out_path = ''

import multiprocessing
manager = multiprocessing.Manager()
# final_list = manager.list()

train_videos=[]
test_videos = []
index_dic = {}
train_video_path = manager.list()
test_video_path = manager.list()

def dump_frames(vid_path):

    import cv2
    video_name = vid_path.split('/')[-1]
    label_index = index_dic[vid_path.split('/')[-2]] 

    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    # vid_name = vid_path.split('/')[-1]
    out_full_path = os.path.join(out_path, vid_name)
    
    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(1,fcount+1):
        ret, frame = video.read()
        if ret:
            # cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)               
            cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, i), frame)  
            # access_path = '{}/{:06d}.jpg'.format(vid_name, i)
            access_path = '{}/img_{:05d}.jpg'.format(vid_name, i)
            file_list.append(access_path)
    print('{} done'.format(vid_name))
    if video_name in train_videos:
        train_video_path.append(out_full_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
    elif video_name in test_videos:
        test_video_path.append(out_full_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
    sys.stdout.flush()
    return file_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UCF dataset preparation")
    parser.add_argument("--src_dir",type=str,default='/home/shahzaa/Downloads/UCF101/UCF-101/')  # give dataset video path
    parser.add_argument("--out_dir", type=str,default='/media/sda1_acces/Data_for_Action_CLIP/UCF_101_2') # give output path
    parser.add_argument("--num_worker", type=int, default=10)   
    parser.add_argument("--new_width", type=int, default=340, help='resize image width')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_height", type=int, default=256, help='resize image height')
    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    
    
    new_size = (args.new_width, args.new_height)

    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)
    print("reading videos from folder: ", src_path)
    print("selected extension of videos:", ext)
    # vid_list = glob.glob(src_path+'/*.'+ext)
    #################################################
    # train_videos=[]
    # train_txt=['/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/trainlist01.txt','/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/trainlist02.txt', '/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/trainlist03.txt']
    train_txt=['/media/sdb_access/UCF_101_dataset/ucfTrainTestlist/trainlist01.txt']  # give path of UCF train list 

    for train_file in train_txt:
        train_file = open(train_file , 'r')
        Lines = train_file.readlines()
        for line in Lines:
            train_videos.append(line.split(' ')[0].split('/')[-1])
    print(len(train_videos))

    # test_videos=[]
    # test_txt=['/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/testlist01.txt','/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/testlist02.txt', '/media/sda1_acces/Data_for_Action_CLIP/ucfTrainTestlist/testlist03.txt']
    test_txt=['/media/sdb_access/UCF_101_dataset/ucfTrainTestlist/testlist01.txt']  # give path of UCF test list 

    for test_file in test_txt:
        test_file = open(test_file , 'r')
        Lines = test_file.readlines()
        for line in Lines:
            test_videos.append(line.split(' ')[0].split('/')[-1].replace("\n", ""))
    print(len(test_videos))

    label_index_file = '/media/sdb_access/UCF_101_dataset/ucfTrainTestlist/classInd.txt'  # give path of UCF class Ind list after dataset download
    label_file = open(label_index_file , 'r')
    Lines = label_file.readlines()
    for line in Lines:
        index = int(line.replace("\n", "").split(' ')[0])-1
        label = line.replace("\n", "").split(' ')[1]
        index_dic.update({label:index})
    #################################################
    vid_list = []
    for folder, subs, files in os.walk(src_path):
      for filename in files:
        vid_list.append(os.path.abspath(os.path.join(folder, filename)))
    #################################################
    print("total number of videos found: ", len(vid_list))

    pool = Pool(num_worker)
    pool.map(dump_frames, vid_list)

    # for video_path in tqdm(vid_list[:100]):
    #     file_list = dump_frames(video_path)

    print(len(train_video_path))
    print(len(test_video_path))
    

    # writing to file
    file1 = open('../dataset_txt/UCF/train.txt', 'w')
    file1.writelines(list(train_video_path))
    file1.close()

    file2 = open('../dataset_txt/UCF/test.txt', 'w')
    file2.writelines(list(test_video_path))
    file2.close()




