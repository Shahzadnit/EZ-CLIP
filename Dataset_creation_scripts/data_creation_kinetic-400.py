import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
from tqdm import tqdm
import csv
import random
import argparse
import time
out_path = ''

import multiprocessing
manager = multiprocessing.Manager()
# final_list = manager.list()

train_videos=[]
test_videos = []
index_dic = {}
video_dic = {}
train_video_path = manager.list()
test_video_path = manager.list()
val_video_path = manager.list()

def dump_frames(vid_path):

    import cv2
    video_name = vid_path.split('/')[-1]
    video_id = video_name[:-18] 
    if video_id in video_dic.keys():
        label = video_dic[video_id][0]
        video_type = video_dic[video_id][1] # val, test, train
        label_index = index_dic[label ]
        # label_index = index_dic[vid_path.split('/')[-2]] 

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
                access_path = '{}/img_{:05d}.jpg'.format(vid_name, i)
                file_list.append(access_path)
        print('{} done'.format(vid_name))

        if len(file_list)>7:  
            if video_type=='train':
                train_video_path.append(vid_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
            if video_type=='test':
                test_video_path.append(vid_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
            if video_type=='val':
                val_video_path.append(vid_path+' '+str(len(file_list))+' '+str(label_index)+'\n') 
            sys.stdout.flush()
            return file_list
        else:
            None
    else:
        return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="K-400 dataset preparation")
    parser.add_argument("--src_dir",type=str,default='/media/sdb_access/Kinetics/k_400_data_set') # give dataset video path
    parser.add_argument("--out_dir", type=str,default='/media/sdb_access/Data_for_Action_CLIP/KINETICS_400') # give output path
    parser.add_argument("--num_worker", type=int, default=10)   
    parser.add_argument("--new_width", type=int, default=224, help='resize image width')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_height", type=int, default=224, help='resize image height')
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
    ############################################################################

    '''
    downlaod dataset K-400 from  'https://github.com/cvdfoundation/kinetics-dataset' give the '.csv' path here
    '''
    csv_files = glob.glob('/media/sdb_access/Kinetics/kinetics-dataset/k400/annotations/*.csv')
    for csv_file in csv_files:
        with open(csv_file, 'r') as file:
            csvreader = csv.reader(file)
            for i,row in enumerate(csvreader):
                video_dic.update({row[1]:[row[0],row[4]]})
    ############################################################################

    with open("../lists/kinetics_400_labels.csv", 'r') as file:
        csvreader = csv.reader(file)
        for i,row in enumerate(csvreader):
            if i>0:
                index = int(row[0])
                label = row[1]
                index_dic.update({label:index})
    #################################################
    vid_list = []
    for folder, subs, files in os.walk(src_path):
      for filename in files:
        if '.mp4' in filename:
            vid_list.append(os.path.abspath(os.path.join(folder, filename)))
    #################################################
    print("total number of videos found: ", len(vid_list))

    random.shuffle(vid_list)

    tic = time.time()

    pool = Pool(num_worker)
    pool.map(dump_frames, vid_list)


    
    # for video_path in tqdm(vid_list[0:500]):
    #     file_list = dump_frames(video_path)
    print('time taken----',time.time()-tic)

    print(len(train_video_path))
    print(len(test_video_path))
    print(len(val_video_path))

    # writing to file
    file1 = open('../dataset_txt/K400/train.txt', 'w')
    file1.writelines(list(train_video_path))
    file1.close()

    file2 = open('../dataset_txt/K400/test.txt', 'w')
    file2.writelines(list(test_video_path))
    file2.close()

    file2 = open('../dataset_txt/K400/val.txt', 'w')
    file2.writelines(list(val_video_path))
    file2.close()



