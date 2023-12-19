import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
from tqdm import tqdm
import csv
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
    vid_name = vid_path[0].split('/')[-1]
    label_index = vid_path[2] 
    type = vid_path[1]

    video = cv2.VideoCapture(vid_path[0])
    # vid_name = vid_path.split('/')[-1].split('.')[0]
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
    if type == 'train':
        train_video_path.append(out_full_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
    elif type == 'test':
        test_video_path.append(out_full_path+' '+str(len(file_list))+' '+str(label_index)+'\n')
    sys.stdout.flush()
    return file_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hmdb dataset preparation")
    parser.add_argument("--src_dir",type=str,default='/media/sdb_access/HMDB_51/all_video/') # give dataset video path
    parser.add_argument("--out_dir", type=str,default='/media/sdb_access/Data_for_Action_CLIP/HMDB51/') # give output path
    parser.add_argument("--num_worker", type=int, default=8)   
    parser.add_argument("--new_width", type=int, default=224, help='resize image width')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
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
    
    class_ind_dict={}
    with open('/media/sdb_access/HMDB_51/hmdb51_labels.csv', mode ='r')as file:       # Give csv path of hmdb dataset
        csvFile = csv.reader(file)
        for cls_ind,lines in enumerate(csvFile):
            class_ind_dict.update({lines[1].replace(' ','_'):cls_ind-1})
    
    
    vid_list = []
    video_list_txts = glob.glob('/media/sdb_access/HMDB_51/test_train_splits/split_1/*')    # Give split path of hmdb dataset
    for  video_list_txt in video_list_txts:
        # print(video_list_txt)
        class_name = video_list_txt.split('/')[-1][:-16]
        class_ind = class_ind_dict[class_name]
        # print(class_name,class_ind)
        video_file = open(video_list_txt , 'r')
        Lines = video_file.readlines()
        for line in Lines:
            video_name = line.split(' ')[0]
            # print('type',int(line.split(' ')[1]))
            type = 'train' if int(line.split(' ')[1]) == 1 or int(line.split(' ')[1]) == 0 else 'test'
            video_path = os.path.abspath(os.path.join(src_path, video_name))
            # print(video_path, type)
            vid_list.append([video_path, type, class_ind])
    print("total number of videos found: ", len(vid_list))
                                         

    pool = Pool(num_worker)
    pool.map(dump_frames, vid_list[:100])

    # for video_path in tqdm(vid_list):
    #     file_list = dump_frames(video_path)

    print(len(train_video_path))
    print(len(test_video_path))
    

    # writing to file
    file1 = open('../dataset_txt/HMDB_51/train.txt', 'w')
    file1.writelines(list(train_video_path))
    file1.close()

    file2 = open('../dataset_txt/HMDB_51/test.txt', 'w')
    file2.writelines(list(test_video_path))
    file2.close()




