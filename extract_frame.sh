SRC_FOLDER=$1
OUT_FOLDER=$2
NUM_WORKER=$3

echo "Extracting optical flow from videos in folder: ${SRC_FOLDER}"
# python data_creation.py ${SRC_FOLDER} ${OUT_FOLDER} --num_worker ${NUM_WORKER} --new_width 340 --new_height 256
python data_creation_kinetic.py --src_dir /media/sdb_access/Kinetics/kinetics-dataset/k400 --out_dir /media/sdb_access/Data_for_Action_CLIP/KINETICS_400_2 --num_worker 8 --new_width 224 --new_height 224