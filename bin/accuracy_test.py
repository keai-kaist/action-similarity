import os
import argparse
from typing import List
from pathlib import Path
from pprint import pprint
from glob import glob
import random

import numpy as np
from tqdm import tqdm

from bpe import Config
from action_similarity.utils import cache_file, Timer, save_file
from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from action_similarity.predictor import Predictor

""" 
Current accuracy_test hyperparameters:

    Parameters:
        fps (int): framerate of training videos. (current = 10)
        k_neighbors (int): number of neighbors to use for KNN (current = 5)
        k_clusters (int): number of cluster to use for KMeans (current = 0)
        video_sampling_window_size (int): window size to use for similarity prediction (current = 16)
        video_sampling_window_stride (int): stride determining when to start next window of frames (current = 8)
"""
def main():
    random.seed(1234)
    target_actions = [8, 9, 10, 11, 12, 13]
    data_path = Path(config.data_dir)
    video_path = data_path / "testset"
    info = {action_idx: [0, 0] for action_idx in target_actions}
    
    timer = Timer()
    timer.log("DB")
    print("Compute standard db...")
    db = ActionDatabase(
        config=config,
        database_path=data_path / 'embeddings',
        label_path=data_path / 'action_label.txt',
        target_actions=target_actions
    )
    for action_idx in db.db.keys():
        db.db[action_idx] = random.sample(db.db[action_idx], 12)
        #features = random.sample(features, 10)
        print(db.actions[action_idx], len(db.db[action_idx]))
        

    predictor = Predictor(
        config=config, 
        model_path='./data/model_best.pth.tar',
        std_db=db,)

    for video_dir in glob(f'{video_path}/*'):
        action_str = os.path.basename(os.path.normpath(video_dir))
        if not action_str.isdigit(): # 숫자가 아닌 디렉토리인 경우 넘어감
            continue
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        if action_idx not in target_actions:
            continue
        
        print(f"Current action idx: {action_idx}")
        for video_filepath in glob(f'{video_dir}/*'):
            if os.path.splitext(video_filepath)[1] not in ['.mp4', '.avi', '.mkv']:
                continue
            info[action_idx][1] += 1 # 전체 비디오 갯수

            fps = args.fps
            if fps == 30:
                pickle_name = video_filepath
            else:
                basename, ext = os.path.splitext(video_filepath)
                pickle_name = basename + f"_{fps}" + ext
                
            keypoints_by_id = cache_file(pickle_name, extract_keypoints, 
                *(video_filepath,), **{'fps': fps,})
            for id in keypoints_by_id:
                print(video_filepath, len(keypoints_by_id[id]))
            
            predictions = predictor.predict(keypoints_by_id)
            if not predictions: # 빈리스트
                print(f"[Warning] Number of frames lacks, {len(keypoints_by_id[id])}")
                info[action_idx][1] -= 1 # 전체 비디오 갯수
                continue
            if len(predictions[0]['predictions'][0]['actions']) == 0:
                print("[Warning] There is no predicted actions")
                pprint(predictions)
                continue
            predict = predictions[0]['predictions'][0]['actions'][0]['label']
            #print(predict, action_idx)
            if predict == action_idx:
                info[action_idx][0] += 1 # 맞은 갯수
    
    total_n = 0
    total_k = 0
    for id in info:
        k = info[id][0]
        n = info[id][1]
        total_k += k
        total_n += n
        if n == 0:
            print(f"[{id}] {k}/{n}")
        else:
            print(f"[{id}] {k}/{n}, {k/n}")
    print(f"[total] {total_k}/{total_n}, {total_k/total_n}")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="data", help="path to dataset dir")
    parser.add_argument('--k_neighbors', type=int, default=5, help="number of neighbors to use for KNN")
    parser.add_argument('--frames', type=int, default=0, help="number of frames to predict")
    parser.add_argument('--fps', type=int, default=30, required=False, help="fps to embed video")
    
    parser.add_argument('--k_clusters', type=int, default=None, help="number of cluster to use for KMeans")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=8,
                        help='stride determining when to start next window of frames')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")

    parser.add_argument('--similarity_measurement_window_size', type=int, default=1,
                        help='measuring similarity over # of oversampled video sequences')
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--privacy_on', action='store_true',
                        help='when on, no original video or sound in present in the output video')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to seprate positive and negative classes')
    parser.add_argument('--connected_joints', action='store_true', help='connect joints with lines in the output video')

    args = parser.parse_args()
    config = Config(args)
    main()