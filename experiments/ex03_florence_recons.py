import argparse
import time
from typing import List
from pprint import pprint
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from bpe import Config
from bpe.functional.utils import pad_to_height
from action_similarity.utils import cache_file, Timer, save_file
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from stgcn.predictor import Predictor as STGCNPredictor
from action_similarity.simple_predictor import Predictor as KNNPredicor
from action_similarity.simple_database import ActionDatabase

from .utils import raw_to_keypoints_by_id, normalize_florence_to_ntu_distribution

def main():
    # Florence: (23, 32, 15, 3) max(991.4373503654234, 924.6611328125, 3553.3125), min(-894.992427702873, -1635.0623779296875, 2160.5501275831652)
    # config.data_dir = "experiments_data/data_stgcn/"
    model_path= Path(config.data_dir) / 'model-best.pkl'
    device = 'cuda'

    # config.data_dir = "experiments_data/data_triplet/"
    # model_path= Path(config.data_dir) / 'model-46.pkl'

    x_labels, y_labels = torch.load("./dataset/Florence_3d_actions/train.pkl") 
    x_labels = x_labels.numpy()
    y_labels = y_labels.numpy()
    x_labels, y_labels, = normalize_florence_to_ntu_distribution(x_labels, y_labels)
    
    timer = Timer()
    timer.log("DB")
    print("Compute standard db...")
    db = ActionDatabase(action_label_path=Path(config.data_dir) / 'action_label.txt')
    db.compute_standard_action_database(config=config)
    for action_idx, features in db.db.items():
        print(db.actions[action_idx], len(features))

    print("Predict action...")
    timer.log("predict") 
    stgcn_predictor = STGCNPredictor(model_path=model_path, device=device, model_name='stgcn-recons')
    knn_predictor = KNNPredicor(config=config, std_db=db)
        
    y_preds = []
    tic = time.time()
    for x_label, y_label in tqdm(zip(x_labels, y_labels)):
        x_label = torch.from_numpy(x_label)
        embedding = stgcn_predictor.encode(x_label.to(device))
        predictions = knn_predictor.predict({0: embedding})
        pred = predictions[0]['predictions']['actions'][0]['label']
        y_preds.append(pred)
        #print(pred, y_label)
    toc = time.time()
    
    macro_f1 = metrics.f1_score(y_pred=y_preds, y_true=y_labels, average='macro')
    micro_f1 = metrics.f1_score(y_pred=y_preds, y_true=y_labels, average='micro')
    print(f"macro f1: {macro_f1}")    
    print(f"micro f1: {micro_f1}")    
    print(f"elasp: {toc - tic}")
    print(f"fps: {len(y_labels)/(toc - tic)}")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="experiments_data/data_triplet/", help="path to dataset dir")
    #parser.add_argument('--clustering', type=str, default=None, help="clustering for standard database")
    parser.add_argument('--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('--k_clusters', type=int, default=None, help="number of cluster to use for KMeans")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    parser.add_argument('--update', action='store_true', help="Update database using custom skeleton")
    # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=16,
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