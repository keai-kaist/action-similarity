import os
import json
import pickle
import argparse
from typing import Dict, List
from pathlib import Path

from glob import glob
from tqdm import tqdm
import numpy as np
import torch

from bpe import Config
from bpe.functional.utils import pad_to_height
from bpe.similarity_analyzer import SimilarityAnalyzer
from action_similarity.utils import cache_file, save_embeddings, exist_embeddings, take_best_id

from action_similarity.motion import extract_keypoints, compute_motion_embedding
from experiments.utils import raw_to_keypoints_by_id, normalize_florence

def main(config: Config):
    # from video to embeddings
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    assert not exist_embeddings(config), f"The embeddings(k = {config.k_clusters}) already exist"
    
    n_max_videos_per_class = 4
    dataset_path = Path("dataset/Florence_3d_actions/test.pkl")
    x_labels_test, y_labels_test = torch.load(dataset_path) 
    x_labels_test = x_labels_test.numpy()
    y_labels_test = y_labels_test.numpy()
    dataset_path = Path("dataset/Florence_3d_actions/valid.pkl")
    x_labels_valid, y_labels_valid = torch.load(dataset_path) 
    x_labels_valid = x_labels_valid.numpy()
    y_labels_valid = y_labels_valid.numpy()
    data_dir = Path(config.data_dir)
    model_path= data_dir / 'model_best.pth.tar'
    x_labels = np.concatenate((x_labels_test, x_labels_valid), axis=0)
    y_labels = np.concatenate((y_labels_test, y_labels_valid), axis=0)
    
    x_labels, y_labels, height, width = normalize_florence(x_labels, x_labels)
    h1, w1, scale = pad_to_height(config.img_size[0], height, width)

    db: Dict[int, List] = {}
    similarity_analyzer = SimilarityAnalyzer(config, model_path)
    mean_pose_bpe = np.load(os.path.join(data_dir, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(data_dir, 'stdpose_rc_with_view_unit64.npy'))

    print("Extract keypoints from videos")
    for x_label, y_label in zip(x_labels, y_labels): 
        action_idx = int(y_label)
        if action_idx not in db:
            db[action_idx] = []
        if len(db[action_idx]) == n_max_videos_per_class:
            continue
        #print(f"Current action idx: {action_idx}")
        
        keypoints_by_id = raw_to_keypoints_by_id(x_label)
        
        id = take_best_id(keypoints_by_id)
        seq_features = compute_motion_embedding(
            annotations=keypoints_by_id[id],
            similarity_analyzer=similarity_analyzer,
            mean_pose_bpe=mean_pose_bpe,
            std_pose_bpe=std_pose_bpe,
            scale=scale,
            device=config.device,
        )
        db[action_idx].append(seq_features)

    for key, items in db.items():
        print(f"[{key}] number of videos: {len(items)}")

    save_embeddings(db, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="experiments_data/data_bpe/", required=False, help="path to dataset dir")
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
    main(config)