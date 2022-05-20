import os
import pickle
import argparse
from typing import Dict, List

from glob import glob
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

from bpe import Config
from bpe.functional.utils import pad_to_height
from bpe.similarity_analyzer import SimilarityAnalyzer

from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from action_similarity.predictor import Predictor
from action_similarity.utils import cache_file, seq_feature_to_motion_embedding, time_align
from action_similarity.dtw import accelerated_dtw

def list_transpose(x: List[List]) -> List[List]:
    return list(map(list, zip(*x)))

def motion_embeddings_to_embedding_per_bodypart(
    motion_embeddings: List[List[np.ndarray]]) -> List[np.ndarray]:
    # input value:  #videos x #body part ndarray(x #windows x #features)
    # output value: #body part x ndarray(#windows x #videos x #features)

    # #body part x #videos x ndarray(#windows x #features)
    motion_embeddings_t = list_transpose(motion_embeddings)
    # #body part x ndarray(#windows x #videos x #features)
    embedding_per_bodypart: List[np.ndarray] = []
    for i in range(len(motion_embeddings_t)):
        embedding_per_bodypart.append(np.stack(motion_embeddings_t[i], axis=0).transpose((1, 0, 2))) 
    return embedding_per_bodypart

def main(config: Config):
    # video_path = "custom_data/videos"
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    # embedding_path = "data/embeddings"
    data_path = 'data'
    k_clusters = config.k_clusters
    embeddings_dir = os.path.join(data_path, 'embeddings')
    result_dir = os.path.join(data_path, f'embeddings_k={k_clusters}')
    assert not os.path.exists(result_dir), f"{result_dir} already exists"
    print(f"[db] Load motion embedding from {embeddings_dir}...")
    database: Dict = {}
    for embedding_file in glob(f'{embeddings_dir}/*'):
        with open(embedding_file, 'rb') as f:
            # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
            # seq_features: List[List[List[np.ndarray]]]
            # 64 * (T=16 / 8), 128 * (T=16 / 8)
            seq_features = pickle.load(f)
            file_name = os.path.basename(embedding_file).rstrip(".pickle")
            action_idx = int(file_name.split("_")[-1])
            database[action_idx] = seq_features

    # value: #videos x #body part x (#windows x #features)
    processed_database: Dict = {}
    for action_idx, seq_features in database.items():
        std_motion_embedding = seq_feature_to_motion_embedding(seq_features[0])
        processed_database[action_idx] = [std_motion_embedding]
        # print(f"[{action_idx}] length of videos: {len(std_motion_embedding[0])}")
        for seq_feature in seq_features[1:]:
            motion_embedding = seq_feature_to_motion_embedding(seq_feature)
            processed_embedding = []
            for i in range(5): # #body part
                _, processed_sub_embedding = time_align(std_motion_embedding[i], motion_embedding[i])
                processed_embedding.append(processed_sub_embedding)
            processed_database[action_idx].append(processed_embedding)
        #     print(f"[{action_idx}] length of videos: {len(processed_embedding[0])}")
        # print(f"**[{action_idx}] number of videos: {len(processed_database[action_idx])}")
    # value: #body part x (#windows x #k_clusters x #features)
    
    print(f"[db] K-means clustering to process embeddings...")
    kmeans_database: Dict = {}
    for action_idx, seq_features in tqdm(processed_database.items()):
        kmeans_database[action_idx] = []
        motion_embeddings = processed_database[action_idx]
        embedding_per_bodypart = motion_embeddings_to_embedding_per_bodypart(motion_embeddings)
        
        for embedding_per_windows in embedding_per_bodypart:
            seq_k_features = []
            for features in embedding_per_windows:
                kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(features)
                k_features = kmeans.cluster_centers_
                seq_k_features.append(k_features)
                #kmeans_database[action_idx][-1].append(k_features)
            #np.stack(kmeans_database[action_idx][-1])
            #kmeans_database[action_idx].append([])
            kmeans_database[action_idx].append(np.stack(seq_k_features, axis=0))
            #print(kmeans_database[action_idx][-1].shape)
    
    print(f"[db] save embeddings in {result_dir}...")
    os.mkdir(result_dir)
    for action_idx, seq_k_features_per_bp in kmeans_database.items():
        _motion_embeddings = [] # #bodypart x k_clusters x ndarray(#windows x #features)
        for seq_k_features in seq_k_features_per_bp:
            _seq_k_features = seq_k_features.transpose((1, 0, 2)) # ndarray(#k_clusters x #windows x #features)
            _seq_k_features = list(_seq_k_features) # #k_clusters x ndarray(#windows x #features)
            _motion_embeddings.append(_seq_k_features)
        motion_embeddings = list_transpose(_motion_embeddings) #k_clusters x #body part x ndarray(#windows x #features)

        embeddings_filename = os.path.join(result_dir, f'action_embeddings_{action_idx:03d}.pickle')
        with open(embeddings_filename, 'wb') as f:
            pickle.dump(motion_embeddings, f)
    # save kmeans_database
    # body part x ndarray(#windows x #k_clusters x #features)
    # -> #k_clusters x #body part x ndarray(#windows x #features)
    # or #k_clusters x #windows x #body part x #features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=False, help="path to dataset dir")
    parser.add_argument('--k_neighbers', type=int, default=1, help="number of neighbors to use for KNN")
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