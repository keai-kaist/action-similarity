import os
import pickle
import argparse
from typing import Dict, List

from glob import glob
from tqdm import tqdm
import numpy as np

from bpe import Config
from bpe.functional.utils import pad_to_height
from bpe.similarity_analyzer import SimilarityAnalyzer
from utils import cache_file

from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from action_similarity.predictor import Predictor

def main(config: Config):
    video_path = "custom_data/videos"
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    data_path = "custom_data/"
    model_path='custom_data/model_best.pth.tar'

    height, width = 1080, 1920
    h1, w1, scale = pad_to_height(config.img_size[0], height, width)

    db: Dict[int, List] = {}
    similarity_analyzer = SimilarityAnalyzer(config, model_path)
    mean_pose_bpe = np.load(os.path.join(data_path, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(data_path, 'stdpose_rc_with_view_unit64.npy'))

    for video_dir in glob(f'{video_path}/*'): 
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        db[action_idx] = []
        for video_filepath in tqdm(glob(f'{video_dir}/*')):
            keypoints_by_id = extract_keypoints(video_filepath, fps=30)
            # TODO: skeleton도 저장하기
            seq_features = compute_motion_embedding(
                skeletons_json_path=keypoints_by_id,
                similarity_analyzer=similarity_analyzer,
                mean_pose_bpe=mean_pose_bpe,
                std_pose_bpe=std_pose_bpe,
                scale=scale,
                device=config.device,
            )
            db[action_idx].append(seq_features)

    embeddings_dir = os.path.join(data_path, 'embeddings')
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    for action_idx, seq_features in db.items():
        embeddings_filename = os.path.join(embeddings_dir, f'action_embeddings_{action_idx:03d}.pickle')
        with open(embeddings_filename, 'wb') as f:
            # seq_features.shape == (#videos, #windows, 5, 128 or 256)
            # seq_features: List[List[List[np.ndarray]]]
            # 64 * (T=16 / 8), 128 * (T=16 / 8)
            pickle.dump(seq_features, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=False, help="path to dataset dir")
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