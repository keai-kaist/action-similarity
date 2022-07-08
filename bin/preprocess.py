import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List

from glob import glob
from torch import embedding
from tqdm import tqdm
import numpy as np

from bpe import Config
from bpe.functional.utils import pad_to_height
from bpe.similarity_analyzer import SimilarityAnalyzer
from action_similarity.utils import cache_file, save_embeddings, exist_embeddings, take_best_id

from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from action_similarity.predictor import Predictor

def main(config: Config):
    # from video to embeddings
    data_dir = Path(args.data_dir)
    video_path = data_dir/"videos"
    # skeleton_path = "custom_data/custom_skeleton"
    # embedding_path = "custom_data/embeddings"
    data_path = data_dir
    model_path=data_dir/'model_best.pth.tar'
    k_clusters = config.k_clusters
    embedding_dir = Path(config.data_dir) / 'embeddings'
    #assert not exist_embeddings(config, embeddings_dir=embedding_dir), f"The embeddings(k = {k_clusters}) already exist"
    
    height, width = 1080, 1920
    h1, w1, scale = pad_to_height(config.img_size[0], height, width)

    db: Dict[int, List] = {}
    similarity_analyzer = SimilarityAnalyzer(config, model_path)
    mean_pose_bpe = np.load(os.path.join(data_path, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(data_path, 'stdpose_rc_with_view_unit64.npy'))

    print("Extract keypoints from videos")
    for video_dir in glob(f'{video_path}/*'): 
        action_idx = int(os.path.basename(os.path.normpath(video_dir)))
        db[action_idx] = []
        print(f"Current action idx: {action_idx}")
        for video_filepath in tqdm(glob(f'{video_dir}/*')):
            if os.path.splitext(video_filepath)[1] not in ['.mp4', '.avi', '.mkv']:
                continue
            # print(count)
            # count += 1
            # if count == 3:
            #     break
            if args.fps is None or args.fps=='30':
                fps=30
                pickle_name = video_filepath
            else:
                fps = int(args.fps)
                basename, ext = os.path.splitext(video_filepath)
                pickle_name = basename + f"_{fps}" + ext
            keypoints_by_id = cache_file(pickle_name, extract_keypoints, 
                *(video_filepath,), **{'fps':fps,})
            #print(video_filepath)
            #print(json.dumps(keypoints_by_id, indent = 4))
            #breakpoint()
            id = take_best_id(keypoints_by_id)
            seq_features = compute_motion_embedding(
                annotations=keypoints_by_id[id],
                similarity_analyzer=similarity_analyzer,
                mean_pose_bpe=mean_pose_bpe,
                std_pose_bpe=std_pose_bpe,
                scale=scale,
                device=config.device,
            )
            if len(seq_features) == 0:
                print(f"[Warning] length of keypoints is not enough, {len(keypoints_by_id[id])}")
                continue
                
            db[action_idx].append(seq_features)

    save_embeddings(db, config, embeddings_dir=embedding_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="data", required=False, help="path to dataset dir")
    parser.add_argument('--fps', default=None, required=False, help="fps to embed video")
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