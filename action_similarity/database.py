from typing import Dict, List

import os
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.motion import preprocess_motion2d_rc
# , cocopose2motion
from bpe.functional.utils import pad_to_height
from bpe.functional.visualization import preprocess_sequence

from action_similarity.utils import parse_action_label
from action_similarity.motion import cocopose2motion

class ActionDatabase():

    def __init__(
        self,
        config: Config,
        data_dir: str,
        action_label_path: str,
        model_path: str,
    ):
        self.config = config
        self.data_dir = data_dir
        self.actions = parse_action_label(action_label_path)
        self.similarity_analyzer = SimilarityAnalyzer(config, model_path)

        self.mean_pose_bpe = np.load(os.path.join(data_dir, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(data_dir, 'stdpose_rc_with_view_unit64.npy'))



    def compute_standard_action_database(self, skeleton_path: str):

        # TODO:
        # refined_skeleton format 대신 brain format 사용해서 skeleton parsing 할 수 있게끔 변경

        height, width = 1080, 1920
        h1, w1, scale = pad_to_height(self.config.img_size[0], height, width)

        self.db = {}
        for action_dir in glob(f'{skeleton_path}/*'): 
            action_idx = int(os.path.basename(os.path.normpath(action_dir)))
            self.db[action_idx] = []
            print(f'create embeddings of action: {self.actions[action_idx]}...')
            for skeleton in tqdm(glob(f'{action_dir}/*')):
                seq = cocopose2motion(
                    # num_joints=self.config.unique_nr_joints, 
                    json_dir=skeleton,
                    scale=scale,
                )

                seq = preprocess_sequence(seq)
                seq_origin = preprocess_motion2d_rc(
                    motion=seq,
                    mean_pose=self.mean_pose_bpe,
                    std_pose=self.std_pose_bpe,
                )

                # move input to device
                seq_origin = seq_origin.to(self.config.device)

                seq_features = self.similarity_analyzer.get_embeddings(
                    seq=seq_origin,
                    video_window_size=16,
                    video_stride=2,
                )
                self.db[action_idx].append(seq_features)

        embeddings_dir = os.path.join(self.data_dir, 'embeddings')
        if not os.path.exists(embeddings_dir):
            os.mkdir(embeddings_dir)
        
        for action_idx, seq_features in self.db.items():
            embeddings_filename = os.path.join(embeddings_dir, f'action_embeddings_{action_idx:03d}.pickle')
            with open(embeddings_filename, 'wb') as f:
                # seq_features.shape == (#videos, #windows, 5, 128 or 256)
                # seq_features: List[List[List[np.ndarray]]]
                # 64 * (T=16 / 8), 128 * (T=16 / 8)
                pickle.dump(seq_features, f)