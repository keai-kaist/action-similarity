from typing import Dict, List

import os
import pickle

import numpy as np
from glob import glob
from tqdm import tqdm

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer

from bpe.functional.utils import pad_to_height

from action_similarity.utils import parse_action_label
from action_similarity.motion import compute_motion_embedding

class ActionDatabase():

    def __init__(
        self,
        config: Config,
        action_label_path: str,
    ):
        self.config = config
        self.actions = parse_action_label(action_label_path)
        self.db = {}

        self.scale = None

    def compute_standard_action_database(
        self, 
        skeleton_path: str, 
        data_path: str,
        model_path: str
    ):
        self.similarity_analyzer = SimilarityAnalyzer(self.config, model_path)
        self.mean_pose_bpe = np.load(os.path.join(data_path, 'meanpose_rc_with_view_unit64.npy'))
        self.std_pose_bpe = np.load(os.path.join(data_path, 'stdpose_rc_with_view_unit64.npy'))
    
        if not self.config.update:
            height, width = 1080, 1920
            h1, w1, self.scale = pad_to_height(self.config.img_size[0], height, width)
            embeddings_dir = os.path.join(data_path, 'embeddings')
            print(f"[db] Load motion embedding from {embeddings_dir}...")
            self.db = {}
            for embedding_file in glob(f'{embeddings_dir}/*'):
                with open(embedding_file, 'rb') as f:
                    # seq_features.shape == (#videos, #windows, 5, 128[0:4] or 256[4])
                    # seq_features: List[List[List[np.ndarray]]]
                    # 64 * (T=16 / 8), 128 * (T=16 / 8)
                    seq_features = pickle.load(f)
                    file_name = os.path.basename(embedding_file).rstrip(".pickle")
                    action_idx = int(file_name.split("_")[-1])
                    self.db[action_idx] = seq_features
                    # TODO: Should remove below line later
                    # Use only 5 kinds of actions and 6 videos each
                    # self.db[action_idx] = self.db[action_idx][:6] 
                    # if len(self.db) > 5:
                    #     break
                     
        else:
            height, width = 1080, 1920
            h1, w1, self.scale = pad_to_height(self.config.img_size[0], height, width)
            
            self.db = {}
            for action_dir in glob(f'{skeleton_path}/*'): 
                action_idx = int(os.path.basename(os.path.normpath(action_dir)))
                self.db[action_idx] = []
                print(f'create embeddings of action: {self.actions[action_idx]}...')
                for skeleton in tqdm(glob(f'{action_dir}/*')):
                    seq_features = compute_motion_embedding(
                        skeletons_json_path=skeleton,
                        similarity_analyzer=self.similarity_analyzer,
                        mean_pose_bpe=self.mean_pose_bpe,
                        std_pose_bpe=self.std_pose_bpe,
                        scale=self.scale,
                        device=self.config.device,
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

    def load_database(self, db_path: str):
        self.db = {}
        for db_filename in glob(db_path + '/*.pickle'):
            db_basename, _ = os.path.splitext(os.path.basename(db_filename))
            # 'action_embeddings_001' --> '001' --> 1
            action_idx = int(db_basename.split('_')[-1])
            with open(db_filename, 'rb') as f:
                embeddings = pickle.load(f)
                self.db[action_idx] = embeddings
