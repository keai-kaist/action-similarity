from __future__ import annotations 

import os
from typing import List, Dict, Tuple, TYPE_CHECKING
import numpy as np
import torch
from scipy.spatial.distance import cosine

from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.utils import pad_to_height

from action_similarity.dtw import accelerated_dtw
from action_similarity.utils import seq_feature_to_motion_embedding, time_align
from action_similarity.motion import compute_motion_embedding

if TYPE_CHECKING:
    from action_similarity.database import ActionDatabase
    from bpe.common_bpe import Config


class Predictor:
    def __init__(
        self, 
        config: Config, 
        std_db: ActionDatabase,
    ):
        self.config = config
        self.std_db = std_db
        self.data_path = self.config.data_dir
        self.min_frames = 15

    def compute_action_similarities(
        self, 
        anchor: List[List[np.ndarray]]) -> Dict[str, List[float]]:
        """
        param anchor: reference motion embedding to recognize, #windows x 5 x #features
        param std_db: standard action database, value: #videos x #windows x 5 x #features
        return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embedding and each motion embedding of std_db.
        Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
        """
        # use teslean dtw
        self.config.similarity_window_size = 1
        similarities_per_actions: Dict[str, List[float]] = {} 
        for action_label, seq_features_list in self.std_db.db.items():
            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for seq_features in seq_features_list:
                similarities = self.similarity_analyzer.get_similarity_score(seq_features, anchor, 
                    similarity_window_size=self.config.similarity_window_size)
                similarity = np.mean([
                    ( similarities_fer_frame['ra']
                    + similarities_fer_frame['la']
                    + similarities_fer_frame['rl']
                    + similarities_fer_frame['ra']
                    + similarities_fer_frame['torso'])/5 for similarities_fer_frame in similarities])
                # breakpoint()
                # for body_part_idx in range(5): # number of body part(n_b)
                #     breakpoint()
                #     similarity = accelerated_dtw(anchor[:][body_part_idx], seq_feature[:][body_part_idx], dist_fun='cosine')
                #     body_part_similarities.append(similarity)
                # body_part_similarity = np.mean(body_part_similarities)
                similarities_per_actions[action_label].append(similarity)
        return similarities_per_actions

    def compute_action_similarities2(
        self, 
        embedding_anchor: np.ndarray) -> Dict[str, List[float]]:
        """
        param anchor: reference motion embedding to recognize, #windows x 5 x #features
        param std_db: standard action database, value: #videos x #windows x 5 x #features
        return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embedding and each motion embedding of std_db.
        Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
        """
        # use accelerated_dtw
        similarities_per_actions: Dict[str, List[float]] = {} 
        #motion_embedding_anchor = seq_feature_to_motion_embedding(anchor)
        
        for action_label, embedding_list in self.std_db.db.items():
            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for embedding in embedding_list:
                #motion_embedding = seq_feature_to_motion_embedding(seq_features)
                similarity = 1-cosine(embedding_anchor, embedding)
                similarities_per_actions[action_label].append(similarity)
        return similarities_per_actions
    
    def compute_action_similarities_k(
        self, 
        anchor: List[List[np.ndarray]]) -> Dict[str, List[float]]:
        """
        param anchor: reference motion embedding to recognize, #windows x 5 x #features
        param std_db: standard action database, value: #videos x #windows x 5 x #features
        return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embedding and each motion embedding of std_db.
        Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
        """
        # use accelerated_dtw
        similarities_per_actions: Dict[str, List[float]] = {} 
        # #body part x #windows x #features
        motion_embedding_anchor = seq_feature_to_motion_embedding(anchor)
        for action_label, seq_features_list in self.std_db.db.items():
            motion_embedding_aligned = []
            for i in range(len(motion_embedding_anchor)): # #bodypart
                _, sub_embedding = time_align(seq_features_list[0][i], motion_embedding_anchor[i])
                motion_embedding_aligned.append(sub_embedding)

            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for seq_features in seq_features_list:
                # #body part x #windows x #features
                motion_embedding = seq_features # already processed in postprocess.py
                similarity_per_body_part = []
                for i in range(len(motion_embedding_aligned)): # equal to # body part
                    similarities = []
                    for j in range(len(motion_embedding_aligned[i])):                        
                        # cosine_sim = self.cosine_score(torch.Tensor(motion_embedding_aligned[i][j]),
                        #                             torch.Tensor(motion_embedding[i][j])).numpy()
                        cosine_sim = 1-cosine(motion_embedding_aligned[i][j], motion_embedding[i][j])

                        similarities.append(cosine_sim)
                    total_path_similarity = sum(similarities) / len(motion_embedding_aligned[i])
                    similarity_per_body_part.append(total_path_similarity)
                #breakpoint()
                similarity = np.mean(similarity_per_body_part)
                similarities_per_actions[action_label].append(similarity)
        return similarities_per_actions

    def make_prediction(
        self,
        id: int,
        annotations: List[Dict],
        action_label: str,
        score: float):

        prediction = {}
        prediction['id'] = id
        prediction['predictions'] = {}
        # 예측에 사용한 첫번째 프레임의 정보

        prediction['predictions']['frame'] = None
        prediction['predictions']['box'] = None
        prediction['predictions']['score'] = None
        prediction['predictions']['actions'] = []
        action = {'label': action_label, 'score': score}
        prediction['predictions']['actions'].append(action)
        return prediction

    def valid_frames(
        self,
        annotations: List[Dict]):
        return len(annotations) >= self.min_frames
    
    def info(self):
        return self._action_label_per_id, self._similarities_per_id

    def predict(
        self,
        embedding_by_id: Dict[str, np.ndarray]):
        # keypoints_by_id 거르는 함수, 양식 맞추는 함수 필요
        predictions = []
        action_label_per_id = {}
        similarities_per_id = {}
        for id, embedding in embedding_by_id.items():
            action_label, score, similarities_per_actions = self._predict(embedding)
            prediction = self.make_prediction(id, annotations, action_label, score)
            predictions.append(prediction)
            action_label_per_id[id] = action_label
            similarities_per_id[id] = similarities_per_actions
        self._action_label_per_id = action_label_per_id
        self._similarities_per_id = similarities_per_id
        return predictions

    def _predict(
        self, 
        motion_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Params
            motion_embedding: #windows, #body_part, #features
        return action label, similarities
        Predict action based on similarities.
        The action that has the least similarity between reference motion embedding and std_db is determined.  
        """

        if self.config.clustering:
            similarities_per_actions = self.compute_action_similarities_k(motion_embedding)
        else:
            similarities_per_actions = self.compute_action_similarities2(motion_embedding)
        actions_similarities_pair = [[], []] # actions, similarities
        for action, similarities in similarities_per_actions.items():
            n = len(similarities)
            actions_similarities_pair[0].extend([action] * n) # actions
            actions_similarities_pair[1].extend(similarities) # similarities
        actions = actions_similarities_pair[0]
        similarities = actions_similarities_pair[1]
        sorted_actions_by_similarity = [(action, similarity) for similarity, action in sorted(zip(similarities, actions), reverse=True)]
        for pair in sorted_actions_by_similarity:
            action_label, similarity = pair
     
        k = self.config.k_neighbors
        if k == 1:
            action_label = sorted_actions_by_similarity[0][0]
            score = sorted_actions_by_similarity[0][1]
        else:
            bin_dict = {}
            score_dict = {}
            for i in range(k):
                candidate_label, similarity = sorted_actions_by_similarity[i]
                if candidate_label not in bin_dict:
                    bin_dict[candidate_label] = 1
                    score_dict[candidate_label] = [similarity]
                else:
                    bin_dict[candidate_label] += 1
                    score_dict[candidate_label].append(similarity)
            action_label = max(bin_dict, key = bin_dict.get)
            score = np.mean(score_dict[action_label])
        return action_label, score, similarities_per_actions