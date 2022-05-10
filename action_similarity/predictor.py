from __future__ import annotations 

from typing import List, Dict, Tuple, TYPE_CHECKING
from tqdm import tqdm
import numpy as np

from action_similarity.dtw import accelerated_dtw

if TYPE_CHECKING:
    from action_similarity.database import ActionDatabase
    from bpe.common_bpe import Config


class Predictor:
    def __init__(self, config: Config, std_db: ActionDatabase):
        self.config = config
        self.std_db = std_db
        self.similarity_analyzer = std_db.similarity_analyzer
    
    def compute_action_similarities(
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
        for action_label, seq_features_list in tqdm(self.std_db.db.items()):
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

    def seq_feature_to_motion_embedding(
        self, 
        seq_features: List[List[np.ndarray]]) -> List[np.ndarray]:
        seq_features_np = np.array(seq_features, dtype=np.ndarray) # #windows x 5, dtype = ndarray
        seq_features_t = seq_features_np.transpose() # 5 x #windows, dtype = ndarray
        motion_embedding = []
        for i in range(len(seq_features_t)):
            motion_embedding.append(np.stack(seq_features_t[i], axis=0))
        return motion_embedding # #body part x #windows x #features

    def compute_action_similarities2(
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
        motion_embedding_anchor = self.seq_feature_to_motion_embedding(anchor)
        
        for action_label, seq_features_list in tqdm(self.std_db.db.items()):
            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for seq_features in seq_features_list:
                motion_embedding = self.seq_feature_to_motion_embedding(seq_features)
                similarity_per_body_part = []
                for i in range(len(motion_embedding_anchor)): # equal to # body part
                    body_part_similarity, _, _, _ = accelerated_dtw(
                        motion_embedding_anchor[i], 
                        motion_embedding[i], 
                        dist_fun='cosine')
                    similarity_per_body_part.append(body_part_similarity)
                #breakpoint()
                similarity = np.mean(similarity_per_body_part)
                similarities_per_actions[action_label].append(similarity)
        return similarities_per_actions

    def predict(
        self, 
        motion_embedding: List[List[np.ndarray]]) -> Tuple[str, float]:
        """
        Params
            motion_embedding: #windows, #body_part, #features
        return action label, similarities
        Predict action based on similarities.
        The action that has the least similarity between reference motion embedding and std_db is determined.  
        """
        similarities_per_actions = self.compute_action_similarities2(motion_embedding)
        actions_similarities_pair = [[], []] # actions, similarities
        for action, similarities in similarities_per_actions.items():
            n = len(similarities)
            actions_similarities_pair[0].extend([action] * n) # actions
            actions_similarities_pair[1].extend(similarities) # similarities
        actions = actions_similarities_pair[0]
        similarities = actions_similarities_pair[1]
        sorted_actions_by_similarity = [(action, similarity) for similarity, action in sorted(zip(similarities, actions))]
        for pair in sorted_actions_by_similarity:
            #print(pair)
            action_label, similarity = pair
            print(f"{self.std_db.actions[action_label]}: {similarity}")
        print()
        for action, similarities in similarities_per_actions.items():
            print(f"mean similarity of {self.std_db.actions[action]}: {np.mean(similarities)}")
                
        k = self.config.k_neighbors
        if k == 1:
            action_label = sorted_actions_by_similarity[0][0]
        else:
            bin_dict = {}
            for i in range(k):
                candidate_label, similarity = sorted_actions_by_similarity[i]
                if candidate_label not in bin_dict:
                    bin_dict[candidate_label] = 1
                else:
                    bin_dict[candidate_label] += 1
            action_label = max(bin_dict, key = bin_dict.get)
        return action_label