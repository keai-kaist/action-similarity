from __future__ import annotations 

from typing import List, Dict, Tuple, TYPE_CHECKING
from matplotlib.pyplot import jet
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.distance import cosine

from action_similarity.dtw import accelerated_dtw
from action_similarity.utils import seq_feature_to_motion_embedding, time_align

if TYPE_CHECKING:
    from action_similarity.database import ActionDatabase
    from bpe.common_bpe import Config


class Predictor:
    def __init__(self, config: Config, std_db: ActionDatabase):
        self.config = config
        self.std_db = std_db
        self.similarity_analyzer = std_db.similarity_analyzer
        self.cosine_score = torch.nn.CosineSimilarity(dim=0, eps=1e-50)
        # self.cosine_score = cosine

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
        motion_embedding_anchor = seq_feature_to_motion_embedding(anchor)
        
        for action_label, seq_features_list in tqdm(self.std_db.db.items()):
            if not action_label in similarities_per_actions:
                similarities_per_actions[action_label] = []
            # body_part_similarities = []
            for seq_features in seq_features_list:
                motion_embedding = seq_feature_to_motion_embedding(seq_features)
                similarity_per_body_part = []
                for i in range(len(motion_embedding_anchor)): # equal to # body part
                    _, _, _, path = accelerated_dtw(
                        motion_embedding_anchor[i], 
                        motion_embedding[i], 
                        dist_fun='euclidean')
                    path = [(x, y) for x, y in zip(path[0], path[1])] 
                    similarities_per_path = []
                    for j in range(len(path)):
                        #breakpoint()
                        cosine_sim = self.cosine_score(torch.Tensor(motion_embedding_anchor[i][path[j][0]]),
                                                    torch.Tensor(motion_embedding[i][path[j][1]])).numpy()
                        similarities_per_path.append(cosine_sim)
                    total_path_similarity = sum(similarities_per_path) / len(path)
                    similarity_per_body_part.append(total_path_similarity)
                #breakpoint()
                similarity = np.mean(similarity_per_body_part)
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
        for action_label, seq_features_list in tqdm(self.std_db.db.items()):
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
                        #breakpoint()
                        #print("motion_embedding_aligned:", len(motion_embedding_aligned), len(motion_embedding_aligned[i]), motion_embedding_aligned[i][j].shape)
                        #print("motion_embedding:",len(motion_embedding), len(motion_embedding[i]), motion_embedding[i][j].shape)
                        
                        cosine_sim = self.cosine_score(torch.Tensor(motion_embedding_aligned[i][j]),
                                                    torch.Tensor(motion_embedding[i][j])).numpy()
                        similarities.append(cosine_sim)
                    total_path_similarity = sum(similarities) / len(motion_embedding_aligned[i])
                    similarity_per_body_part.append(total_path_similarity)
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
            #print(pair)
            action_label, similarity = pair
            print(f"{self.std_db.actions[action_label]}: {similarity}")
        #print()
        #for action, similarities in similarities_per_actions.items():
        #    print(f"mean similarity of {self.std_db.actions[action]}: {np.mean(similarities)}")
                
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
        return action_label, similarities_per_actions