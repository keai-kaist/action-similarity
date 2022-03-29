from typing import Tuple, List, Dict

import numpy as np

from dtw import accelerated_dtw
from motion_embedding import MotionEmbedding

def compute_standard_action_database(data_path) -> Dict[str, List[MotionEmbedding]]:
    std_db = {}
    # dummy
    feat1_1 = MotionEmbedding()
    feat1_1.init_by_random()
    feat1_2 = MotionEmbedding()
    feat1_2.init_by_random()
    feat2_1 = MotionEmbedding()
    feat2_1.init_by_random()
    feat2_2 = MotionEmbedding()
    feat2_2.init_by_random()
    feat3_1 = MotionEmbedding()
    feat3_1.init_by_random()
    feat3_2 = MotionEmbedding()
    feat3_2.init_by_random()

    std_db['action1'] = [feat1_1, feat1_2]
    std_db['action2'] = [feat2_1, feat2_2]
    std_db['action3'] = [feat3_1, feat3_2]
    return std_db

def keypoint_detect(video):
    return None # coordinate(2) x nb(5) x T, T is not fixed value

def motion_encode(keypoints) -> MotionEmbedding:
    """
    return dict motion embedding: h1 x T/8, h1=128 for the torso motion encoder and h1=64 for the other encoders, T is not fixed value
    """
    # dummy
    motion_embedding = MotionEmbedding()
    motion_embedding.init_by_random()
    return motion_embedding

def compute_action_similarities(anchor: MotionEmbedding, std_db: Dict[str, List[MotionEmbedding]]) -> Dict[str, List[float]]:
    """
    param anchor: reference motion embedding to recognize
    param std_db: standard action database
    return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embedding and each motion embedding of std_db.
    Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
    """
    # use accelerated_dtw
    action_similarities: Dict[str, List[float]] = {} 
    for action_label, motion_embeddings in std_db.items():
        if not action_label in action_similarities:
            action_similarities[action_label] = []
        for motion_embedding in motion_embeddings:
            # 평균값인지 합인지 확인하기
            torso_similarity = accelerated_dtw(anchor.torso, motion_embedding.torso)
            left_leg_similarity = accelerated_dtw(anchor.left_leg, motion_embedding.left_leg)
            right_leg_similarity = accelerated_dtw(anchor.right_leg, motion_embedding.right_leg)
            left_arm_similarity = accelerated_dtw(anchor.left_arm, motion_embedding.left_arm)
            right_arm_similarity = accelerated_dtw(anchor.right_arm, motion_embedding.right_arm)
            similarity = np.mean([torso_similarity, left_leg_similarity, right_leg_similarity, left_arm_similarity, right_arm_similarity])
            action_similarities[action_label].append(similarity)
    return action_similarities

def predict_action(motion_embedding: MotionEmbedding, std_db: Dict[str, List[MotionEmbedding]]) -> Tuple[str, float]:
    """
    return action label, similarities
    Predict action based on similarities.
    The action that has the least similarity between reference motion embedding and std_db is determined.  
    """
    action_similarities_dict = compute_action_similarities(motion_embedding, std_db)
    actions_similarities_pair = [[], []] # actions, similarities
    for action, similarities in action_similarities_dict:
        n = len(similarities)
        actions_similarities_pair[0].extend([action] * n) # actions
        actions_similarities_pair[1].extend(similarities) # similarities
    actions = actions_similarities_pair[0]
    similarities = actions_similarities_pair[1]
    sorted_actions_by_similarity = [x for _, x in sorted(zip(similarities, actions))]
    # TODO: consider k actions of similarities not only 1.  
    return sorted_actions_by_similarity[0]

def main():
    video = None
    std_db = compute_standard_action_database(None)
    keypoints = keypoint_detect(video)
    motion_embedding = motion_encode(keypoints=keypoints)
    action_label, similarity = predict_action(motion_embedding, std_db)
    print(f"Predicted action is {action_label}, and similarity is {similarity}")

if __name__ == '__main__':
    main()