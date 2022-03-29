from  typing import Tuple, List, Dict

import numpy as np

from dtw import accelerated_dtw
from motion_embeding import MotionEmbeding

def compute_standard_action_database(data_path) -> Dict(str, List(MotionEmbeding)):
    std_db = {}
    # dummy
    feat1_1 = MotionEmbeding()
    feat1_1.init_by_random()
    feat1_2 = MotionEmbeding()
    feat1_2.init_by_random()
    feat2_1 = MotionEmbeding()
    feat2_1.init_by_random()
    feat2_2 = MotionEmbeding()
    feat2_2.init_by_random()
    feat3_1 = MotionEmbeding()
    feat3_1.init_by_random()
    feat3_2 = MotionEmbeding()
    feat3_2.init_by_random()

    std_db['action1'] = [feat1_1, feat1_2]
    std_db['action2'] = [feat2_1, feat2_2]
    std_db['action3'] = [feat3_1, feat3_2]
    return std_db

def keypoint_detect(video):
    return None # coordinate(2) x nb(5) x T, T is not fixed value

def encode(keypoints) -> MotionEmbeding:
    """
    return dict motion embeding: h1 x T/8, h1=128 for the torso motion encoder and h1=64 for the other encoders, T is not fixed value
    """
    # dummy
    motion_embeding = MotionEmbeding()
    motion_embeding.init_by_random
    return motion_embeding

def compute_action_similarities(motion_embeding: MotionEmbeding, std_db: Dict(str, List(MotionEmbeding))) -> Dict(str, List(float)):
    """
    param motion_embeding: reference motion embeding to recognize
    param std_db: standard action database
    return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embeding and each motion embeding of std_db.
    Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
    """
    # use accelerated_dtw
    pass

def predict_action(motion_embeding: MotionEmbeding, std_db: Dict(str, List(MotionEmbeding))) -> Tuple(str, float):
    """
    return action label, similarities
    Predict action based on similarities.
    The action that has the least similarity between reference motion embeding and std_db is determined.  
    """
    action_similarities_dict = compute_action_similarities(motion_embeding, std_db)
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
    motion_embeding = encode(keypoints=keypoints)
    action_label, similarity = predict_action(motion_embeding, std_db)
    print(f"Predicted action is {action_label}, and similarity is {similarity}")

if __name__ == '__main__':
    main()