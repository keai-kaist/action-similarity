from typing import Tuple, List, Dict

import argparse

from dtw import accelerated_dtw
from motion_embedding import MotionEmbedding

from bpe import Config

from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints


def motion_encode(skeleton_path: str) -> MotionEmbedding:
    """
    return dict motion embedding: h1 x T/8, h1=128 for the torso motion encoder and h1=64 for the other encoders, T is not fixed value
    """
    # dummy
    motion_embedding = MotionEmbedding()
    motion_embedding.init_by_random()
    return motion_embedding

def compute_action_similarities(motion_embeding: MotionEmbedding, std_db: Dict[str, List[MotionEmbedding]]) -> Dict[str, List[float]]:
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
            torso_similarity = accelerated_dtw(anchor.torso, motion_embedding.torso, dist_fun='cosine')
            left_leg_similarity = accelerated_dtw(anchor.left_leg, motion_embedding.left_leg, dist_fun='cosine')
            right_leg_similarity = accelerated_dtw(anchor.right_leg, motion_embedding.right_leg, dist_fun='cosine')
            left_arm_similarity = accelerated_dtw(anchor.left_arm, motion_embedding.left_arm, dist_fun='cosine')
            right_arm_similarity = accelerated_dtw(anchor.right_arm, motion_embedding.right_arm, dist_fun='cosine')
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
    video_path = './samples/CCTV.mp4'
    video_path = './samples/S001C001P001R001A007_rgb.avi'
    db = ActionDatabase(
        config=config,
        data_dir=args.data_dir,
        action_label_path='./data/action_label.txt',
        model_path='./data/model_best.pth.tar',
    )
    db.compute_standard_action_database(skeleton_path='./data/custom_skeleton')
    keypoints = extract_keypoints(video_path, fps=4)
    motion_embeding = motion_encode(keypoints=keypoints)
    action_label, similarity = predict_action(motion_embeding, std_db)
    print(f"Predicted action is {action_label}, and similarity is {similarity}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
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
    main()