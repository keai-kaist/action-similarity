from typing import Tuple, List, Dict

import os
import re
import argparse
import json
import pickle
import base64

import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import moviepy.editor as mpy
import requests

from pprint import pprint

from dtw import accelerated_dtw
from motion_embeding import MotionEmbeding

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.motion import preprocess_motion2d_rc, cocopose2motion
from bpe.functional.utils import pad_to_height
from bpe.functional.visualization import preprocess_sequence

def parse_action_label(action_label):
    actions = {}
    with open(action_label) as f:
        lines = f.readlines()
        for line in lines:
            no, action = line.split(None, maxsplit=1)
            no = int(re.search(r'\d+', no).group())
            actions[no] = action.strip()
    return actions

def compute_standard_action_database(data_dir: str, model_path: str) -> Dict[str, List[MotionEmbeding]]:

    # TODO:
    # refined_skeleton format 대신 brain format 사용해서 skeleton parsing 할 수 있게끔 변경

    label_path = os.path.join(data_dir, 'action_label.txt')
    skeleton_path = os.path.join(data_dir, 'refined_skeleton')
    actions = parse_action_label(label_path)
    pprint(actions)

    mean_pose_bpe = np.load(os.path.join(data_dir, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(data_dir, 'stdpose_rc_with_view_unit64.npy'))
    
    height, width = 1080, 1920
    h1, w1, scale = pad_to_height(config.img_size[0], height, width)

    similarity_analyzer = SimilarityAnalyzer(config, model_path)

    db = {}

    for action_dir in glob(f'{skeleton_path}/*'): 
        action_idx = int(os.path.basename(os.path.normpath(action_dir)))
        db[action_idx] = []
        print(f'create embeddings of action: {actions[action_idx]}...')
        for skeleton in tqdm(glob(f'{action_dir}/*')):
            seq = cocopose2motion(
                num_joints=config.unique_nr_joints, 
                json_dir=skeleton,
                scale=scale,
            )

            seq = preprocess_sequence(seq)
            seq_origin = preprocess_motion2d_rc(
                motion=seq,
                mean_pose=mean_pose_bpe,
                std_pose=std_pose_bpe,
            )

            # move input to device
            seq_origin = seq_origin.to(config.device)

            seq_features = similarity_analyzer.get_embeddings(
                seq=seq_origin,
                video_window_size=16,
                video_stride=2,
            )
            db[action_idx].append(seq_features)

    embeddings_dir = os.path.join(data_dir, 'embeddings')
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)
    
    for action_idx, seq_features in db.items():
        embeddings_filename = os.path.join(embeddings_dir, f'action_embeddings_{action_idx:03d}.pickle')
        with open(embeddings_filename, 'wb') as f:
            # seq_features.shape == (#videos, #windows, 5, 128 or 256)
            # seq_features: List[List[List[np.ndarray]]]
            # 64 * (T=16 / 8), 128 * (T=16 / 8)
            pickle.dump(seq_features, f)

    return db

def extract_keypoints(video_path: str):
    
    # TODO
    # 1. convert brain skeleton format to bpe skeleton format
    # --> brain format 자체를 refined_skeleton과 동일하게 변경하거나,
    # --> brain format 도 사용할 수 있도록 코드를 만들거나,
    # 2. dealing with multiple people
    # --> 여러 사람이 동시에 이미지에 등장하는 경우
    # --> object tracking 기능 활용해서 tracker_id 마다 skeleton json 생성
    # ! bpe의 경우:
    # - 한 영상안에 1. 한 사람이, 2. 계속해서 출현하는 것을 가정
    # - track_id 마다 (사람마다) keypoints_sequence 생성
    # - 그런 형식을 만든 다음 연동 방법 강구

    video_name, _ = os.path.splitext(video_path)
    images_path = os.path.join(video_name, 'images')
    os.makedirs(images_path, exist_ok=True)
    
    json_path = os.path.join(video_name, 'json')
    os.makedirs(json_path, exist_ok=True)

    clip = mpy.VideoFileClip(video_path)
    w, h = clip.size
    fps = 1
    step = 1 / fps
    print('fps:', clip.fps, '#frmaes:', len(np.arange(0, clip.duration, step)))
    for i, timestep in tqdm(enumerate(np.arange(0, clip.duration, step))):
        frame_name = os.path.join(images_path, f'frame{i:03d}.jpg')
        clip.save_frame(frame_name, timestep)

    # vid = cv2.VideoCapture(video_path)
    # print(video_path)
    # success, img = vid.read()
    # count = 0
    # while success:
    #     cv2.imwrite(os.path.join(images_path, f'frame{count:03d}.jpg'), img)
    #     success, img = vid.read()
    #     count += 1
    
    tracker_id = None
    url = 'https://brain.keai.io/vision'
    keypoints_by_id = {}
    for filename in tqdm(sorted(glob(f'{images_path}/*.jpg'))):
        with open(filename, 'rb') as input_file:
            image_bytes = input_file.read()
        
        if tracker_id is None:
            response = requests.post(
                url=f'{url}/keypoints',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'tracking': True,
                })
        else:
            response = requests.put(
                url=f'{url}/keypoints/{tracker_id}',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                }
            )
        response_json = response.json()
        for keypoints in response_json['keypoints']:
            track_id = keypoints['track_id']
            if track_id not in keypoints_by_id:
                keypoints_by_id[track_id] = []
            keypoints_by_id[track_id].append(keypoints)
        
        json_filename, _ = os.path.splitext(os.path.basename(filename))
        with open(os.path.join(json_path, f'{json_filename}.json'), 'w') as f:
            json.dump(response_json, f, indent=4)

    video_basename = os.path.basename(video_path)

    skeletons = {
        'video': {
            'path': os.path.basename(video_path),
            'width': clip.w,
            'height': clip.h,
            'original_length': clip.duration,
            'fps': clip.fps,
            'track_id': 1
        },
        # 'annotations': [{
        #     'frame_num': i,
        # }]
    }

    

    return skeletons # coordinate(2) x nb(5) x T, T is not fixed value

def motion_encode(skeleton_path: str) -> MotionEmbeding:
    """
    return dict motion embeding: h1 x T/8, h1=128 for the torso motion encoder and h1=64 for the other encoders, T is not fixed value
    """
    # dummy
    motion_embeding = MotionEmbeding()
    motion_embeding.init_by_random
    return motion_embeding

def compute_action_similarities(motion_embeding: MotionEmbeding, std_db: Dict[str, List[MotionEmbeding]]) -> Dict[str, List[float]]:
    """
    param motion_embeding: reference motion embeding to recognize
    param std_db: standard action database
    return Dict(str, List(float) similarities: Averages of similarity for each body part between reference motion embeding and each motion embeding of std_db.
    Similarity for each body part is computed by average cosine similarity between the embedding pairs in path. 
    """
    # use accelerated_dtw
    pass

def predict_action(motion_embeding: MotionEmbeding, std_db: Dict[str, List[MotionEmbeding]]) -> Tuple[str, float]:
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
    video_path = './samples/CCTV.mp4'
    model_path = './data/model_best.pth.tar'
    data_dir = args.data_dir
    std_db = compute_standard_action_database(data_dir, model_path)
    keypoints = extract_keypoints(video_path)
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