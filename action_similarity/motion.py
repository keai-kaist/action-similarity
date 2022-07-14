import os
import base64
import json
from typing import Dict, List, Union
import requests
import shutil

from tqdm import tqdm
from glob import glob
import numpy as np
from scipy.ndimage import gaussian_filter1d

from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.motion import preprocess_motion2d_rc
from bpe.functional.visualization import preprocess_sequence


def cocopose2motion(json_dir: Union[str, Dict], scale: float = 1.0, num_joints: int = 15):
    if isinstance(json_dir, str):
        # json_dir is path of keypoints
        with open(json_dir) as f:
            annotations = json.load(f)
            annotations = annotations['annotations']
    elif isinstance(json_dir, Dict):
        # json_dir is keypoints_by_id object
        # id는 한개만 있다고 가정
        _, keypoints = json_dir.popitem()  # id, keypoints
        annotations = keypoints['annotations']
    else:
        raise NotImplementedError

    motion = []
    for anno in annotations:
        keypoints = anno['keypoints']
        nose = np.array([keypoints['nose']['x'], keypoints['nose']['y']])
        right_shoulder = np.array([keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']])
        right_elbow = np.array([keypoints['right_elbow']['x'], keypoints['right_elbow']['y']])
        right_wrist = np.array([keypoints['right_wrist']['x'], keypoints['right_wrist']['y']])
        left_shoulder = np.array([keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']])
        left_elbow = np.array([keypoints['left_elbow']['x'], keypoints['left_elbow']['y']])
        left_wrist = np.array([keypoints['left_wrist']['x'], keypoints['left_wrist']['y']])
        right_hip = np.array([keypoints['right_hip']['x'], keypoints['right_hip']['y']])
        right_knee = np.array([keypoints['right_knee']['x'], keypoints['right_knee']['y']])
        right_ankle = np.array([keypoints['right_ankle']['x'], keypoints['right_ankle']['y']])
        left_hip = np.array([keypoints['left_hip']['x'], keypoints['left_hip']['y']])
        left_knee = np.array([keypoints['left_knee']['x'], keypoints['left_knee']['y']])
        left_ankle = np.array([keypoints['left_ankle']['x'], keypoints['left_ankle']['y']])
        neck = (right_shoulder + left_shoulder) / 2
        mid_hip = (right_hip + left_hip) / 2

        joint = np.array([
            nose,
            neck,
            right_shoulder,
            right_elbow,
            right_wrist,
            left_shoulder,
            left_elbow,
            left_wrist,
            mid_hip,
            right_hip,
            right_knee,
            right_ankle,
            left_hip,
            left_knee,
            left_ankle,
        ])
        if len(motion) > 0:
            joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
        motion.append(joint)
    
    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]
    motion = np.array(motion)
    motion = np.stack(motion, axis=2)

    motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    # TODO: check mean_height available
    motion = motion * scale

    return motion
    
def custompose2motion(annotations: List[Dict], scale: float = 1.0, num_joints: int = 15):
    assert isinstance(annotations, List), f"type of input should be list but, {type(annotations)}"

    motion = []
    for anno in annotations:
        keypoints = anno['keypoints']
        nose = np.array([keypoints['nose']['x'], keypoints['nose']['y']])
        right_shoulder = np.array([keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']])
        right_elbow = np.array([keypoints['right_elbow']['x'], keypoints['right_elbow']['y']])
        right_wrist = np.array([keypoints['right_wrist']['x'], keypoints['right_wrist']['y']])
        left_shoulder = np.array([keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']])
        left_elbow = np.array([keypoints['left_elbow']['x'], keypoints['left_elbow']['y']])
        left_wrist = np.array([keypoints['left_wrist']['x'], keypoints['left_wrist']['y']])
        right_hip = np.array([keypoints['right_hip']['x'], keypoints['right_hip']['y']])
        right_knee = np.array([keypoints['right_knee']['x'], keypoints['right_knee']['y']])
        right_ankle = np.array([keypoints['right_ankle']['x'], keypoints['right_ankle']['y']])
        left_hip = np.array([keypoints['left_hip']['x'], keypoints['left_hip']['y']])
        left_knee = np.array([keypoints['left_knee']['x'], keypoints['left_knee']['y']])
        left_ankle = np.array([keypoints['left_ankle']['x'], keypoints['left_ankle']['y']])
        neck = (right_shoulder + left_shoulder) / 2
        mid_hip = (right_hip + left_hip) / 2

        joint = np.array([
            nose,
            neck,
            right_shoulder,
            right_elbow,
            right_wrist,
            left_shoulder,
            left_elbow,
            left_wrist,
            mid_hip,
            right_hip,
            right_knee,
            right_ankle,
            left_hip,
            left_knee,
            left_ankle,
        ])
        if len(motion) > 0:
            joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
        motion.append(joint)
    
    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]
    motion = np.array(motion)
    motion = np.stack(motion, axis=2)

    motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    # TODO: check mean_height available
    motion = motion * scale

    return motion

def compute_motion_embedding(
    annotations: List[Dict],
    similarity_analyzer: SimilarityAnalyzer,
    mean_pose_bpe: np.ndarray,
    std_pose_bpe: np.ndarray,
    scale: float,
    video_window_size: int = 16,
    video_stride: int = 2,
    device: str = 'cuda') -> List[List[np.ndarray]]:
    """
    Params
    skeletons_json_path: keypoints_by_id 객체 또는 해당 객체가 저장된 json 경로

    """
    seq = custompose2motion(
        annotations=annotations,
        scale=scale,
    )

    seq = preprocess_sequence(seq)
    seq_origin = preprocess_motion2d_rc(
        motion=seq,
        mean_pose=mean_pose_bpe,
        std_pose=std_pose_bpe,
    )
    
    seq_origin = seq_origin.to(device)

    seq_features = similarity_analyzer.get_embeddings(
        seq=seq_origin,
        video_window_size=video_window_size,
        video_stride=video_stride,
    )

    return seq_features


def extract_keypoints(video_path: str, fps: int) -> Dict[int, Dict]:
    
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

    import moviepy.editor as mpy

    video_name, _ = os.path.splitext(video_path)
    images_path = os.path.join(video_name, 'images')
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.makedirs(images_path, exist_ok=True)
    
    json_path = os.path.join(video_name, 'json')
    if os.path.exists(json_path):
        shutil.rmtree(json_path)
    os.makedirs(json_path, exist_ok=True)

    clip = mpy.VideoFileClip(video_path)
    for i, timestep in enumerate(np.arange(0, clip.duration, 1 / fps)):
        frame_name = os.path.join(images_path, f'frame{i:03d}.jpg')
        clip.save_frame(frame_name, timestep)

    tracker_id = None
    url = 'https://brain.keai.io/vision'
    keypoints_by_id = {}
    for i, filename in enumerate(sorted(glob(f'{images_path}/*.jpg'))):
        with open(filename, 'rb') as input_file:
            image_bytes = input_file.read()
        
        if tracker_id is None:
            response = requests.post(
                url=f'{url}/keypoints',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'deviceID': 'demo',
                    'tracking': True,
                })
        else:
            response = requests.put(
                url=f'{url}/keypoints/{tracker_id}',
                json={
                    'image': base64.b64encode(image_bytes).decode(),
                    'deviceID': 'demo',
                }
            )
        response_json = response.json()
        #breakpoint()
        for keypoints in response_json['keypoints']:
            if 'track_id' not in keypoints:
                continue
            track_id = keypoints['track_id']
            if track_id not in keypoints_by_id:
                keypoints_by_id[track_id] = []
            keypoints_by_id[track_id].append({
                'frame': i,
                'keypoints': keypoints,
            })
        tracker_id = response_json['tracker_id']
    return keypoints_by_id

    # skeletons_by_id = {}
    # for id, keypoints in keypoints_by_id.items():
    #     json_filename = f'track_id_{id:03d}.json'
    #     with open(os.path.join(json_path, json_filename), 'w') as f:
    #         skeletons = {
    #             'video': {
    #                 'path': os.path.basename(video_path),
    #                 'width': clip.w,
    #                 'height': clip.h,
    #                 'original_length': clip.duration,
    #                 'fps': fps,
    #                 'track_id': id
    #             },
    #             'annotations': keypoints
    #         }
    #         json.dump(skeletons, f, indent=4)
    #         skeletons_by_id[id] = skeletons

    # return skeletons_by_id