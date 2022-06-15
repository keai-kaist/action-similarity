import numpy as np

def normalize_florence(x_labels, y_labels):
    x_max, y_max = x_labels[:,:,:,0].max(), x_labels[:,:,:,1].max()
    x_min, y_min = x_labels[:,:,:,0].min(), x_labels[:,:,:,1].min()
    x_labels[:,:,:,0] = x_labels[:,:,:,0] - x_min
    x_labels[:,:,:,1] = x_labels[:,:,:,1] - y_min
    height, width = x_max - x_min, y_max - y_min
    return x_labels, y_labels, height, width

def normalize_florence_to_ntu_distribution(x_labels, y_labels):
    # Florence: (23, 32, 15, 3) max(991.4373503654234, 924.6611328125, 3553.3125), min(-894.992427702873, -1635.0623779296875, 2160.5501275831652)
    # NTU: max(3.209906 1.728893 5.110941), min(-2.834219 -2.15311 0.0)
    x_max_f, x_min_f = 991.4, -894.4
    y_max_f, y_min_f = 924.6, -1635.0
    z_max_f, z_min_f = 3553.3, 2160.5
    
    x_max_n, x_min_n = 3.2, -2.8
    y_max_n, y_min_n = 1.7, -2.1
    z_max_n, z_min_n = 5.1, 0
    
    x_rate = (x_max_n - x_min_n)/(x_max_f - x_min_f)
    y_rate = (y_max_n - y_min_n)/(y_max_f - y_min_f)
    z_rate = (z_max_n - z_min_n)/(z_max_f - z_min_f)
    
    x_labels[:,:,:,0] = (x_labels[:,:,:,0] - x_min_f) * x_rate + x_min_n
    x_labels[:,:,:,1] = (x_labels[:,:,:,1] - y_min_f) * y_rate + y_min_n
    x_labels[:,:,:,2] = (x_labels[:,:,:,2] - z_min_f) * z_rate + z_min_n
    
    return x_labels, y_labels

def raw_point_to_coordinate(raw_point: np.ndarray, score=1):
    coordinate = {
        'x': raw_point[0],
        'y': raw_point[1],
        'score': score,
    }
    return coordinate

def raw_to_keypoints_by_id(raw_keypoints: np.ndarray):
    # raw_keypoints는 한개 비디오에 대한 keypoints
    # 길이 x keypoints x 3
    id = 0
    keypoints_score = 1
    keypoints_by_id = {id: []}

    for frame_id, frame in enumerate(raw_keypoints):
        keypoints = {}
        x1 = frame[:, 0].min()
        x2 = frame[:, 0].max()
        y1 = frame[:, 1].min()
        y2 = frame[:, 1].max()
        keypoints['box'] = {
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2,
        }
        keypoints['nose'] = raw_point_to_coordinate(frame[0,:2])
        keypoints['left_shoulder'] = raw_point_to_coordinate(frame[3,:2])
        keypoints['left_elbow'] = raw_point_to_coordinate(frame[4,:2])
        keypoints['left_wrist'] = raw_point_to_coordinate(frame[5,:2])
        keypoints['right_shoulder'] = raw_point_to_coordinate(frame[6,:2])
        keypoints['right_elbow'] = raw_point_to_coordinate(frame[7,:2])
        keypoints['right_wrist'] = raw_point_to_coordinate(frame[8,:2])
        keypoints['left_hip'] = raw_point_to_coordinate(frame[9,:2])
        keypoints['left_knee'] = raw_point_to_coordinate(frame[10,:2])
        keypoints['left_ankle'] = raw_point_to_coordinate(frame[11,:2])
        keypoints['right_hip'] = raw_point_to_coordinate(frame[12,:2])
        keypoints['right_knee'] = raw_point_to_coordinate(frame[13,:2])
        keypoints['right_ankle'] = raw_point_to_coordinate(frame[14,:2])
        keypoints['neck'] = raw_point_to_coordinate((frame[6,:2] + frame[3,:2]) / 2)
        keypoints['mid_hip'] = raw_point_to_coordinate((frame[12,:2] + frame[9,:2]) / 2)
        keypoints['score'] = keypoints_score
        keypoints_by_id[id].append({
                'frame': frame_id,
                'keypoints': keypoints
            }
        )
    return keypoints_by_id