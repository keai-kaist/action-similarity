import argparse

from bpe import Config
from utils import cache_file

from action_similarity.database import ActionDatabase
from action_similarity.motion import extract_keypoints, compute_motion_embedding
from action_similarity.predictor import Predictor

def main():
    video_path = './samples/CCTV.mp4'
    video_path = './samples/S001C001P001R001A007_rgb.mp4'
    #video_path = './custom_data/samples/hand_signal01.mp4'
    #video_path = './custom_data/samples/jump01.mp4'
    video_path = './custom_data/testset/001/S002C002P004R001A001.mp4'
    #video_path = './custom_data0419/samples/stop01.mp4'
    
    db = ActionDatabase(
        config=config,
        action_label_path='./custom_data/action_label.txt',
    )
    print("Compute standard db...")
    db.compute_standard_action_database(
        skeleton_path='./data/custom_skeleton',
        data_path=args.data_dir,
        model_path='./data/model_best.pth.tar',)
    for action_idx, features in db.db.items():
        print(db.actions[action_idx], len(features))
    
    print("Extract keypoints...")
    #keypoints_by_id = extract_keypoints(video_path, fps=30)
    keypoints_by_id = cache_file(video_path, extract_keypoints, 
         *(video_path,), **{'fps':30,})

    print("Encode motion embeddings...")
    seq_features = compute_motion_embedding(
        skeletons_json_path=keypoints_by_id,
        similarity_analyzer=db.similarity_analyzer,
        mean_pose_bpe=db.mean_pose_bpe,
        std_pose_bpe=db.std_pose_bpe,
        scale=db.scale,
        device=db.config.device,)
    
    print("Predict action...")
    predictor = Predictor(config=config, std_db=db)
    action_label = predictor.predict(seq_features)
    print(f"Predicted action is {db.actions[action_label]}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--clustering', type=str, default=None, help="clustering for standard database")
    parser.add_argument('-k', '--k_neighbors', type=int, default=1, help="number of neighbors to use for KNN")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    parser.add_argument('--update', action='store_true', help="Update database using custom skeleton")
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