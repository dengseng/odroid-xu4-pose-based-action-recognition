import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator_coords import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


actions = ['clap', 'jump', 'sit', 'walk', 'wave']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--dataset', type=str, default='JHMDB', help='JHMDB')
    parser.add_argument('--resolution', type=str, default='320x240', help='network input resolution. default=320x240')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    # iterate videos in action
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        videos = os.listdir(action_dir)
        
        for video in videos:
            if '.' in video:
                print('Skipped irrelevant dir=%s' % video)
                continue
            print('Video: %s' % video)
            video_dir = os.path.join(action_dir, video)
            all_humans = dict()
            all_coords = dict()
            coords_list = []
            num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
            
            # iterate frames in video
            for i in range(num_frames):
                frame = '%05d.png' % (i+1)
                frame_dir = os.path.join(video_dir, frame)
                
                # estimate human poses from a single image
                image = common.read_imgfile(frame_dir, None, None)
                if image is None:
                    print('Image can not be read, path=%s' % frame_dir)
                    sys.exit(-1)
                t = time.time()
                humans = e.inference(image, resize_to_default=False, upsample_size=args.resize_out_ratio)
                elapsed = time.time() - t
                
                print('Inference image %s: %.4f seconds.' % (frame, elapsed))
                
                coords, image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                all_humans[frame] = humans
                all_coords[frame] = coords
                coords_list.extend(coords)
                
            with open(os.path.join(video_dir, 'pose_'+args.model+'.dil'), 'wb') as f:
                dill.dump(all_humans, f, protocol=dill.HIGHEST_PROTOCOL)
                        
            with open(os.path.join(video_dir, 'coords_'+args.model+'.dil'), 'wb') as f:
                dill.dump(all_coords, f, protocol=dill.HIGHEST_PROTOCOL)
            
            with open(os.path.join(video_dir, 'coords_list_'+args.model+'.dil'), 'wb') as f:
                dill.dump(all_coords, f, protocol=dill.HIGHEST_PROTOCOL)