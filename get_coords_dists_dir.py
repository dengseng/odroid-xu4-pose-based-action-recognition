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


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


actions_5 = ['clap', 'jump', 'sit', 'walk', 'wave']


# get all dists_list and coords_list, overwrites existing
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
    
    x_train = []
    y_train = []
    
    t_total = time.time()
    num_frame_total = 0
    
    actions = os.listdir(args.dataset)
    
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
            coords_dir = os.path.join(video_dir, 'coords_list_%s.dil' % args.model);
            dists_dir = os.path.join(video_dir, 'dists_list_%s.dil' % args.model);
            coords_list = []
            dists_list = []
            num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
            
            # save actor neck position to prevent losing actor tracking
            neck_position = [0,0]
            
            # iterate frames in video
            for frame_num in range(num_frames):
                frame = '%05d.png' % (frame_num+1)
                frame_dir = os.path.join(video_dir, frame)
                        
                # estimate human poses from a single image
                image = common.read_imgfile(frame_dir, None, None)
                if image is None:
                    print('Image can not be read, path=%s' % frame_dir)
                    sys.exit(-1)
                t = time.time()
                humans = e.inference(image, resize_to_default=False, upsample_size=args.resize_out_ratio)
                elapsed = time.time() - t
                
                print('Inference image %s: %.4f seconds.'  % (frame, elapsed))
                        
                coords, image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                cv2.imshow('tf-pose', image)
                cv2.waitKey(5)
                
                actor_idx = -1
                
                if neck_position == [0,0]:
                    if len(humans) > 1:
                        w_min = w/2
                        w_max = w/2
                        h_min = h/2
                        h_max = h/2
                        
                        # recursively expand search area for main actor
                        while actor_idx == -1:
                            w_min -= 10
                            w_max += 10
                            h_min -= 10
                            h_max += 10
                            for human_idx in range(len(humans)):
                                coord = coords[human_idx]
                                if w_min<coord[2] and coord[2]<w_max and h_min<coord[3] and coord[3]<h_max:
                                    print('Actor %d selected' % human_idx)
                                    neck_position = [coord[2], coord[3]]
                                    actor_idx = human_idx
                                elif w_min <= 10 or h_min <= 10:
                                    actor_idx = -2
                    else:
                        actor_idx = 0
                                
                else:
                    w_min = neck_position[0] - 10
                    w_max = neck_position[0] + 10
                    h_min = neck_position[1] - 10
                    h_max = neck_position[1] + 10
                    for human_idx in range(len(humans)):
                        coord = coords[human_idx]
                        if w_min<coord[2] and coord[2]<w_max and h_min<coord[3] and coord[3]<h_max:
                            print('Actor %d selected' % human_idx)
                            neck_position = [coord[2], coord[3]]
                            actor_idx = human_idx
                            break

                if actor_idx >= 0:
                    coord = coords[actor_idx]
                else:
                    coord = [0.0 for i in range(36)]
                
                coords_list.extend(coord)
                print(coord)
                
                num_frame_total += 1
                
                joint1 = [coord[1*2], coord[1*2+1]]
                
                dist = [joint1[0]-coord[0], 
                        joint1[1]-coord[1], 
                        joint1[0]-coord[4*2], 
                        joint1[1]-coord[4*2+1], 
                        joint1[0]-coord[7*2], 
                        joint1[1]-coord[7*2+1], 
                        joint1[0]-coord[10*2], 
                        joint1[1]-coord[10*2+1], 
                        joint1[0]-coord[13*2], 
                        joint1[1]-coord[13*2+1]]
                
                dists_list.extend(dist)
                print(dist)
                
            
            with open(coords_dir, 'wb') as f:
                dill.dump(coords_list, f, protocol=dill.HIGHEST_PROTOCOL)
            with open(dists_dir, 'wb') as f:
                dill.dump(dists_list, f, protocol=dill.HIGHEST_PROTOCOL)
    
    elapsed_total = time.time() - t_total
    print('Total time elapsed %.4f seconds.' % elapsed_total)
    if num_frame_total > 0:
        print('Average time elapsed per frame %.4f seconds.' % (elapsed_total/num_frame_total))