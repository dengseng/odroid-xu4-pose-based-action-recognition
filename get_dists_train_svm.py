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

from sklearn import svm


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


actions_5 = ['clap', 'jump', 'sit', 'walk', 'wave']


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
            dists_dir = os.path.join(video_dir, 'dists_list_%s.dil' % args.model);
            coords_dir = os.path.join(video_dir, 'coords_list_%s.dil' % args.model);
            
            num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
            
            if num_frames != 40:
                print('Skipped video <40 frames: %s' % video)
                continue
            
            if os.path.isfile(dists_dir):
                print('dists_list_%s.dil already computed' % args.model)
                with open(dists_dir, 'rb') as f:
                    dists_list = dill.load(f)
                    f.close()
            
            else:
                if os.path.isfile(coords_dir):
                    print('coords_list_%s.dil already computed' % args.model)
                    with open(coords_dir, 'rb') as f:
                        coords_list = dill.load(f)
                        f.close()
                    
                else:
                    print('computing coords_list_%s.dil and dists_list_%s.dil' % (args.model, args.model))
                    coords_list = []
                    
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
                        
                        coords_list.extend(coords)
                        
                        num_frame_total += 1
                
                print(np.shape(coords_list))
                        
                dists_list = []
                for i in range(num_frames):
                    frame_idx = i*36
                    joint1 = [coords_list[1*2+frame_idx], coords_list[1*2+1+frame_idx]]
                    dists_list.extend([joint1[0]-coords_list[0+frame_idx], 
                                        joint1[1]-coords_list[1+frame_idx], 
                                        joint1[0]-coords_list[4*2+frame_idx], 
                                        joint1[1]-coords_list[4*2+1+frame_idx], 
                                        joint1[0]-coords_list[7*2+frame_idx], 
                                        joint1[1]-coords_list[7*2+1+frame_idx], 
                                        joint1[0]-coords_list[10*2+frame_idx], 
                                        joint1[1]-coords_list[10*2+1+frame_idx], 
                                        joint1[0]-coords_list[13*2+frame_idx], 
                                        joint1[1]-coords_list[13*2+1+frame_idx]])
            
                with open(coords_dir, 'wb') as f:
                    dill.dump(coords_list, f, protocol=dill.HIGHEST_PROTOCOL)
                    f.close()
                    
            
            print(np.shape(dists_list))
            
            x_train.append(dists_list)
            y_train.append(action)
            
            with open(dists_dir, 'wb') as f:
                dill.dump(dists_list, f, protocol=dill.HIGHEST_PROTOCOL)
                f.close()
    
    clf = svm.SVC(gamma='scale')
    t = time.time()
    clf.fit(x_train, y_train)
    elapsed = time.time() - t
    elapsed_total = time.time() - t_total
    
    print('SVM model fitted %.4f seconds.' % elapsed)
    print('Number of samples: %d' % np.shape(x_train)[0])
    print('Total time elapsed %.4f seconds.' % elapsed_total)
    if num_frame_total > 0:
        print('Average time elapsed per frame %.4f seconds.' % (elapsed_total/num_frame_total))
    
    with open('svm_dist_40frames_%s' % args.model, 'wb') as f:
        dill.dump(clf, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()