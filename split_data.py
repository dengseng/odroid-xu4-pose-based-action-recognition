import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


datasets = ['train', 'test']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split jhmdb dataset into training and testing')
    parser.add_argument('--dataset', type=str, default='JHMDB', help='JHMDB')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    args = parser.parse_args()
    
    # videos linked to each action for each split number
    dataset_coords_40 = dict()
    dataset_dists_40 = dict()
    dataset_coords = dict()
    dataset_dists = dict()
    
    actions = os.listdir(args.dataset)
    
    for data in datasets:
        dataset_coords_40[data] = dict()
        dataset_dists_40[data] = dict()
        dataset_coords[data] = dict()
        dataset_dists[data] = dict()
        
    # using splits text file to acquire training and testing dataset split
    # for each action
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        
        dataset_coords_40['train'][action] = [dict(),dict(),dict()]
        dataset_coords_40['test'][action] = [dict(),dict(),dict()]
        dataset_dists_40['train'][action] = [dict(),dict(),dict()]
        dataset_dists_40['test'][action] = [dict(),dict(),dict()]
        dataset_coords['train'][action] = [dict(),dict(),dict()]
        dataset_coords['test'][action] = [dict(),dict(),dict()]
        dataset_dists['train'][action] = [dict(),dict(),dict()]
        dataset_dists['test'][action] = [dict(),dict(),dict()]
        
        # for each split number
        for split in range(3):
            print('Split: %d' % (split+1))
            
            # read lines and intepret video for training or testing
            with open(os.path.join('splits', '%s_test_split%d.txt' % (action, (split+1))), 'r') as f:
                videos = f.readlines()
                f.close()
            
            # for each video
            for video in videos:
                print('Video: %s' % video[:-1])
                video_dir = os.path.join(action_dir, video[:-7])
                num_frames =  len(glob.glob(os.path.join(video_dir, '*.png')))
                
                with open(os.path.join(video_dir, 'dists_list_%s.dil' % args.model), 'rb') as f:
                    dists_list = dill.load(f)
                    f.close()
                    
                with open(os.path.join(video_dir, 'coords_list_%s.dil' % args.model), 'rb') as f:
                    coords_list = dill.load(f)
                    f.close()
                
                if num_frames != 40:
                    empty_frames = range(num_frames, 40)
                    for empty_frame in empty_frames:
                        coords_list.extend([0.0 for i in range(36)])
                        dists_list.extend([0.0 for i in range(10)])
                
                    # add to training dataset
                    if video[-2] == '1':
                        dataset_coords['train'][action][split][video[:-7]] = coords_list
                        dataset_dists['train'][action][split][video[:-7]] = dists_list
                        
                    # add to testing dataset
                    elif video[-2] == '2':
                        dataset_coords['test'][action][split][video[:-7]] = coords_list
                        dataset_dists['test'][action][split][video[:-7]] = dists_list
                else:
                    # add to training dataset
                    if video[-2] == '1':
                        dataset_coords_40['train'][action][split][video[:-7]] = coords_list
                        dataset_dists_40['train'][action][split][video[:-7]] = dists_list
                        dataset_coords['train'][action][split][video[:-7]] = coords_list
                        dataset_dists['train'][action][split][video[:-7]] = dists_list
                        
                    # add to testing dataset
                    elif video[-2] == '2':
                        dataset_coords_40['test'][action][split][video[:-7]] = coords_list
                        dataset_dists_40['test'][action][split][video[:-7]] = dists_list
                        dataset_coords['test'][action][split][video[:-7]] = coords_list
                        dataset_dists['test'][action][split][video[:-7]] = dists_list
                
    with open('dataset_coords_40_%s.dil' % args.model, 'wb') as f:
        dill.dump(dataset_coords_40, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open('dataset_dists_40_%s.dil' % args.model, 'wb') as f:
        dill.dump(dataset_dists_40, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open('dataset_coords_%s.dil' % args.model, 'wb') as f:
        dill.dump(dataset_coords, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open('dataset_dists_%s.dil' % args.model, 'wb') as f:
        dill.dump(dataset_dists, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()