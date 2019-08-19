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
    parser.add_argument('--datatype', type=str, default='dists', help='dists / coords')
    args = parser.parse_args()
    
    # videos linked to each action for each split number
    dataset = dict()
    
    actions = os.listdir(args.dataset)
    
    for data in datasets:
        dataset[data] = dict()
        
    # using splits text file to acquire training and testing dataset split
    # for each action
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        dataset['train'][action] = [dict(),dict(),dict()]
        dataset['test'][action] = [dict(),dict(),dict()]
        
        # for each split number
        for split in range(3):
            print('Split: %d' % (split+1))
            
            # read lines and intepret video for training or testing
            with open(os.path.join('splits', '%s_test_split%d.txt' % (action, (split+1))), 'r') as f:
                videos = f.readlines()
                f.close()
            
            # for each video
            for video in videos:
                print('Video: %s' % video)
                video_dir = os.path.join(action_dir, video[:-7])
                num_frames =  len(glob.glob(os.path.join(video_dir, '*.png')))
                
                if num_frames != 40:
                    print('Skipped video <40 frames.')
                    continue
                
                with open(os.path.join(video_dir, '%s_list_%s.dil' % (args.datatype, args.model)), 'rb') as f:
                    data_list = dill.load(f)
                    f.close()
                
                print(np.shape(data_list))
                
                # add to training dataset
                if video[-2] == '1':
                    dataset['train'][action][split][video[:-7]] = data_list
                    
                # add to testing dataset
                elif video[-2] == '2':
                    dataset['test'][action][split][video[:-7]] = data_list
                
    with open('%s_list_type-action-split_%s.dil' % (args.datatype, args.model), 'wb') as f:
        dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL)