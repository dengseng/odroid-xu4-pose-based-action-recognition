import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from sklearn import svm


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train svm fit with JHMDB video with 40 frames')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    args = parser.parse_args()
    
    x_train = []
    y_train = []
    
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        videos = os.listdir(action_dir)
        
        for video in videos:
            if '.' in video:
                print('Skipped irrelevant dir=%s' % video)
                continue
            
            video_dir = os.path.join(action_dir, video)
            num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
            
            if num_frames != 40:
                print('Skipped video <40 frames: %s' % video)
                continue
            
            print('Video: %s' % video)
            
            with open(os.path.join(video_dir, 'dists_'+args.model+'.dil'), 'rb') as f:
                dists_list = dill.load(f)
                f.close()
            
            x_train.append(dists_list)
            y_train.append(action)
        
    xrow, xcol = np.shape(x_train)
        
    clf = svm.SVC(gamma='scale')
    t = time.time()
    clf.fit(x_train, y_train)
    elapsed = time.time() - t
        
    print('SVM model fitted %.4f seconds.' % elapsed)
    print('Number of samples: %d' % xrow)
    print('Number of distance vectors per sample: %d' % (xcol/2))
    
    with open('svm_dists_40frames_%s' % args.model, 'wb') as f:
        dill.dump(clf, f, protocol=dill.HIGHEST_PROTOCOL)
    