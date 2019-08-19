import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    
actions = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test svm fitted with JHMDB video with 40 frames')
    parser.add_argument('--model', type=str, default='mobilenet_v2_small', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    args = parser.parse_args()
    
    x_test = []
    y_true = []
    
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
            
            with open(os.path.join(video_dir, 'coords_list_%s.dil' % args.model), 'rb') as f:
                coords_list = dill.load(f)
                f.close()
            
            print(np.shape(coords_list))
            x_test.append(coords_list)
            y_true.append(action)
    
        
    with open('svm_40frames_%s' % args.model, 'rb') as f:
        clf = dill.load(f)
        f.close()
    
    t = time.time()
    y_pred = clf.predict(x_test)
    elapsed = time.time() - t
        
    print('Prediction time %.4f seconds.' % elapsed)
    print('True class:', y_true)
    print('Predicted class:', y_pred)
    print(confusion_matrix(y_true, y_pred))
    print('Accurate prediction: %d/%d' % (accuracy_score(y_true, y_pred, normalize=False), len(y_true)))
    print('Accurate prediction: %.4f%%' % (accuracy_score(y_true, y_pred, normalize=True)*100.0))
    print(classification_report(y_true, y_pred))