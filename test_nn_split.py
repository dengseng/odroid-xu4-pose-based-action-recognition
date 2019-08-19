import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate nn for JHMDB split1,2,3')
    parser.add_argument('--dataset', type=str, default='JHMDB', help='JHMDB')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--dists-list', type=str, default='', help='overrides model chosen if value is provided, default=dists_list_type-action-split_cmu.dil')
    args = parser.parse_args()
    
    if args.dists_list == '':
        dists_list_dir = 'dists_list_type-action-split_%s.dil' % args.model
        clf_save_dir = 'nn_dists_split%d_' + args.model
    # gets dists_list_dir and clf_save_dir from given dists-list file
    else:
        dists_list_dir = args.dists_list
        clf_save_dir = 'nn_dists_split%d_' + args.dists_list[30:-4]
    
    with open(dists_list_dir, 'rb') as f:
        dataset = dill.load(f)
        f.close()
    
    data = dataset['test']
    
    actions = os.listdir(args.dataset)
    
    x_test = [[], [], []]
    y_true = [[], [], []]
    y_pred = [[], [], []]
    clf = dict()
    
    # for each action
    for action in data:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        
        # for each split number
        for split in range(3):
            print('Split: %d' % (split+1))
            
            # for each video
            for video in data[action][split]:
                print('Video: %s' % video)
                
                x_test[split].append(data[action][split][video])
                y_true[split].append(action)
    
    
    for split in range(3):
        print('Split: %d' % (split+1))
        
        with open(clf_save_dir % (split+1), 'rb') as f:
            clf[split] = dill.load(f)
            f.close()
        
        t = time.time()
        y_pred[split] = clf[split].predict(x_test[split])
        elapsed = time.time() - t
        
        xrow = np.shape(x_test[split])[0]
        
        print('Accurate prediction: %d/%d' % (accuracy_score(y_true[split], y_pred[split], normalize=False), len(y_true[split])))
        print('Accurate prediction: %.4f%%' % (accuracy_score(y_true[split], y_pred[split])*100.0))
        print('Number of samples: %d' % xrow)
        print('Prediction time %.4f seconds.' % elapsed)
        
        print('True class:', y_true[split])
        print('Predicted class:', y_pred[split])
        print(confusion_matrix(y_true[split], y_pred[split]))  
        print(classification_report(y_true[split], y_pred[split]))
        
        y_prob = clf[split].predict_proba(x_test[split])
        print('Confidence scores for first video')
        for i in range(21):
            print(actions[i])
            print(y_prob[0,i])
    