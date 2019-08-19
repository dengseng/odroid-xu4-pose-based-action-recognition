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
    parser = argparse.ArgumentParser(description='train and test nn for JHMDB split1,2,3')
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
    
    
    actions = os.listdir(args.dataset)
    
    x_train = [[], [], []]
    y_train = [[], [], []]
    
    x_test = [[], [], []]
    y_true = [[], [], []]
    y_pred = [[], [], []]
    
    clf = dict()
    
    # get training set
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        
        # for each split number
        for split in range(3):
            print('Split: %d' % (split+1))
            
            # for each video
            for video in dataset['train'][action][split]:
                video_dir = os.path.join(action_dir, video)
                num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
                
                if num_frames != 40:
                    print('Skipped video <40 frames: %s' % video)
                    continue
                
                print('Video: %s' % video)
                
                x_train[split].append(dataset['train'][action][split][video])
                y_train[split].append(action)
                
    # get testing set
    for action in actions:
        print('Action: %s' % action)
        action_dir = os.path.join('JHMDB', action)
        
        # for each split number
        for split in range(3):
            print('Split: %d' % (split+1))
            
            # for each video
            for video in dataset['test'][action][split]:
                video_dir = os.path.join(action_dir, video)
                num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
                
                if num_frames != 40:
                    print('Skipped video <40 frames: %s' % video)
                    continue
                
                print('Video: %s' % video)
                
                x_test[split].append(dataset['test'][action][split][video])
                y_true[split].append(action)
    
    for split in range(3):
        print('Split: %d' % (split+1))
        xrow = np.shape(x_train[split])[0]
        
        clf[split] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(xrow*xcol, 2), random_state=1)
        t = time.time()
        clf[split].fit(x_train[split], y_train[split])
        elapsed = time.time() - t
        
        print('Number of samples: %d' % xrow)
        print('NN model fitted %.4f seconds.' % elapsed)
        
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
        
        
        with open(clf_save_dir % (split+1), 'wb') as f:
            dill.dump(clf[split], f, protocol=dill.HIGHEST_PROTOCOL)
            f.close()
    