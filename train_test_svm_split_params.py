import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and test svm for JHMDB split1,2,3')
    parser.add_argument('--dataset', type=str, default='JHMDB', help='JHMDB')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--datatype', type=str, default='coords_40', help='coords_40 / dists_40 / coords / dists')
    parser.add_argument('--C', type=float, default=28.0340840873003, help='SVM C parameter')
    parser.add_argument('--gamma', type=float, default=2.84880332450157e-7, help='SVM gamma parameter')
    parser.add_argument('--kernel', type=str, default='rbf', help='SVM kernel parameter')
    args = parser.parse_args()
    
    dataset_dir = 'dataset_%s_%s.dil' % (args.datatype, args.model)
    clf_save_dir = os.path.join('classifiers', 'svm_' + args.datatype + '_split%d_' + args.model + '.dil')
    
    with open(dataset_dir, 'rb') as f:
        dataset = dill.load(f)
        f.close()
    
    actions = os.listdir(args.dataset)
    
    print('----------------------------------------------------------')
    print('Dataset: %s %s' % (args.datatype, args.model))
    
    x_train = [[], [], []]
    y_train = [[], [], []]
    
    x_test = [[], [], []]
    y_true = [[], [], []]
    y_pred = [[], [], []]
    
    for split in range(3):
        print('\nSplit: %d' % (split+1))
        
        for action in actions:
            print('Action: %s' % action)
            action_dir = os.path.join('JHMDB', action)
            
            for video in dataset['train'][action][split]:
                x_train[split].append(dataset['train'][action][split][video])
                y_train[split].append(action)
                
            for video in dataset['test'][action][split]:
                x_test[split].append(dataset['test'][action][split][video])
                y_true[split].append(action)
                    
        clf = svm.SVC(kernel=args.kernel, gamma=args.gamma, C=args.C)
        t = time.time()
        clf.fit(x_train[split], y_train[split])
        train_elapsed = time.time() - t
        train_num = np.shape(x_train[split])[0]
        
        t = time.time()
        y_pred[split] = clf.predict(x_test[split])
        test_elapsed = time.time() - t
        test_num = np.shape(x_test[split])[0]
                    
        with open(clf_save_dir % (split+1), 'wb') as f:
            dill.dump(clf, f, protocol=dill.HIGHEST_PROTOCOL)
            f.close()
                        
            print('Accurate prediction: %d/%d' % (accuracy_score(y_true[split], y_pred[split], normalize=False), len(y_true[split])))
            print('Accurate prediction: %.2f%%' % (accuracy_score(y_true[split], y_pred[split])*100.0))
            print('Number of training samples: %d' % train_num)
            print('SVM model fitted %.4f seconds.' % train_elapsed)
            print('Number of testing samples: %d' % test_num)
            print('Prediction time %.4f seconds.' % test_elapsed)
            continue
            print('True class:', y_true[split])
            print('Predicted class:', y_pred[split])
            print(confusion_matrix(y_true[split], y_pred[split]))  
            print(classification_report(y_true[split], y_pred[split]))
                    