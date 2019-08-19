import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np
import pandas as pd

from sklearn import neural_network
from sklearn.model_selection import GridSearchCV


actions_21 = ['stand', 'climb_stairs', 'golf', 'pick', 'throw', 'push', 'clap', 'shoot_gun', 'catch', 'brush_hair', 'wave', 'kick_ball', 'run', 'swing_baseball', 'pullup', 'pour', 'walk', 'shoot_bow', 'jump', 'shoot_ball', 'sit']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='grid search SVM')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    args = parser.parse_args()
    
    
    actions = os.listdir('JHMDB')
    
    
    x_train_coords_40 = []
    x_train_dists_40 = []
    y_train_40 = []
    x_train_coords = []
    x_train_dists = []
    y_train = []
    
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
            num_frames = len(glob.glob(os.path.join(video_dir, '*.png')))
            
            with open(os.path.join(video_dir, 'coords_list_%s.dil' % args.model), 'rb') as f:
                coords_list = dill.load(f)
                f.close()
            with open(os.path.join(video_dir, 'dists_list_%s.dil' % args.model), 'rb') as f:
                dists_list = dill.load(f)
                f.close()
            
            if num_frames != 40:
                empty_frames = range(num_frames, 40)
                for empty_frame in empty_frames:
                    coords_list.extend([0.0 for i in range(36)])
                    dists_list.extend([0.0 for i in range(10)])
                x_train_coords.append(coords_list)
                x_train_dists.append(dists_list)
                y_train.append(action)
                
            else:
                
                x_train_coords_40.append(coords_list)
                x_train_dists_40.append(dists_list)
                y_train_40.append(action)
    
                x_train_coords.append(coords_list)
                x_train_dists.append(dists_list)
                y_train.append(action)
    
    # Create my estimator and prepare the parameter grid dictionary
    params_dict = {"alpha": np.linspace(0,20,21)}
    clf = neural_network.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(800,2), random_state=1)
    
    start = time.time()
    # Fit the grid search
    search_coords_40 = GridSearchCV(estimator=clf, param_grid=params_dict, cv=3)
    search_coords_40.fit(x_train_coords_40, y_train_40)
    elapsed = time.time() - start
    df = pd.DataFrame(search_coords_40.cv_results_)
    print("----------------------------------------------------------")
    print("coords_40")
    print("Time elapsed", elapsed)
    print("Best parameter values:", search_coords_40.best_params_)
    print("CV Score with best parameter values:", search_coords_40.best_score_)
    
    
    start = time.time()
    # Fit the grid search
    search_dists_40 = GridSearchCV(estimator=clf, param_grid=params_dict, cv=3)
    search_dists_40.fit(x_train_dists_40, y_train_40)
    elapsed = time.time() - start
    df = pd.DataFrame(search_dists_40.cv_results_)
    print("----------------------------------------------------------")
    print("dists_40")
    print("Time elapsed", elapsed)
    print("Best parameter values:", search_dists_40.best_params_)
    print("CV Score with best parameter values:", search_dists_40.best_score_)
    
    
    start = time.time()
    # Fit the grid search
    search_coords = GridSearchCV(estimator=clf, param_grid=params_dict, cv=3)
    search_coords.fit(x_train_coords, y_train)
    elapsed = time.time() - start
    df = pd.DataFrame(search_coords.cv_results_)
    print("----------------------------------------------------------")
    print("coords")
    print("Time elapsed", elapsed)
    print("Best parameter values:", search_coords.best_params_)
    print("CV Score with best parameter values:", search_coords.best_score_)
    
    
    start = time.time()
    # Fit the grid search
    search_dists = GridSearchCV(estimator=clf, param_grid=params_dict, cv=3)
    search_dists.fit(x_train_dists, y_train)
    elapsed = time.time() - start
    df = pd.DataFrame(search_dists.cv_results_)
    print("----------------------------------------------------------")
    print("dists")
    print("Time elapsed", elapsed)
    print("Best parameter values:", search_dists.best_params_)
    print("CV Score with best parameter values:", search_dists.best_score_)
    
    with open(os.path.join('classifiers', 'search_coords_40_%s.dil' % args.model), 'wb') as f:
        dill.dump(search_coords_40, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open(os.path.join('classifiers', 'search_coords_40_%s.dil' % args.model), 'wb') as f:
        dill.dump(search_dists_40, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open(os.path.join('classifiers', 'search_coords_%s.dil' % args.model), 'wb') as f:
        dill.dump(search_coords, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    with open(os.path.join('classifiers', 'search_dists_%s.dil' % args.model), 'wb') as f:
        dill.dump(search_dists, f, protocol=dill.HIGHEST_PROTOCOL)
        f.close()
    