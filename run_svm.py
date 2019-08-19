import argparse
import time
import glob
import os
import dill

import cv2
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score

from tf_pose import common
from tf_pose.estimator_coords import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


fps_time = 0
actions = ['clap', 'jump', 'sit', 'walk', 'wave']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run svm on top of tf-pose-estimation for action classification')
    parser.add_argument('--input', type=str, default='camera', help='default=camera, frames / video / camera')
    parser.add_argument('--frames', type=str, default='', help='folder containing 40 frames, 00001.* to 00040.*')
    parser.add_argument('--image-format', type=str, default='png')
    parser.add_argument('--video', type=str, default='', help='video file')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--svm', type=str, default='svm_40frames_mobilenet_v2_small', help='svm model to use for classification')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                            'default=320x240, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()
    
    # make sure svm model name is "svm_##frames" or similar
    svm_num = int(args.svm[4:6])
    
    with open(args.svm, 'rb') as f:
        clf = dill.load(f)
    
    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(320, 240))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    action = ''
    
    
    # frames input selected, args.frames is path to directory of 40 frames with files named 00001.* to 00040.*
    # image format specified by args.image_format
    if args.input == 'frames':
        num_frames = len(glob.glob(os.path.join(args.frames, '*.'+args.image_format)))
        if num_frames != svm_num:
            print('Number of frames not equal to %d, path=%s' % (svm_num, args.frames))
            sys.exit(-1)
        
        coords_list = []
        
        for i in range(num_frames):
            frame = ('%05d.'+args.image_format) % (i+1)
            frame_dir = os.path.join(args.frames, frame)
            
            # estimate human poses from a single image
            image = common.read_imgfile(frame_dir, None, None)
            if image is None:
                print('Image can not be read, path=%s' % frame_dir)
                sys.exit(-1)

            t = time.time()
            humans = e.inference(image, resize_to_default=False, upsample_size=args.resize_out_ratio)
            elapsed = time.time() - t

            print('Inference image: %s in %.4f seconds.' % (frame, elapsed))

            if not args.showBG:
                image = np.zeros(image.shape)
            coords, image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            coords_list.extend(coords)
            
            cv2.putText(image, "FPS: %f" % (1.0/(time.time()-fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        
        t = time.time()
        action = clf.predict([coords_list])
        elapsed = time.time() - t
        print('Prediction time %.4f seconds.' % elapsed)
        print("Prediction: %s" % action)
        cv2.putText(image, "Prediction: %s" % action, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.destroyAllWindows()
    
    
    # video input selected, args.video is video file to be sampled at <svm_num> frames per second
    elif args.input == 'video':
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        if not cap.isOpened():
            print("Error opening video stream or file")
        
        coords_list = []
        while cap.isOpened():
            ret_val, image = cap.read()

            humans = e.inference(image, resize_to_default=False, upsample_size=args.resize_out_ratio)
            if not args.showBG:
                image = np.zeros(image.shape)
            coords, image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            coords_list.extend(coords)
            
            if len(coords_list) == svm_num*36:
                t = time.time()
                action = clf.predict([coords_list])
                elapsed = time.time() - t
                print('Prediction time %.4f seconds.' % elapsed)
                print("Prediction: %s" % action)
                coords_list = []
            
            cv2.putText(image, "Prediction: %s" % action, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(image, "FPS: %f" % (1.0/(time.time()-fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
    
    
    # video input selected, args.video is video file to be sampled at <svm_num> frames per second
    elif args.input == 'camera':
        print('Camera input')
        cam = cv2.VideoCapture(args.camera)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        coords_list = []
        
        while True:
            ret_val, image = cam.read()

            humans = e.inference(image, resize_to_default=False, upsample_size=args.resize_out_ratio)
            if not args.showBG:
                image = np.zeros(image.shape)
            coords, image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            coords_list.extend(coords)
            
            if len(coords_list) == svm_num*36:
                t = time.time()
                action = clf.predict([coords_list])
                elapsed = time.time() - t
                print('Prediction time %.4f seconds.' % elapsed)
                print("Prediction: %s" % action)
                coords_list = []
            
            cv2.putText(image, "Prediction: %s" % action, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(image, "FPS: %f" % (1.0/(time.time()-fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()