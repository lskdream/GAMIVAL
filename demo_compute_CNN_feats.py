import ffmpeg
import  numpy as np
import imageio
np.random.seed(7)
import tensorflow as tf
tf.compat.v1.Session()
tf.random.set_random_seed(9)
from tensorflow.keras.models import  load_model
from tensorflow.keras.applications import densenet
import glob
import os
import pandas as pd
import argparse
import scipy.io
import math
import time

def test_video(NDGmodel, videopath, videoname, framerate):
    filepath = os.path.join(videopath, videoname)
    probe = ffmpeg.probe(filepath)
    video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    out, err = (ffmpeg
                .input(filepath)
                .output('pipe:', format='rawvideo', pix_fmt = 'rgb24')
                .run(capture_stdout = True)
                )
    video = np.frombuffer(out, np.uint8).reshape([-1, height,width,3])
    
    count = 0
    preds_patch = []
    
    for k in range(math.floor(np.size(video,0)/framerate)):
        #frame 1
        ims1 = video[k*math.ceil(framerate),:,:,:]    
        patches1 = np.zeros((ims1.shape[0]//299, ims1.shape[1]//299, 299, 299, 3))
                
        for i in range(patches1.shape[0]):
            for j in range(patches1.shape[1]):
                patches1[i,j] = ims1[i*299:(i+1)*299, j*299:(j+1)*299]
                        
        patches1 = densenet.preprocess_input(patches1.reshape((-1, 299, 299, 3)))
        pred_patch1 = NDGmodel.predict(patches1)
        
        #frame 2
        ims2 = video[k*math.ceil(framerate)+math.floor(framerate/2),:,:,:]    
        patches2 = np.zeros((ims2.shape[0]//299, ims2.shape[1]//299, 299, 299, 3))
                
        for i in range(patches2.shape[0]):
            for j in range(patches2.shape[1]):
                patches2[i,j] = ims2[i*299:(i+1)*299, j*299:(j+1)*299]
                        
        patches2 = densenet.preprocess_input(patches2.reshape((-1, 299, 299, 3)))
        pred_patch2 = NDGmodel.predict(patches2)
        preds_patch.append((sum(pred_patch1)/len(pred_patch1)+sum(pred_patch2)/len(pred_patch2))/2)

        count = count + 1;
    return sum(preds_patch)/count, preds_patch

if __name__== "__main__":

    parser = argparse.ArgumentParser()
                    
    parser.add_argument('--dataset_name', type=str, default='LIVE-Meta-Gaming',
                      help='Evaluation dataset.')
    
    parser.add_argument('-mp', '--model', action='store', dest='model', default=r'./models/subjectiveDemo2_DMOS_Final.model' ,
                    help='Specify model together with the path, e.g. ./models/subjectiveDemo2_DMOS_Final.model')
    parser.add_argument('--video_path', type=str, default='.',
                    help='Directory containing the input videos.')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#cpu only

    csv_file = os.path.join('mos_files', args.dataset_name+'_metadata.csv')
    df = pd.read_csv(csv_file)
    videoname = df['File'].to_numpy()
    framerate = df['framerate'].to_numpy()
    feature_patch_total = []
    feats_frame_patch_total = np.empty((len(videoname), 1), dtype=object)
    
    NDGmodel = load_model(args.model)
    x = NDGmodel.layers[-2].output
    NDGmodel = tf.keras.Model(inputs = NDGmodel.input, outputs = x)
    for i in range(len(videoname)):
        t_overall_start = time.time()
        feature_patch, feats_frame_patch = test_video(NDGmodel, args.video_path, videoname[i], framerate[i])
        feature_patch_total.append(feature_patch)
        feats_frame_patch_total[i,0] = feats_frame_patch
        print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
        scipy.io.savemat('feat_files/LIVE-Meta-Mobile-Cloud-Gaming_CNN_bicubic_feats.mat', mdict={'feats_mat': np.asarray(feature_patch_total,dtype=np.float)})
    
    #scipy.io.savemat('feats_files/LIVE-Meta-Mobile-Cloud-Gaming_CNN_bicubic_feats_frame.mat', mdict={'feats_mat_frames': np.asarray(feats_frame_patch_total,dtype=np.object)})
