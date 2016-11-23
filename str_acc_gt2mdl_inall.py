import os
import re
import csv
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.io as sio
from collections import Counter
from sklearn import preprocessing
from scipy import stats
from scipy import *
from scipy.stats import *
from scipy.signal import *
from sklearn.metrics import matthews_corrcoef
import numpy.polynomial.polynomial as poly
import plotly 
from stru_utils import *
import _pickle as cPickle


save_flg = 1
# Rawan
subjs = ['Dzung' ]#'Rawan','Eric', ,['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',


for subj in subjs:
    subj = 'train'+subj


    subjfolder = subj + '(8Hz)/'
    folder = '../inlabStr/subject/'
    featFoler = folder+subjfolder+"feature/all_features/"

    datafile =  folder+subjfolder+"testdata.csv"
    segfolder = folder+subjfolder+"segmentation/"
    # act_rootfolder = segfolder+'activity/'

    # for act_ind in range(17):
    if 1:

# 
#           generate feeding data
# 
# 
        saveFileFolder = segfolder + 'gt_f_data/'
        if not os.path.exists(saveFileFolder):
            os.makedirs(saveFileFolder)

        # read in the raw data file
        # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
        #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
        r_df = pd.read_csv(datafile)

        gtActFilePath = segfolder+'/rawdata_gt/gt_feeding_headtail.csv'

        # read ground truth csv file
        # created by matlab
        if os.path.exists(gtActFilePath):
            gt_headtail = pd.read_csv(gtActFilePath, names = ['EnergyStart','EnergyEnd','EnergyDur'])
            
            gt_headtail['Start'] = gt_headtail['EnergyStart']
            gt_headtail['End'] = gt_headtail['EnergyEnd']

            for i in range(len(gt_headtail)):

                dataStart = int(gt_headtail['Start'].iloc[i])
                dataEnd = int(gt_headtail['End'].iloc[i])
                r_df_gesture = r_df.iloc[dataStart:dataEnd]

                saveFilePath = saveFileFolder + 'gt_f_gesture_' + str(i) + '.csv'
                r_df_gesture.to_csv(saveFilePath)




# 
#           generate non-feeding data
# 
# 
        saveFileFolder = segfolder + 'gt_nf_data/'
        
        if not os.path.exists(saveFileFolder):
            os.makedirs(saveFileFolder)

        # read in the raw data file
        # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
        #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
        r_df = pd.read_csv(datafile)

        gtActFilePath = segfolder+'/rawdata_gt/gt_nonfeeding_headtail.csv'

        # read ground truth csv file
        # created by matlab
        if os.path.exists(gtActFilePath):
            gt_headtail = pd.read_csv(gtActFilePath, names = ['EnergyStart','EnergyEnd','EnergyDur'])
            
            gt_headtail['Start'] = gt_headtail['EnergyStart']
            gt_headtail['End'] = gt_headtail['EnergyEnd']

            for i in range(len(gt_headtail)):

                dataStart = int(gt_headtail['Start'].iloc[i])
                dataEnd = int(gt_headtail['End'].iloc[i])
                r_df_gesture = r_df.iloc[dataStart:dataEnd]

                saveFilePath = saveFileFolder + 'gt_nf_gesture_' + str(i) + '.csv'
                r_df_gesture.to_csv(saveFilePath)




# 
#           generate feeding features
# 
# 
        gtFolder = segfolder + 'gt_f_data/'
        f_allfeatDF = pd.DataFrame()

        for _, dirnames, filenames in os.walk(gtFolder):
            n_gest = len(filenames)

            if n_gest == 0:
                continue
            else:
                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    gtActFile = 'gt_f_gesture_' + str(i) + '.csv'
                    gtActFilePath = gtFolder + gtActFile
                    r_df = pd.read_csv(gtActFilePath)

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x', 'Linear_Accel_y', 'Linear_Accel_z']]
                    r_df = df_iter_flt(r_df)

                    r_df = add_pitch_roll(r_df)

                    # generate the features
                    feat = gen_feat(r_df)

                    print(feat)


                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    # featDF["activity"] = act_ind


                    f_allfeatDF = pd.concat([f_allfeatDF,featDF])

        if not os.path.exists(featFoler):
            os.makedirs(featFoler)

        outfile = featFoler + "gt_f_features.csv"
        f_allfeatDF.to_csv(outfile, index=None)

# 
#           generate nonfeeding features
# 
# 
        gtFolder = segfolder + 'gt_nf_data/'
        nf_allfeatDF = pd.DataFrame()

        for _, dirnames, filenames in os.walk(gtFolder):
            n_gest = len(filenames)

            if n_gest == 0:
                continue
            else:
                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    gtActFile = 'gt_nf_gesture_' + str(i) + '.csv'
                    gtActFilePath = gtFolder + gtActFile
                    r_df = pd.read_csv(gtActFilePath)

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x', 'Linear_Accel_y', 'Linear_Accel_z']]
                    r_df = df_iter_flt(r_df)

                    r_df = add_pitch_roll(r_df)

                    # generate the features
                    feat = gen_feat(r_df)

                    print(feat)


                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    # featDF["activity"] = act_ind


                    nf_allfeatDF = pd.concat([nf_allfeatDF,featDF])



        if not os.path.exists(featFoler):
            os.makedirs(featFoler)

        outfile = featFoler + "gt_nf_features.csv"
        nf_allfeatDF.to_csv(outfile, index=None)




        f_allfeatDF['f-nf'] = 1
        nf_allfeatDF['f-nf'] = 0
        allfeatDF = pd.DataFrame()
        allfeatDF = pd.concat([f_allfeatDF,nf_allfeatDF])
            
        print(len(allfeatDF))
        allfeatDF = allfeatDF.dropna() 
        print(len(allfeatDF))
            
            
        allfeatDF.to_csv(featFoler+"gt_fnf_feats.csv",index=None)



        # 
        #   build model 
        # 

        df = pd.read_csv(featFoler+'gt_fnf_feats.csv' )
        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 
        X = df.iloc[:,:-2].as_matrix()
        Y = df['f-nf'].as_matrix()

        classifier = RandomForestClassifier(n_estimators=185)
        classifier.fit(X, Y)


        # save the classifier
        mdlFolder = folder+subjfolder+"model/"
        if not os.path.exists(mdlFolder):
            os.makedirs(mdlFolder)

        with open(mdlFolder+'RF_185_train_exact_seg.pkl', 'wb') as fid:
            cPickle.dump(classifier, fid)    