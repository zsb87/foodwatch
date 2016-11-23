import os
import re
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import random

from sklearn import preprocessing
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *            
from scipy.signal import *
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import _pickle as cPickle
from stru_utils import *

save_flg = 1



subjs = ['Dzung']#['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',

if 1:

    for subj in subjs:

        testsubj = 'test'+subj

        subjfolder = testsubj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        featFolder = folder+subjfolder+"feature/all_features/"

        datafile =  folder+subjfolder+"testdata.csv"
        segfolder = folder+subjfolder+"segmentation/"
           
        # 
        #   generate prediction segment features
        #   1. generate raw data files from Head tail 
        # 

        if not os.path.exists(featFolder):
            os.makedirs(featFolder)

        # read in the raw data file
        # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
        #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
        r_df = pd.read_csv(datafile)
        
        for _, dirnames, filenames in os.walk(segfolder+'pred/'):
            n_gest = len(filenames)

            if n_gest == 0:
                continue
            else:

                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    
                    # note: this is especially for latest version
                    predActFile = 'pred_acc_headtail_reduced_1.csv'
                    # predActFile = 'pred_acc_headtail_reduced_' + str(i+1) + '.csv'
                    predActFilePath = segfolder+'pred/' + predActFile
                    gt_headtail = pd.read_csv(predActFilePath, names = ['EnergyStart','EnergyEnd','EnergyDur','0'])

                    gt_headtail['Start'] = gt_headtail['EnergyStart']*2
                    gt_headtail['End'] = gt_headtail['EnergyEnd']*2 + 3

                    for i in range(len(gt_headtail)):

                        saveFileName = 'pred_gesture_' + str(i) + '.csv'

                        if not os.path.exists(segfolder+'pred_data/'):
                            os.makedirs(segfolder+'pred_data/')
                        saveFilePath = segfolder+'pred_data/' + saveFileName

                        dataStart = int(gt_headtail['Start'].iloc[i])
                        dataEnd = int(gt_headtail['End'].iloc[i])

                        r_df_gesture = r_df.iloc[dataStart:dataEnd]
                        r_df_gesture.to_csv(saveFilePath)



        # 
        #   generate prediction segment features
        #   2. from raw data of segments generate features
        # 

        predFolder = segfolder+'pred_data/'

        allfeatDF = pd.DataFrame()
        for _, dirnames, filenames in os.walk(predFolder):
            n_gest = len(filenames)

            if n_gest == 0:
                print("no prediction file for current subject")
                continue
            else:
                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    predActFile = 'pred_gesture_' + str(i) + '.csv'
                    r_df = pd.read_csv(predFolder + predActFile)

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                    r_df = df_iter_flt(r_df)
                    r_df = add_pitch_roll(r_df)
                    # generate the features
                    feat = gen_feat(r_df)

                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    allfeatDF = pd.concat([allfeatDF,featDF])


        outfile = featFolder + "pred_features.csv"
        allfeatDF.to_csv(outfile)

        trainsubj = 'train'+subj
        subjfolder = trainsubj + '(8Hz)/'

        testfeatFolder = '../inlabStr/subject/'+testsubj + '(8Hz)/'+'/feature/all_features/'
        testfeatFile = testfeatFolder+'pred_features.csv'
        df_all = pd.read_csv(testfeatFile)

        labelFile = folder+testsubj + '(8Hz)/'+"segmentation/pred_label/seg_labels.csv"
        labelDf = pd.read_csv(labelFile, names = ['label'])


        mdlFolder = folder+subjfolder+"model/"
        # save the classifier
        with open(mdlFolder+'RF_185.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)

        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 
        X = df_all.iloc[:,1:-1].as_matrix()
        Y = labelDf['label'].as_matrix()

        columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
        crossValRes = pd.DataFrame(columns = columns, index = range(1))
        active_p_cnt = 0 

        prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_ = clf_cm_pickle(classifier, X, Y)

        crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
        crossValRes['F1(pos)'][active_p_cnt] = f1_pos
        crossValRes['TPR'][active_p_cnt] = TPR
        crossValRes['FPR'][active_p_cnt] = FPR
        crossValRes['Specificity'][active_p_cnt] = Specificity
        crossValRes['MCC'][active_p_cnt] = MCC
        crossValRes['CKappa'][active_p_cnt] = CKappa
        crossValRes['w-acc'][active_p_cnt] = w_acc

    outfolder = '../inlabStr/result/seg_clf/IS2ISseg_personalized/'+subj
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    crossValRes.to_csv( outfolder+"RF_185_result(109).csv", index = None)