import os
import re
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import random
import glob

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
import shutil

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


        predActFilePath = segfolder+'accx_run1_pred/pred_acc_headtail_reduced_1.csv'
        gt_headtail = pd.read_csv(predActFilePath, names = ['Start','End','EnergyDur','dist'])

        if not os.path.exists(segfolder+'accx_run1_pred_data/'):
            os.makedirs(segfolder+'accx_run1_pred_data/')

        for f in glob.glob(segfolder+'accx_run1_pred_data/*'):
            os.remove(f)
        
        for i in range(len(gt_headtail)):

            # if os.path.exists(segfolder+'accx_pred_data/'):
            #     shutil.rmtree(segfolder+'accx_pred_data/')

            saveFilePath = segfolder+'accx_run1_pred_data/' + 'accx_pred_gesture_' + str(i) + '.csv'

            dataStart = int(gt_headtail['Start'].iloc[i])
            dataEnd = int(gt_headtail['End'].iloc[i])

            r_df_gesture = r_df.iloc[dataStart:dataEnd]
            r_df_gesture.to_csv(saveFilePath)



        # 
        #   generate prediction segment features
        #   2. from raw data of segments generate features
        # 

        predFolder = segfolder+'accx_run1_pred_data/'

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

                    r_df = pd.read_csv(predFolder + 'accx_pred_gesture_' + str(i) + '.csv')

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                    r_df = df_iter_flt(r_df)
                    r_df = add_pitch_roll(r_df)
                    # generate the features
                    feat = gen_feat(r_df)

                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    allfeatDF = pd.concat([allfeatDF,featDF])


        outfile = featFolder + "accx_run1_pred_features.csv"
        allfeatDF.to_csv(outfile, index =None)





        trainsubj = 'train'+subj

        subjfolder = trainsubj + '(8Hz)/'
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


        predActFilePath = segfolder+'accx_run1_pred/pred_acc_headtail_reduced_1.csv'
        gt_headtail = pd.read_csv(predActFilePath, names = ['Start','End','EnergyDur','dist'])

        if not os.path.exists(segfolder+'accx_run1_pred_data/'):
            os.makedirs(segfolder+'accx_run1_pred_data/')

        for f in glob.glob(segfolder+'accx_run1_pred_data/*'):
            os.remove(f)
        
        for i in range(len(gt_headtail)):

            # if os.path.exists(segfolder+'accx_pred_data/'):
            #     shutil.rmtree(segfolder+'accx_pred_data/')

            saveFilePath = segfolder+'accx_run1_pred_data/' + 'accx_pred_gesture_' + str(i) + '.csv'

            dataStart = int(gt_headtail['Start'].iloc[i])
            dataEnd = int(gt_headtail['End'].iloc[i])

            r_df_gesture = r_df.iloc[dataStart:dataEnd]
            r_df_gesture.to_csv(saveFilePath)



        # 
        #   generate prediction segment features
        #   2. from raw data of segments generate features
        # 

        predFolder = segfolder+'accx_run1_pred_data/'

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

                    r_df = pd.read_csv(predFolder + 'accx_pred_gesture_' + str(i) + '.csv')

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                    r_df = df_iter_flt(r_df)
                    r_df = add_pitch_roll(r_df)
                    # generate the features
                    feat = gen_feat(r_df)

                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    allfeatDF = pd.concat([allfeatDF,featDF])


        outfile = featFolder + "accx_run1_pred_features.csv"
        allfeatDF.to_csv(outfile, index =None)
        #
        # import features and model, do classification
        #         
        

    #     folder = '../inlabStr/subject/'

    #     trainsubj = 'train'+subj
    #     trainsubjfolder = trainsubj + '(8Hz)/'

    #     testsubj = 'test'+subj
    #     testsubjfolder = testsubj + '(8Hz)/'


    #     testfeatFolder = '../inlabStr/subject/'+testsubj + '(8Hz)/feature/all_features/'
    #     testfeatFile = testfeatFolder+'pred_features.csv'
    #     df_all = pd.read_csv(testfeatFile)
    #     print(len(df_all))

    #     labelFile = folder+testsubj + '(8Hz)/segmentation/accx_pred_label/seg_labels.csv'
    #     labelDf = pd.read_csv(labelFile, names = ['label'])
    #     print(len(labelDf))


    #     mdlFolder = folder+trainsubjfolder+"model/"
    #     # save the classifier
    #     with open(mdlFolder+'RF_185_train_exact_seg.pkl', 'rb') as fid:
    #         classifier = cPickle.load(fid)

    #     # 
    #     # notice:   duration should not be included in features 
    #     #           as in detection period this distinguishable feature will be in different distribution
    #     # 
    #     X = df_all.iloc[:,:-1].as_matrix()
    #     Y = labelDf['label'].as_matrix()

    #     columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
    #     crossValRes = pd.DataFrame(columns = columns, index = range(1))
    #     active_p_cnt = 0 

    #     prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_ = clf_cm_pickle(classifier, X, Y)

    #     crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
    #     crossValRes['F1(pos)'][active_p_cnt] = f1_pos
    #     crossValRes['TPR'][active_p_cnt] = TPR
    #     crossValRes['FPR'][active_p_cnt] = FPR
    #     crossValRes['Specificity'][active_p_cnt] = Specificity
    #     crossValRes['MCC'][active_p_cnt] = MCC
    #     crossValRes['CKappa'][active_p_cnt] = CKappa
    #     crossValRes['w-acc'][active_p_cnt] = w_acc

    # outfolder = '../inlabStr/result/seg_clf/accx_IS2ISseg_personalized/'+subj
    # if not os.path.exists(outfolder):
    #     os.makedirs(outfolder)

    # crossValRes.to_csv( outfolder+"RF_185_exact_seg_on_trainmdl_5measureThre(109).csv", index = None)