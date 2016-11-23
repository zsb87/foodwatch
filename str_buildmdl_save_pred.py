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


# ,'Cao', 'JC', 'Eric', 'Jiapeng','Rawan'
subjs = ['Cao']#['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',

for run in range(1):
    run = 3

    for subj in subjs:

        #
        # import features and model, do classification
        #         
        columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
        crossValRes = pd.DataFrame(columns = columns, index = range(5))
        active_p_cnt = 0     



        for threshold_str in ['0.7']:

            testsubj ='test'+subj
            trainsubj = 'train'+subj

            trainsubjF = trainsubj + '(8Hz)/'
            testsubjF = testsubj + '(8Hz)/'
            folder = '../inlabStr/subject/'
            trainfeatFoler = folder+trainsubjF+"feature/all_features/"
            trainsegfolder = folder+trainsubjF+"segmentation/"
            testfeatFoler = folder+testsubjF+"feature/all_features/"
            testsegfolder = folder+testsubjF+"segmentation/"


            df = pd.read_csv(trainfeatFoler+ "engy_run"+ str(run) +"_pred_features.csv" )
            labelDf = pd.read_csv(trainsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv',names = ['label'])

            # 
            # notice:   duration should not be included in features 
            #           as in detection period this distinguishable feature will be in different distribution
            # 
            X = df.iloc[:,:-1].as_matrix()
            Y = labelDf['label'].iloc[:].as_matrix()

            classifier = RandomForestClassifier(n_estimators=185)
            classifier.fit(X, Y)


            # save the classifier
            mdlFolder = folder+trainsubjF+"model/"
            if not os.path.exists(mdlFolder):
                os.makedirs(mdlFolder)

            with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'wb') as fid:
                cPickle.dump(classifier, fid)    



            folder = '../inlabStr/subject/'


            testfeatFolder = '../inlabStr/subject/'+testsubj + '(8Hz)/feature/all_features/'
            testfeatFile = testfeatFolder+ "engy_run"+ str(run) +'_pred_features.csv'
            df_all = pd.read_csv(testfeatFile)
            print(len(df_all))

            labelFile = folder+testsubj + '(8Hz)/segmentation/engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv'
            labelDf = pd.read_csv(labelFile, names = ['label'])
            print(len(labelDf))


            mdlFolder = folder+trainsubjF+"model/"
            # save the classifier
            with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'rb') as fid:
                classifier = cPickle.load(fid)

            # 
            # notice:   duration should not be included in features 
            #           as in detection period this distinguishable feature will be in different distribution
            # 
            X = df_all.iloc[:,:-1].as_matrix()
            Y = labelDf['label'].as_matrix()

            

            prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_, y_pred = clf_cm_pickle(classifier, X, Y)

            y_predDf = pd.DataFrame(y_pred,columns=['label'])
            y_predDf.to_csv(testsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/pred_seg_labels.csv',header=None, index = None )

            crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
            crossValRes['F1(pos)'][active_p_cnt] = f1_pos
            crossValRes['TPR'][active_p_cnt] = TPR
            crossValRes['FPR'][active_p_cnt] = FPR
            crossValRes['Specificity'][active_p_cnt] = Specificity
            crossValRes['MCC'][active_p_cnt] = MCC
            crossValRes['CKappa'][active_p_cnt] = CKappa
            crossValRes['w-acc'][active_p_cnt] = w_acc
            active_p_cnt = active_p_cnt+1

        outfolder = '../inlabStr/result/seg_clf/accx_IS2ISseg_personalized/'+subj+'/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        # crossValRes.to_csv( outfolder+'RF_185_exact_seg_on_trainmdl_thres0.7_run'+str(run)+'(109).csv', index = None)
