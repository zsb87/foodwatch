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
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support as score
from stru_utils import *
from stru_settings import *
from scipy.stats import *



#  this subject list is following the order in the subject ID map.
subjs = ['Rawan','Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric']#
Nfolds = 10
Ntrials = 5

# avgRes = pd.DataFrame(columns = ['subject','mean fscore','var fscore'], index = range(len(subj_list)))
clf = RandomForestClassifier(n_estimators = 100)
allResultFolder = "./subject/overall/result/personalized/10fCV/"

if not os.path.exists(allResultFolder):
    os.makedirs(allResultFolder)

columns = ['Fold' + str(i + 1) + meas for i in range(10) for meas in ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']]
columns = columns +['aveFoldPrec(pos)','aveFoldF1(pos)','aveFoldTPR', 'aveFoldFPR','aveFoldSpecificity','aveFoldMCC','aveFoldCKappa','aveFoldw-acc']

for active_participant_counter, subj in enumerate(subjs):
    crossValRes = pd.DataFrame(columns = columns, index = range(Ntrials+1))

    if (not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)) :
        print(subj)

        subjfolder = subj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        featFolder = folder+subjfolder+"feature/"
        datafile =  folder+subjfolder+"testdata.csv"
        segfolder = folder+subjfolder+"segmentation/"
        clsfolder = folder+subjfolder+"classification/"

        act_rootfolder = segfolder+'activity/'
        allfeatFolder = featFolder+"all_features/"
        detAllfeatFolder = allfeatFolder+"detection/"

        if not os.path.exists(clsfolder):
            os.makedirs(clsfolder)
        if not os.path.exists(allfeatFolder):
            os.makedirs(allfeatFolder)
        
        gtFeatPath = featFolder + "gt_features.csv"
        outfile = clsfolder + "cm_gt_cls.csv"

        df = pd.read_csv(gtFeatPath)

        print(len(df))
        df = df.dropna() 
        print(len(df))

        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 
        X = df.iloc[:,:-2].as_matrix()

        equiv = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0,7:0, 8:0, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}
        df["f-nf"] = df["activity"].map(equiv)
        
        Y = df['f-nf'].as_matrix()


        for i in range(Ntrials+1):
            print(i)
            if i == Ntrials:
                for j in range(Nfolds):   
                    crossValRes['Fold' + str(j + 1)+'Prec(pos)'][i] = crossValRes['Fold' + str(j + 1)+'Prec(pos)'].mean()
                    crossValRes['Fold' + str(j + 1)+'F1(pos)'][i] = crossValRes['Fold' + str(j + 1)+'F1(pos)'].mean()
                    crossValRes['Fold' + str(j + 1)+'TPR'][i] = crossValRes['Fold' + str(j + 1)+'TPR'].mean()
                    crossValRes['Fold' + str(j + 1)+'FPR'][i] = crossValRes['Fold' + str(j + 1)+'FPR'].mean()
                    crossValRes['Fold' + str(j + 1)+'Specificity'][i] = crossValRes['Fold' + str(j + 1)+'Specificity'].mean()
                    crossValRes['Fold' + str(j + 1)+'MCC'][i] = crossValRes['Fold' + str(j + 1)+'MCC'].mean()
                    crossValRes['Fold' + str(j + 1)+'CKappa'][i] = crossValRes['Fold' + str(j + 1)+'CKappa'].mean()
                    crossValRes['Fold' + str(j + 1)+'w-acc'][i] = crossValRes['Fold' + str(j + 1)+'w-acc'].mean()
                break

            for j in range(Nfolds):
                X_train, X_test = k_fold_split(X, Nfolds, j)
                y_train, y_test = k_fold_split(Y, Nfolds, j)
                # cm_file = allResultFolder + "cm_trial"+str(i)+"_fold"+str(j)+".csv"
                prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_ = clf_cm(X_train, X_test, y_train, y_test)

                crossValRes['Fold' + str(j + 1)+'Prec(pos)'][i] = prec_pos
                crossValRes['Fold' + str(j + 1)+'F1(pos)'][i] = f1_pos
                crossValRes['Fold' + str(j + 1)+'TPR'][i] = TPR
                crossValRes['Fold' + str(j + 1)+'FPR'][i] = FPR
                crossValRes['Fold' + str(j + 1)+'Specificity'][i] = Specificity
                crossValRes['Fold' + str(j + 1)+'MCC'][i] = MCC
                crossValRes['Fold' + str(j + 1)+'CKappa'][i] = CKappa
                crossValRes['Fold' + str(j + 1)+'w-acc'][i] = w_acc


        crossValHit = crossValRes[['Fold' + str(i + 1) + 'Prec(pos)' for i in range(10)]]
        crossValRes['aveFoldPrec(pos)'] = crossValHit.mean(axis = 1)
        crossValHit = crossValRes[['Fold' + str(i + 1) + 'F1(pos)' for i in range(10)]]
        crossValRes['aveFoldF1(pos)'] = crossValHit.mean(axis = 1)
        crossValTPR = crossValRes[['Fold' + str(i + 1) + 'TPR' for i in range(10)]]
        crossValRes['aveFoldTPR'] = crossValTPR.mean(axis = 1)
        crossValFPR = crossValRes[['Fold' + str(i + 1) + 'FPR' for i in range(10)]]
        crossValRes['aveFoldFPR'] = crossValFPR.mean(axis = 1)
        crossValSpe = crossValRes[['Fold' + str(i + 1) + 'Specificity' for i in range(10)]]
        crossValRes['aveFoldSpecificity'] = crossValSpe.mean(axis = 1)
        crossValMCC = crossValRes[['Fold' + str(i + 1) + 'MCC' for i in range(10)]]
        crossValRes['aveFoldMCC'] = crossValMCC.mean(axis = 1)
        crossValCKappa = crossValRes[['Fold' + str(i + 1) + 'CKappa' for i in range(10)]]
        crossValRes['aveFoldCKappa'] = crossValCKappa.mean(axis = 1)
        crossValCKappa = crossValRes[['Fold' + str(i + 1) + 'w-acc' for i in range(10)]]
        crossValRes['aveFoldw-acc'] = crossValCKappa.mean(axis = 1)

        crossValRes.to_csv( allResultFolder+"10fCV_subj"+str(active_participant_counter)+".csv", index = None)



