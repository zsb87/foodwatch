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
allResultFolder = "./result/personalized/split/"

if not os.path.exists(allResultFolder):
    os.makedirs(allResultFolder)

columns = ['setSplitRun' + str(i + 1) + meas for i in range(Ntrials) for meas in ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']]
columns = columns +['avePrec(pos)','aveF1(pos)','aveTPR', 'aveFPR','aveSpecificity','aveMCC','aveCKappa','avew-acc']

crossValRes = pd.DataFrame(columns = columns, index = range(len(subjs)+1))

# form is in format:
# 
#           Run1Prec    Run1F1  ....    Run2Prec    Run2F1  ..      avePrec     aveF1    
#   subj1
#   subj2
#   ...
#   aveSubj
if 1:

    for p, subj in enumerate(subjs):

        if (not (p == 3)) and (not (p == 4))  and (not (p == 6)) :
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
            equiv = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1,7:1, 8:1, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0}
            df["f-nf"] = df["activity"].map(equiv)
            
            Y = df['f-nf'].as_matrix()


            for j in range(Ntrials):

                X_train, X_test = k_fold_split(X, Nfolds, j)
                y_train, y_test = k_fold_split(Y, Nfolds, j)
                # cm_file = allResultFolder + "cm_trial"+str(i)+"_fold"+str(j)+".csv"
                prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_ = clf_cm(X_train, X_test, y_train, y_test)

                crossValRes['setSplitRun' + str(j + 1)+'Prec(pos)'][p] = prec_pos
                crossValRes['setSplitRun' + str(j + 1)+'F1(pos)'][p] = f1_pos
                crossValRes['setSplitRun' + str(j + 1)+'TPR'][p] = TPR
                crossValRes['setSplitRun' + str(j + 1)+'FPR'][p] = FPR
                crossValRes['setSplitRun' + str(j + 1)+'Specificity'][p] = Specificity
                crossValRes['setSplitRun' + str(j + 1)+'MCC'][p] = MCC
                crossValRes['setSplitRun' + str(j + 1)+'CKappa'][p] = CKappa
                crossValRes['setSplitRun' + str(j + 1)+'w-acc'][p] = w_acc

    # average for all subjects
    for j in range(Ntrials):   
        crossValRes['setSplitRun' + str(j + 1)+'Prec(pos)'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'Prec(pos)'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'F1(pos)'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'F1(pos)'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'TPR'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'TPR'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'FPR'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'FPR'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'Specificity'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'Specificity'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'MCC'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'MCC'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'CKappa'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'CKappa'].mean()
        crossValRes['setSplitRun' + str(j + 1)+'w-acc'][len(subjs)] = crossValRes['setSplitRun' + str(j + 1)+'w-acc'].mean()



crossValRes['avePrec(pos)'] = crossValRes[['setSplitRun' + str(i + 1) + 'Prec(pos)' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveF1(pos)'] = crossValRes[['setSplitRun' + str(i + 1) + 'F1(pos)' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveTPR'] = crossValRes[['setSplitRun' + str(i + 1) + 'TPR' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveFPR'] = crossValRes[['setSplitRun' + str(i + 1) + 'FPR' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveSpecificity'] = crossValRes[['setSplitRun' + str(i + 1) + 'Specificity' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveMCC'] = crossValRes[['setSplitRun' + str(i + 1) + 'MCC' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveCKappa'] = crossValRes[['setSplitRun' + str(i + 1) + 'CKappa' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['avew-acc'] = crossValRes[['setSplitRun' + str(i + 1) + 'w-acc' for i in range(Ntrials)]].mean(axis = 1)


crossValRes.to_csv( allResultFolder+"73split_allsubj.csv", index = None)
