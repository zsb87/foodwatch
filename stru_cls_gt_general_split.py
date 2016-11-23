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



def tt_split(XY, train_ratio):
    # eg: train_ratio = 0.7
    length = len(XY)
    test_enum = range(int((1-train_ratio)*10))
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


def tt_split_rand(XY, train_ratio, seed):
    # eg: train_ratio = 0.7
    import random

    numL = list(range(10))
    random.seed(seed)
    random.shuffle(numL)

    length = len(XY)
    test_enum = numL[0:int((1-train_ratio)*10)]
    test_ind = []
    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    # test_ind = np.arange(n, length, k)
    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


#  this subject list is following the order in the subject ID map.
subjs = ['Rawan','Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric']#
Ntrials = 5
train_ratio = 0.7
# avgRes = pd.DataFrame(columns = ['subject','mean fscore','var fscore'], index = range(len(subj_list)))
clf = RandomForestClassifier(n_estimators = 100)
allResultFolder = "./result/generalized/split/"

if not os.path.exists(allResultFolder):
    os.makedirs(allResultFolder)

df_all = pd.DataFrame()

overallFeatFolder = '../inlabStr/subject/overall/feature/'
if not os.path.exists(overallFeatFolder):
    os.makedirs(overallFeatFolder)

overallFeatFile = overallFeatFolder+'gt_features_7subj.csv'

if not os.path.isfile(overallFeatFile):
    for active_participant_counter, subj in enumerate(subjs):
        crossValRes = pd.DataFrame(columns = columns, index = range(Ntrials+1))
        if (not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)):
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
            
            # add one column
            equiv = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0,7:0, 8:0, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1}
            df["f-nf"] = df["activity"].map(equiv)
            df["subj"] = active_participant_counter
            
            df_all = pd.concat([df_all,df])

    df_all.to_csv(overallFeatFile,index=None)


df_all = pd.read_csv(overallFeatFile)

# 
# notice:   duration should not be included in features 
#           as in detection period this distinguishable feature will be in different distribution
# 
X = df_all.iloc[:,:-4].as_matrix()
Y = df_all['f-nf'].as_matrix()

columns = ['setSplitRun' + str(i + 1) + meas for i in range(Ntrials) for meas in ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']]
columns = columns +['avePrec(pos)','aveF1(pos)','aveTPR', 'aveFPR','aveSpecificity','aveMCC','aveCKappa','avew-acc']
# mind the index range!@@!
crossValRes = pd.DataFrame(columns = columns, index = range(1))

p=0
for i in range(Ntrials+1):
    print(i)
    for j in range(Ntrials):
        X_train, X_test = tt_split_rand(X, train_ratio,j)
        y_train, y_test = tt_split_rand(Y, train_ratio,j)
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


for j in range(Ntrials):   
    crossValRes['setSplitRun' + str(j + 1)+'Prec(pos)'][p] = crossValRes['setSplitRun' + str(j + 1)+'Prec(pos)'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'F1(pos)'][p] = crossValRes['setSplitRun' + str(j + 1)+'F1(pos)'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'TPR'][p] = crossValRes['setSplitRun' + str(j + 1)+'TPR'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'FPR'][p] = crossValRes['setSplitRun' + str(j + 1)+'FPR'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'Specificity'][p] = crossValRes['setSplitRun' + str(j + 1)+'Specificity'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'MCC'][p] = crossValRes['setSplitRun' + str(j + 1)+'MCC'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'CKappa'][p] = crossValRes['setSplitRun' + str(j + 1)+'CKappa'].mean()
    crossValRes['setSplitRun' + str(j + 1)+'w-acc'][p] = crossValRes['setSplitRun' + str(j + 1)+'w-acc'].mean()


crossValRes['avePrec(pos)'] = crossValRes[['setSplitRun' + str(i + 1) + 'Prec(pos)' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveF1(pos)'] = crossValRes[['setSplitRun' + str(i + 1) + 'F1(pos)' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveTPR'] = crossValRes[['setSplitRun' + str(i + 1) + 'TPR' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveFPR'] = crossValRes[['setSplitRun' + str(i + 1) + 'FPR' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveSpecificity'] = crossValRes[['setSplitRun' + str(i + 1) + 'Specificity' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveMCC'] = crossValRes[['setSplitRun' + str(i + 1) + 'MCC' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['aveCKappa'] = crossValRes[['setSplitRun' + str(i + 1) + 'CKappa' for i in range(Ntrials)]].mean(axis = 1)
crossValRes['avew-acc'] = crossValRes[['setSplitRun' + str(i + 1) + 'w-acc' for i in range(Ntrials)]].mean(axis = 1)

crossValRes.to_csv( allResultFolder+"73split_generalized.csv", index = None)
