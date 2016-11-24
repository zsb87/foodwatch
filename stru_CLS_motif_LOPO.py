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


subj_list = ['Dzung','Will','Eric' ]#,'Gleb'
Ntrials = 1
threshold_str ='0.5'
protocol = 'inlabStr'
folder = '../'+protocol+'/subject/'

# 
# 
# generate all subjects dataframe file 
# 
# 

# trainfeatAllDf = pd.DataFrame()
# trainlabelAllDf = pd.DataFrame()
# testfeatAllDf = pd.DataFrame()
# testlabelAllDf = pd.DataFrame()

# for i_subj in range(len(subj_list)):
#     if i_subj == 0:
#         run = 2
#     elif i_subj == 1:
#         run = 2
#     elif i_subj == 2:
#         run = 5

#     subj = subj_list[i_subj]
#     trainsubjF = 'train'+subj + '/'
#     testsubjF = 'test'+subj + '/'
#     
#     trainfeatFoler = folder+trainsubjF+"feature/all_features/"
#     trainsegfolder = folder+trainsubjF+"segmentation/"

#     testfeatFoler = folder+testsubjF+"feature/all_features/"
#     testsegfolder = folder+testsubjF+"segmentation/"

#     train_df = pd.read_csv(trainfeatFoler+ "engy_run"+ str(run) +"_pred_features.csv" )
#     train_df['subj'] = i_subj
#     train_labelDf = pd.read_csv(trainsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv',names = ['label'])
#     # train_labelDf['subj'] = i_subj

#     test_df = pd.read_csv(testfeatFoler+ "engy_run"+ str(run) +"_pred_features.csv" )
#     test_df['subj'] = i_subj
#     test_labelDf = pd.read_csv(testsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv',names = ['label'])
#     # test_labelDf['subj'] = i_subj

#     trainfeatAllDf = pd.concat([trainfeatAllDf,train_df])
#     print(len(train_df))
#     trainlabelAllDf = pd.concat([trainlabelAllDf,train_labelDf])

#     testfeatAllDf = pd.concat([testfeatAllDf,test_df])
#     testlabelAllDf = pd.concat([testlabelAllDf,test_labelDf])
#     print(len(test_df))

# trainXYDf = pd.concat([trainfeatAllDf, trainlabelAllDf], axis=1, join_axes=[trainfeatAllDf.index])
# testXYDf = pd.concat([testfeatAllDf, testlabelAllDf], axis=1, join_axes=[testfeatAllDf.index])

# if not os.path.exists(folder+'trainOverall/'):
#     os.makedirs(folder+'trainOverall/')

# if not os.path.exists(folder+'testOverall/'):
#     os.makedirs(folder+'testOverall/')

# print(len(trainXYDf))
# print(len(testXYDf))

# trainXYDf.to_csv(folder+'trainOverall/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv', index =None)
# testXYDf.to_csv(folder+'testOverall/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv', index =None)

# trainXYDf = pd.read_csv(folder+'trainOverall/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv')
# testXYDf = pd.read_csv(folder+'testOverall/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv')

# allXYDf = pd.concat([trainXYDf, testXYDf])

# if not os.path.exists(folder+'overall/feature/'):
#     os.makedirs(folder+'overall/feature/')
# allXYDf.to_csv(folder+'overall/feature/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv', index =None)


########################################################################################
# 
# classification
# 
########################################################################################


# clf = RandomForestClassifier(n_estimators = 185)

allResultFolder = folder+'overall/result/seg_clf/engy_IS2IS_motif_generalized/subj'+str(len(subj_list))+'_LOPO/'
if not os.path.exists(allResultFolder):
    os.makedirs(allResultFolder)

crossValRes = pd.DataFrame(columns = ['LS'+str(i+1)+'O'+meas for i in range(len(subj_list)) for meas in ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']], index = range(Ntrials+1))

active_participant_counter_list = []

overallFeatFile = folder+'overall/feature/'+ 'engy_pred_featlabel_subj'+str(len(subj_list))+'.csv'
df = pd.read_csv(overallFeatFile)

for active_participant_counter in range(len(subj_list)):
    if 1:#(not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)):
        active_participant_counter_list.append(active_participant_counter)

        df_test = df.loc[df['subj']==active_participant_counter]
        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 
        X_test = df_test.iloc[:,:-3].as_matrix()
        y_test = df_test['label'].as_matrix()

        df_train = df.loc[df['subj']!=active_participant_counter]
        X_train = df_train.iloc[:,:-3].as_matrix()
        y_train = df_train['label'].as_matrix()
        

        for i in range(Ntrials+1):
            print(i)
            if i == Ntrials:
                crossValRes['LS' + str(active_participant_counter + 1)+'OPrec(pos)'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OPrec(pos)'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OF1(pos)'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OF1(pos)'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OTPR'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OTPR'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OFPR'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OFPR'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OSpecificity'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OSpecificity'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OMCC'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OMCC'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'OCKappa'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'OCKappa'].mean()
                crossValRes['LS' + str(active_participant_counter + 1)+'Ow-acc'][i] = crossValRes['LS' + str(active_participant_counter + 1)+'Ow-acc'].mean()
                break

            # cm_file = allResultFolder+"cm_LS"+str(active_participant_counter)+"O_trial"+str(i)+".csv"
            prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_ = clf_cm(X_train, X_test, y_train, y_test)
            crossValRes['LS' + str(active_participant_counter + 1)+'OPrec(pos)'][i] = prec_pos
            crossValRes['LS' + str(active_participant_counter + 1)+'OF1(pos)'][i] = f1_pos
            crossValRes['LS' + str(active_participant_counter + 1)+'OTPR'][i] = TPR
            crossValRes['LS' + str(active_participant_counter + 1)+'OFPR'][i] = FPR
            crossValRes['LS' + str(active_participant_counter + 1)+'OSpecificity'][i] = Specificity
            crossValRes['LS' + str(active_participant_counter + 1)+'OMCC'][i] = MCC
            crossValRes['LS' + str(active_participant_counter + 1)+'OCKappa'][i] = CKappa
            crossValRes['LS' + str(active_participant_counter + 1)+'Ow-acc'][i] = w_acc


print(active_participant_counter_list)

crossValRes['aveLOPO_Prec(pos)'] = crossValRes[['LS' + str(i + 1) + 'OPrec(pos)' for i in active_participant_counter_list]].mean(axis = 1)
crossValRes['aveLOPO_F1(pos)'] = crossValRes[['LS' + str(i + 1) + 'OF1(pos)' for i in active_participant_counter_list]].mean(axis = 1)
crossValTPR = crossValRes[['LS' + str(i + 1) + 'OTPR' for i in active_participant_counter_list]]
crossValRes['aveLOPO_TPR'] = crossValTPR.mean(axis = 1)
crossValFPR = crossValRes[['LS' + str(i + 1) + 'OFPR' for i in active_participant_counter_list]]
crossValRes['aveLOPO_FPR'] = crossValFPR.mean(axis = 1)

crossValSpe = crossValRes[['LS' + str(i + 1) + 'OSpecificity' for i in active_participant_counter_list]]
crossValRes['aveLOPO_Specificity'] = crossValSpe.mean(axis = 1)
crossValMCC = crossValRes[['LS' + str(i + 1) + 'OMCC' for i in active_participant_counter_list]]
crossValRes['aveLOPO_MCC'] = crossValMCC.mean(axis = 1)
crossValCKappa = crossValRes[['LS' + str(i + 1) + 'OCKappa' for i in active_participant_counter_list]]
crossValRes['aveLOPO_CKappa'] = crossValCKappa.mean(axis = 1)
crossValRes['aveLOPO_w-acc'] = crossValRes[['LS' + str(i + 1) + 'Ow-acc' for i in active_participant_counter_list]].mean(axis = 1)

crossValRes.to_csv(allResultFolder+"differentThres_RF185_result.csv", index = None)

