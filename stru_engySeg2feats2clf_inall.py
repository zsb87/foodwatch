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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("participant_name")
args = parser.parse_args()
print ("stru_engySeg2feats2clf_inall.py")
print (args.participant_name)

subj = args.participant_name

save_flg = 1
protocol = 'inlabUnstr'

# ,'Cao', 'JC', 'Eric', 'Jiapeng','Rawan'
# subjs = [  'Dzung']#['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', ]# 'Matt',

for run in [3]:

    # for subj in subjs:
    if 1:

        # generate feature files both for test and train set of this subj

        split_subjs = ['train'+subj]#,'test'+subj

        for split_subj in split_subjs:

            # testsubj = 'test'+subj
            subjfolder = split_subj + '(8Hz)/'
            folder = '../'+protocol+'/subject/'
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

            predActFilePath = segfolder+'engy_run'+str(run)+'_pred/pred_acc_headtail_reduced_1.csv'
            gt_headtail = pd.read_csv(predActFilePath, names = ['Start','End','EnergyDur','dist'])

            # if os.path.exists(segfolder+'accx_pred_data/'):
            #     shutil.rmtree(segfolder+'accx_pred_data/')



            if not os.path.exists(segfolder+'engy_run'+str(run)+'_pred_data/'):
                os.makedirs(segfolder+'engy_run'+str(run)+'_pred_data/')

            for f in glob.glob(segfolder+'engy_run'+str(run)+'_pred_data/*'):
                os.remove(f)

            for i in range(len(gt_headtail)):
                saveFilePath = segfolder+'engy_run'+str(run)+'_pred_data/' + 'engy_pred_gesture_' + str(i) + '.csv'

                dataStart = int(gt_headtail['Start'].iloc[i])*2 - 2
                dataEnd = int(gt_headtail['End'].iloc[i])*2 + 1

                r_df_gesture = r_df.iloc[dataStart:dataEnd]
                # if dataEnd < len(r_df):
                if 1:
                    r_df_gesture.to_csv(saveFilePath)



            # 
            #   generate prediction segment features
            #   2. from raw data of segments generate features
            # 

            predFolder = segfolder+'engy_run'+str(run)+'_pred_data/'

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

                        r_df = pd.read_csv(predFolder + 'engy_pred_gesture_' + str(i) + '.csv')

                        # pass raw data into filter
                        r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                        r_df = df_iter_flt(r_df)
                        print(i)
                        r_df = add_pitch_roll(r_df)
                        # generate the features
                        feat = gen_feat(r_df)

                        featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                        allfeatDF = pd.concat([allfeatDF,featDF])


            outfile = featFolder + "engy_run"+ str(run) +"_pred_features.csv"
            allfeatDF.to_csv(outfile, index =None)







        #
        # import features and model, do classification
        #         
        columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
        crossValRes = pd.DataFrame(columns = columns, index = range(5))
        active_p_cnt = 0     



        for threshold_str in ['0.5','0.6','0.7','0.8','0.9']:

            testsubj ='test'+subj
            trainsubj = 'train'+subj

            trainsubjF = trainsubj + '(8Hz)/'
            testsubjF = testsubj + '(8Hz)/'
            folder = '../'+protocol+'/subject/'
            trainfeatFoler = folder+trainsubjF+"feature/all_features/"
            trainsegfolder = folder+trainsubjF+"segmentation/"


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



            folder = '../'+protocol+'/subject/'


            testfeatFolder = '../'+protocol+'/subject/'+testsubj + '(8Hz)/feature/all_features/'
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

            

            prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_,y_pred = clf_cm_pickle(classifier, X, Y)

            crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
            crossValRes['F1(pos)'][active_p_cnt] = f1_pos
            crossValRes['TPR'][active_p_cnt] = TPR
            crossValRes['FPR'][active_p_cnt] = FPR
            crossValRes['Specificity'][active_p_cnt] = Specificity
            crossValRes['MCC'][active_p_cnt] = MCC
            crossValRes['CKappa'][active_p_cnt] = CKappa
            crossValRes['w-acc'][active_p_cnt] = w_acc
            active_p_cnt = active_p_cnt+1

        outfolder = '../'+protocol+'/result/seg_clf/accx_IS2ISseg_personalized/'+subj+'/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        crossValRes.to_csv( outfolder+'RF_185_exact_seg_on_trainmdl_multi-thre_run'+str(run)+'(109).csv', index = None)