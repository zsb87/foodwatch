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

subjs = ['Dzung' ]#'Rawan','Eric', ,['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',
mthres_list = ['0.5','0.6','0.7','0.8','0.9']

for mthres in mthres_list:
    for subj in subjs:

        testsubj ='test'+subj
        trainsubj = 'train'+subj

        trainsubjF = trainsubj + '(8Hz)/'
        testsubjF = testsubj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        trainfeatFoler = folder+trainsubjF+"feature/all_features/"
        trainsegfolder = folder+trainsubjF+"segmentation/"

        
        #   # build model 
        

        # df = pd.read_csv(trainfeatFoler+'pred_features.csv'   )
        # labelDf = pd.read_csv(trainsegfolder+'accx_pred_label_thre'+mthres+'/seg_labels.csv',names = ['label'])

        # # 
        # # notice:   duration should not be included in features 
        # #           as in detection period this distinguishable feature will be in different distribution
        # # 
        # X = df.iloc[:,:-1].as_matrix()
        # Y = labelDf['label'].iloc[:].as_matrix()

        # classifier = RandomForestClassifier(n_estimators=185)
        # classifier.fit(X, Y)


        # # save the classifier
        # mdlFolder = folder+trainsubjF+"model/"
        # if not os.path.exists(mdlFolder):
        #     os.makedirs(mdlFolder)

        # with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+mthres+'.pkl', 'wb') as fid:
        #     cPickle.dump(classifier, fid)    



        testsubj = 'test'+subj
        trainsubjF = trainsubj + '(8Hz)/'

        testfeatFolder = '../inlabStr/subject/' + testsubj + '(8Hz)/'+'/feature/all_features/'
        testfeatFile = testfeatFolder+'pred_features.csv'
        df_all = pd.read_csv(testfeatFile)

        labelFile = folder+testsubj + '(8Hz)/'+'segmentation/accx_pred_label_thre'+mthres+'/seg_labels.csv'
        labelDf = pd.read_csv(labelFile, names = ['label'])


        mdlFolder = folder+trainsubjF+"model/"
        # save the classifier
        with open(mdlFolder+'RF_185_train_exact_seg.pkl', 'rb') as fid:
            classifier = cPickle.load(fid)

        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 
        X = df_all.iloc[:,:-1].as_matrix()
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

    outfolder = '../inlabStr/result/seg_clf/accx_IS2ISseg_personalized/'+subj
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    crossValRes.to_csv( outfolder+'RF_185_train_trainset_exact_test_testset_motif_thre'+mthres+'(109).csv', index = None)