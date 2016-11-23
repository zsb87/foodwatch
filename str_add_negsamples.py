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

subjs = ['JC', 'Eric', 'Jiapeng']#'Rawan','Eric', ,['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',
mthres_list = ['0.5']#,'0.6','0.7','0.8','0.9'
threshold_str = '0.5'
run = 3
# if 1:
for mthres in mthres_list:
    for subj in subjs:

        testsubj ='test'+subj
        trainsubj = 'train'+subj

        trainsubjF = trainsubj + '(8Hz)/'
        testsubjF = testsubj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        trainfeatFoler = folder+trainsubjF+"feature/all_features/"
        trainsegfolder = folder+trainsubjF+"segmentation/"
        testfeatFoler = folder+testsubjF+"feature/all_features/"
        testsegfolder = folder+testsubjF+"segmentation/"

        
          # build model 
        
        df_exact = pd.read_csv(trainfeatFoler+'gt_fnf_feats.csv'   )

        df_engy = pd.read_csv(trainfeatFoler+'engy_run3_pred_features.csv'   )
        labelDf = pd.read_csv(trainsegfolder+'engy_run3_pred_label_thre'+mthres+'/seg_labels.csv',names = ['f-nf'])

        engy_XY = pd.concat([df_engy, labelDf], axis=1, join_axes=[df_engy.index])
        # XY = engy_XY.loc[engy_XY['f-nf'] == 1]
        XY = engy_XY
        XY = XY.append(df_exact)

        # 
        # notice:   duration should not be included in features 
        #           as in detection period this distinguishable feature will be in different distribution
        # 

        X = XY.iloc[:,:-2].as_matrix()#[:int(len(XY)/2)]
        Y = XY['f-nf'].iloc[:].as_matrix()#[:int(len(XY)/2)]

        classifier = RandomForestClassifier(n_estimators=185)
        classifier.fit(X, Y)


        # save the classifier
        mdlFolder = folder+trainsubjF+"model/"
        if not os.path.exists(mdlFolder):
            os.makedirs(mdlFolder)

        with open(mdlFolder+'RF_185_exact_and_motif_segs_thre'+mthres+'.pkl', 'wb') as fid:
            cPickle.dump(classifier, fid)    





        testsubj = 'test'+subj
        trainsubjF = trainsubj + '(8Hz)/'

        testfeatFolder = '../inlabStr/subject/' + testsubj + '(8Hz)/'+'/feature/all_features/'
        testfeatFile = testfeatFolder+'engy_run'+str(run)+'_pred_features.csv'
        df_all = pd.read_csv(testfeatFile)

        labelFile = folder + testsubj + '(8Hz)/'+'segmentation/engy_run'+str(run)+'_pred_label_thre'+mthres+'/seg_labels.csv'
        labelDf = pd.read_csv(labelFile, names = ['label'])


        mdlFolder = folder+trainsubjF+"model/"
        # save the classifier
        with open(mdlFolder+'RF_185_exact_and_motif_segs_thre'+mthres+'.pkl', 'rb') as fid:
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

        prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc,_,y_pred = clf_cm_pickle(classifier, X, Y)
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

    outfolder = '../inlabStr/result/seg_clf/accx_IS2ISseg_personalized/'+subj
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    crossValRes.to_csv( outfolder+'RF_185_exact_and_motif_segs_thre'+mthres+'(109).csv', index = None)