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

subjs = ['Rawan']# 'Matt',#'Rawan','Eric', ,['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']# 'Matt',

if 1:

    for subj in subjs:
        trainsubj = 'train'+subj


        trainsubjF = trainsubj + '(8Hz)/'
        folder = '../inlabUnstr/subject/'
        featFolder = folder+trainsubjF+"feature/all_features/"
        engyFolder = folder+trainsubjF+"feature/energy/"
        segfoldder = folder+trainsubjF+"segmentation/"

        
          # build model 
        

        dataFile = folder+trainsubjF+'testdata.csv'     
        engyFile = engyFolder+'engy_ori_win4_str2_labeled.csv'     
        edf = pd.read_csv(engyFile)



        fC = edf['feedingClass'].as_matrix().reshape(len(edf),1)
        print(fC.shape)

        dataDf = pd.read_csv(dataFile)
        fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

        if len(dataDf) > len(edf)*2:
            fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
            print(' len(dataDf) > len(edf) ')
        elif  len(dataDf) < len(edf)*2:
            fC = fC[:len(dataDf)]

        print(fC.shape)
        print(len(dataDf))

        dataDf['feedingClass'] = fC

        dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)


        dataFile = folder+trainsubjF+'testdata_labeled.csv'   
        fC = edf['nonfeedingClass'].as_matrix().reshape(len(edf),1)
        print(fC.shape)

        dataDf = pd.read_csv(dataFile)
        fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

        if len(dataDf) > len(edf)*2:
            fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
            print(' len(dataDf) > len(edf) ')
        elif  len(dataDf) < len(edf)*2:
            fC = fC[:len(dataDf)]

        print(fC.shape)
        print(len(dataDf))

        dataDf['nonfeedingClass'] = fC



        dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)



# 
# 
#       repeat
# 


        trainsubj = 'test'+subj


        trainsubjF = trainsubj + '(8Hz)/'
        featFolder = folder+trainsubjF+"feature/all_features/"
        engyFolder = folder+trainsubjF+"feature/energy/"
        segfoldder = folder+trainsubjF+"segmentation/"

        
          # build model 
        

        dataFile = folder+trainsubjF+'testdata.csv'     
        engyFile = engyFolder+'engy_ori_win4_str2_labeled.csv'     
        edf = pd.read_csv(engyFile)



        fC = edf['feedingClass'].as_matrix().reshape(len(edf),1)
        print(fC.shape)

        dataDf = pd.read_csv(dataFile)
        fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

        if len(dataDf) > len(edf)*2:
            fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
            print(' len(dataDf) > len(edf) ')
        elif  len(dataDf) < len(edf)*2:
            fC = fC[:len(dataDf)]

        print(fC.shape)
        print(len(dataDf))

        dataDf['feedingClass'] = fC

        dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)


        dataFile = folder+trainsubjF+'testdata_labeled.csv'   
        fC = edf['nonfeedingClass'].as_matrix().reshape(len(edf),1)
        print(fC.shape)

        dataDf = pd.read_csv(dataFile)
        fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

        if len(dataDf) > len(edf)*2:
            fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
            print(' len(dataDf) > len(edf) ')
        elif  len(dataDf) < len(edf)*2:
            fC = fC[:len(dataDf)]

        print(fC.shape)
        print(len(dataDf))

        dataDf['nonfeedingClass'] = fC



        dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)





