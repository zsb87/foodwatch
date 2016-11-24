import os
import sys
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
sys.path.append('C:/Users/szh702/Documents/FoodWatch/inlabStr/')
from stru_utils import *


def pointwise2headtail(pointwise_rpr):
    pw = pointwise_rpr
    diff = np.concatenate((pw[:-1],np.array([0]))) - np.concatenate((np.array([0]),pw[:-1]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1

    headtail_rpr = np.vstack((ind_head, ind_tail)).T;

    return headtail_rpr


winsize = 4
stride = 2

subjs = ['Will']# 'Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng'



# 
# 
# 
#   split train test set
#
# 
# 



for subj in subjs:

    # subjfolder = subj + '(8Hz)/'
    subjfolder = subj + '/'
    folder = './subject/'
    allfeatDF = pd.DataFrame()

    featFolder = folder+subjfolder+"feature/"
    segFolder = folder+subjfolder+"segmentation/"
    rawDataFile =  folder+subjfolder+"testdata.csv"
    testDataFolder = segFolder+'test_data/'
    energyfolder = folder+subjfolder+"feature/energy/"
    allFeatFolder = folder+subjfolder+"feature/all_features/"    
    lengyfolder = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"

    trainFolder = './subject/train'+subjfolder+"feature/energy/"
    testFolder = './subject/test'+subjfolder+"feature/energy/"
    trainDataFolder = './subject/train'+subjfolder
    testDataFolder = './subject/test'+subjfolder

    for folder in [trainFolder, testFolder,trainDataFolder,testDataFolder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    energyDf = pd.read_csv(lengyfolder)
    dataDf = pd.read_csv(rawDataFile)



    energyDf['ges']  = energyDf['feedingClass'] + energyDf['nonfeedingClass']
    # print(energyDf['ges'])
    engy_arr = energyDf['ges'].as_matrix()

    headtail_rpr = pointwise2headtail(engy_arr)

    # split_in_middle:

    i=0
    headtail_rpr[i,0] = 0
    while i <(shape(headtail_rpr)[0])-1:
    	headtail_rpr[i,1] =int(headtail_rpr[i,1]+headtail_rpr[i+1,0])/2
    	headtail_rpr[i+1,0] = headtail_rpr[i,1]
    	i = i + 1
    headtail_rpr[i,1] = len(energyDf)


    [train_set, test_set]= tt_split(headtail_rpr, 0.7)

    train_set= train_set[train_set[:,0].argsort()]
    test_set= test_set[test_set[:,0].argsort()]




    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, energyDf.iloc[train_set[i,0]:train_set[i,1]]])

    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, energyDf.iloc[test_set[i,0]:test_set[i,1]]])

    print(trainDf)

    trainDf = trainDf.drop('Unnamed: 0',1)
    testDf = testDf.drop('Unnamed: 0',1)

    trainDf.to_csv(trainFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")
    testDf.to_csv(testFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")




    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, dataDf.iloc[2*train_set[i,0]:2*train_set[i,1]]])

    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, dataDf.iloc[2*test_set[i,0]:2*test_set[i,1]]])


    trainDf.to_csv(trainDataFolder+"testdata.csv", index = None)
    testDf.to_csv(testDataFolder+"testdata.csv", index = None)



# 
# 
# 
#   label train test set raw data from energy label 
#
#
# 


if 1:

    for subj in subjs:
        trainsubj = 'train'+subj

        # trainsubjF = trainsubj + '(8Hz)/'

        trainsubjF = trainsubj + '/'

        folder = '../inlabStr/subject/'
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

        # trainsubjF = trainsubj + '(8Hz)/'

        trainsubjF = trainsubj + '/'
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



