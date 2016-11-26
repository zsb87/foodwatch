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
import sys
sys.path.append('C:/Users/szh702/Documents/FoodWatch/inlabStr/')
from stru_utils import *


def read_r_df_test(subj, file, birthtime, deadtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = ((r_df.index > birthtime) & (r_df.index < deadtime))
    r_df_test = r_df.loc[mask]
    
    return r_df_test

def read_r_df_test_st(subj, file, birthtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = (r_df.index > birthtime)
    r_df_test = r_df.loc[mask]
    
    return r_df_test


def rm_black_out(r_df_test, annotDf):
    annot_blac = annotDf.loc[(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]

    StartTime_list = list(annot_blac.StartTime.tolist())
    EndTime_list = list(annot_blac.EndTime.tolist())

    for n in range(len(StartTime_list)):
        r_df_test = r_df_test[(r_df_test.index < str(StartTime_list[n]))|(r_df_test.index > str(EndTime_list[n]))]

    return r_df_test


def importAnnoFile(annot_file):
    annotDf = pd.read_csv(annot_file, encoding = "ISO-8859-1")
    # print(annotDf['StartTime'])
    annotDf['StartTime'] = pd.to_datetime(annotDf['StartTime'],utc=True)
    annotDf['EndTime'] = pd.to_datetime(annotDf['EndTime'],utc=True)
    return annotDf

def processAnnot(annotDf):
    annotDf['drop'] = 0
    annotDf['duration'] = annotDf['EndTime'] - annotDf['StartTime']
    annotDf['drop'].loc[(annotDf['duration']<'00:00:01.5')&(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]= 1
    annotDf = annotDf.loc[annotDf['drop'] == 0]

    annotDf  = annotDf[['StartTime','EndTime','Annotation','MainCategory']]
    return annotDf

def gen_energy_file(r_df_test, winsize, stride, freq, featsfile):
    i = 0
    allfeatDF = pd.DataFrame()

    while(i+winsize < r_df_test.shape[0]):    
        feat = gen_energy(r_df_test[i:i+winsize], freq)
        featDF = pd.DataFrame(feat[1:] , columns=feat[0])
        i += stride
        if i%100 == 0:
            print(i)
        allfeatDF = pd.concat([allfeatDF,featDF])
        
    if save_flg:
        print("saving energy to csv")
        allfeatDF.to_csv(featsfile)

    return allfeatDF


def checkHandUpDown(annot_HU, annot_HD):
    for i in range(1,max(len(annot_HD),len(annot_HU))):
        if not (annot_HU.StartTime.iloc[i]>annot_HD.StartTime.iloc[i-1]) & (annot_HU.StartTime.iloc[i]<annot_HD.StartTime.iloc[i]):
            print("subj: "+subj)
            print("trouble line: "+str(i))
            print("annot_HandDown.StartTime of "+str(i-1)+" is "+str(annot_HD.StartTime.iloc[i-1]))
            print("annot_HandUp.StartTime of "+str(i)+" is "+str(annot_HU.StartTime.iloc[i]))
            print("annot_HandDown.StartTime of "+str(i)+" is "+str(annot_HD.StartTime.iloc[i]))
            return -1
    return 0



def unstrGenFeedingLabels(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])

    annot_f = annot_f.sort_values(by='StartTime')

    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()
    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/f_HU.csv")
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/f_HD.csv")


    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'feedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def unstrGenNonfeedingLabels_tmp(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])
    
    annot_f = annot_f.sort_values(by='StartTime')


    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()

    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/nf_HU.csv")
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/nf_HD.csv")


    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down not in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'nonfeedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def genWinHandUpHoldingDownLabels(annotDf, r_df_test, activities):
    annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")]
    annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")]

    # for act in activities:
    #     annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")&(annotDf["Annotation"]==act)]
    #     annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")&(annotDf["Annotation"]==act)]
    annot_HD.to_csv("../"+protocol+"/subject/"+subjfolder+"/tmp_HD.csv")
    annot_HU.to_csv("../"+protocol+"/subject/"+subjfolder+"/tmp_HU.csv")

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        exit()
    if len(annot_HU) != len(annot_HD):
        print("hand up and hand down not pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()


    HU_St_list = list(annot_HU.StartTime.tolist())
    HD_Et_list = list(annot_HD.EndTime.tolist())

    UD_dur = []

    for n in range(len(HU_St_list)):
        UD_dur.append([HU_St_list[n],HD_Et_list[n]])

    r_df_test = markClassPeriod( r_df_test,'nonfeedingClass' , UD_dur )


    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test = mark( r_df_test , 'activity', i+1, act_dur )

        print(act_dur)

    return r_df_test


def genFeedingGesture_DrinkingLabels(annotDf, r_df_test, activities):

    # 'feedingClass' = 1 means it is feeding gesture including drinking
    # 'feedingClass' = 0 means it is not feeeding

    # 'activity' implies the activity of the whole period

    annot_feeding = annotDf.loc[annotDf["MainCategory"]=="FeedingGesture"]
    annot_drinking = annotDf.loc[annotDf["MainCategory"]=="Drinking"]

    Feeding_St_list = list(annot_feeding.StartTime.tolist())
    Feeding_Et_list = list(annot_feeding.EndTime.tolist())

    drinking_St_list = list(annot_drinking.StartTime.tolist())
    drinking_Et_list = list(annot_drinking.EndTime.tolist())

    feeding_dur = []
    drinking_dur = []

    for n in range(len(Feeding_St_list)):
        feeding_dur.append([Feeding_St_list[n],Feeding_Et_list[n]])

    for n in range(len(drinking_St_list)):
        drinking_dur.append([drinking_St_list[n],drinking_Et_list[n]])

    r_df_test = markClassPeriod( r_df_test,'feedingClass' , feeding_dur )
    r_df_test = markClassPeriod( r_df_test,'drinkingClass' , drinking_dur )

    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test = mark( r_df_test , 'activity', i+1, act_dur )
        print(act_dur)

    return r_df_test




def mergeFeatsLabels_woAct(allfeatDF, r_df_test_label, lfeatfile):
    
    allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    # while(i+winsize < r_df_test.shape[0]):    
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
        i += stride
        idx += 1


    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.nonfeedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"nonfeedingClass"] = 1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'nonfeedingClass'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF
def mergeFeatsLabels(allfeatDF, r_df_test, activities, lfeatfile):
    
    allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['drinkingClass'] = 0
    allfeatDF['activity'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    # while(i+winsize < r_df_test.shape[0]):    
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0

    while(idx < allfeatDF.shape[0]):    

    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test.drinkingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"drinkingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test.activity[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"activity"] = i_act+1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test.nonfeedingClass[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"nonfeedingClass"] = i_act+1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'drinkingClass', 'activity','nonfeedingClass'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF



def genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile):
    video_sensor_bias_ms = 0
    featDF = pd.read_csv(featsfile)
    featDF.index = range(featDF.shape[0])

    r_df_test['Time'] = r_df_test.index

    birthtime_s = birthtime[:19]

    import time
    
    base_unixtime = time.mktime(datetime.datetime.strptime(birthtime_s,"%Y-%m-%d %H:%M:%S").timetuple())
    base_unixtime = base_unixtime*1000 + video_sensor_bias_ms

    r_df_test["synctime"] = (r_df_test["synctime"] - base_unixtime)/1000
    extr_idx = list(range(0,len(r_df_test)-winsize,stride))
    r_df_tDsample = r_df_test.iloc[extr_idx]

    r_df_tDsample.index = range(len(r_df_tDsample))

    raw_energy = pd.concat([featDF, r_df_tDsample], axis=1)

    if save_flg:
        raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','Class']]
        raw_energy.to_csv(syncfile)


def getTimeError(x, pos):
    if pos == 'tripod':
        return {
                # 10 subjs
            'Dzung': -6,
            'JC': -24,
            'Matt': -2,
            'Jiapeng': 41,
            'Eric': 18,
            'Will': 0, 
            'Shibo': 14,
            'Rawan': 12,
            'Cao': 44,
            'Gleb': -4
        }.get(x, 0)
    if pos == 'desk':
        return {
        # 7 subjs
            'Dzung': 2,
            'JC': 0,
            'Matt': 3,
            'Jiapeng': 0,
            'Eric': 0,
            'Will': 0,
            'Gleb': 2
        }.get(x, 0)

def getFeedingGestures(x):
    return {
        # 10 subjs
        'Dzung': ['Spoon', 'Straw','HandFries','HandChips'],
        'JC': ['Spoon', 'SaladFork','HandSourPatch', 'HandBread', 'HandChips'],
        'Matt': ['HandChips', 'HandBread','HandChips', 'HandBread','Spoon', 'SaladFork', 'Cup', 'Bottle'],
        'Jiapeng': ['HandCracker', 'Popcorn', 'HandChips','Cup', 'LicksFingers', 'Spoon','Cup','SaladFork', 'HandBread', 'SaladSpoon'],
        'Eric': ['HandChips','Cup','Spoon'],
        'Will':  ['popcorn', 'RedFish', 'swedishFish', 'chips', 'EatsFromBag', 'bottle', 'Bottle'], 
        'Shibo': ['Straw', 'HandFries','HandBurger', 'Spoon'],
        'Rawan': ['HandChips'],
        'Cao': ['Cup'],
        'Gleb': ['Bottle','Cup','KnifeFork','Spoon','HandBread','Straw']
        }.get(x, 0)

def getNonfeedingGestures(x):
    return {
        # 10 subjs
        'Dzung': [ 'MovesGlasses', 'Scratches', 'ChinRest', 'Napkin', 'PhoneText','MovesAccessory'],
        'JC': ['Wave', 'Scratches', 'PhoneText', 'ChinRest',   'RaisesBag', 'OutOfSeat', 'MovesAccessory'],
        'Matt': ['AdjustAccessory', 'CombHair', 'ChinRest'],
        'Jiapeng': ['Wave', 'Scratches', 'PhoneText', 'ChinRest', 'Nose', 'SyncSignal', 'PhoneText', 'Phone', 'ChinRest'],
        'Eric': ['Wave', 'AdjustAccessory', 'Scratches', 'Nose', 'nose', 'SyncSignal', 'PhoneText','TextPhone', 'CombHair', 'ChinRest'],
        'Will':  ['MovesGlasses', 'LicksLips',  'WipesMouth'], 
        'Shibo': ['Scratches', 'PhoneText','Nose'],
        'Rawan': ['AdjustAccessory','Scratches','Nose','Wave','PhoneText','HandText','Phone','CombHair','ChinRest'],
        'Cao': ['AdjustAccessory'],
        'Gleb': ['AdjustAccessory','Scratches','Nose','Wave','PhoneText','HandText','Phone','CombHair','ChinRest']
        }.get(x, 0)


# ------------------------------------------------------------------------------
# 
# import raw sensor data
# 
# ------------------------------------------------------------------------------
save_flg = 1

# i_subj = int(sys.argv[1])


subjs = ['JC','Dzung','Will', 'Gleb', 'Shibo','Rawan','Jiapeng', 'Cao', 'Eric']#['JC','Jiapeng','Matt','Rawan', 'Cao']#
freq = 31 # not actually used
winsize = 4
stride = 2

protocol = "inlabHiStr"

for active_participant_counter, subj in enumerate(subjs):
    # if (not (active_participant_counter == 1)):
    if 1:
        subjfolder = subj 

        r_df_test_label = pd.read_csv("../"+protocol+"/subject/" + subjfolder + "/testdata_labeled.csv")


        # ------------------------------------------------------------------------------
        # 
        # generate feature file wo labels
        # 
        # save in engy_ori_win4_str2.csv
        # 
        # ------------------------------------------------------------------------------


        featfolder = "../"+protocol+"/subject/"+subjfolder+"/feature/"
        energyfolder = "../"+protocol+"/subject/"+subjfolder+"/feature/energy/"

        if not os.path.exists(featfolder):
            os.makedirs(featfolder)
        if not os.path.exists(energyfolder):
            os.makedirs(energyfolder)

        featsfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+".csv"
        allfeatDF = gen_energy_file(r_df_test_label, winsize, stride, freq, featsfile)



        # ------------------------------------------------------------------------------
        # 
        # merge label file and feature file
        # 
        # ------------------------------------------------------------------------------
        file = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+".csv"
        allfeatDF = pd.read_csv(file)

        lfeatfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
        # allfeatDF = mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile)
        allfeatDF = mergeFeatsLabels_woAct(allfeatDF, r_df_test_label, lfeatfile)

        # ------------------------------------------------------------------------------
        # 
        # import feature data
        # 
        # ------------------------------------------------------------------------------

        # featsfile = featfolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
        # syncfile = featfolder + "raw_engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled_time.csv"

        # # featsfile = featfolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
        # # syncfile = featfolder + "raw_engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled_time.csv"

        # genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile)










def pointwise2headtail(pointwise_rpr):
    pw = pointwise_rpr
    diff = np.concatenate((pw[:-1],np.array([0]))) - np.concatenate((np.array([0]),pw[:-1]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1

    headtail_rpr = np.vstack((ind_head, ind_tail)).T;

    return headtail_rpr


folder = "../"+protocol+"/subject/"

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
    allfeatDF = pd.DataFrame()

    featFolder = folder+subjfolder+"feature/"
    segFolder = folder+subjfolder+"segmentation/"
    rawDataFile =  folder+subjfolder+"testdata_labeled.csv"
    testDataFolder = segFolder+'test_data/'
    energyfolder = folder+subjfolder+"feature/energy/"
    allFeatFolder = folder+subjfolder+"feature/all_features/"    
    lengyfolder = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"

    trainFolder = folder + 'train'+subjfolder+"feature/energy/"
    testFolder = folder + 'test'+subjfolder+"feature/energy/"
    trainDataFolder = folder + 'train'+subjfolder
    testDataFolder = folder + 'test'+subjfolder

    for subfolder in [trainFolder, testFolder,trainDataFolder,testDataFolder]:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    
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


    trainDf.to_csv(trainDataFolder+"testdata_labeled.csv", index = None)
    testDf.to_csv(testDataFolder+"testdata_labeled.csv", index = None)



# # 
# # 
# # 
# #   label train test set raw data from energy label 
# #
# #
# # 


# if 1:

#     for subj in subjs:
#         trainsubj = 'train'+subj

#         # trainsubjF = trainsubj + '(8Hz)/'

#         trainsubjF = trainsubj + '/'

        
#         featFolder = folder+trainsubjF+"feature/all_features/"
#         engyFolder = folder+trainsubjF+"feature/energy/"
#         segfoldder = folder+trainsubjF+"segmentation/"

        
#           # build model 
        

#         dataFile = folder+trainsubjF+'testdata.csv'     
#         engyFile = engyFolder+'engy_ori_win4_str2_labeled.csv'     
#         edf = pd.read_csv(engyFile)



#         fC = edf['feedingClass'].as_matrix().reshape(len(edf),1)
#         print(fC.shape)

#         dataDf = pd.read_csv(dataFile)
#         fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

#         if len(dataDf) > len(edf)*2:
#             fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
#             print(' len(dataDf) > len(edf) ')
#         elif  len(dataDf) < len(edf)*2:
#             fC = fC[:len(dataDf)]

#         print(fC.shape)
#         print(len(dataDf))

#         dataDf['feedingClass'] = fC

#         dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)


#         dataFile = folder+trainsubjF+'testdata_labeled.csv'   
#         fC = edf['nonfeedingClass'].as_matrix().reshape(len(edf),1)
#         print(fC.shape)

#         dataDf = pd.read_csv(dataFile)
#         fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

#         if len(dataDf) > len(edf)*2:
#             fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
#             print(' len(dataDf) > len(edf) ')
#         elif  len(dataDf) < len(edf)*2:
#             fC = fC[:len(dataDf)]

#         print(fC.shape)
#         print(len(dataDf))

#         dataDf['nonfeedingClass'] = fC



#         dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)



# # 
# # 
# #       repeat
# # 


#         trainsubj = 'test'+subj

#         # trainsubjF = trainsubj + '(8Hz)/'

#         trainsubjF = trainsubj + '/'
#         featFolder = folder+trainsubjF+"feature/all_features/"
#         engyFolder = folder+trainsubjF+"feature/energy/"
#         segfoldder = folder+trainsubjF+"segmentation/"

        
#           # build model 
        

#         dataFile = folder+trainsubjF+'testdata.csv'     
#         engyFile = engyFolder+'engy_ori_win4_str2_labeled.csv'     
#         edf = pd.read_csv(engyFile)



#         fC = edf['feedingClass'].as_matrix().reshape(len(edf),1)
#         print(fC.shape)

#         dataDf = pd.read_csv(dataFile)
#         fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

#         if len(dataDf) > len(edf)*2:
#             fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
#             print(' len(dataDf) > len(edf) ')
#         elif  len(dataDf) < len(edf)*2:
#             fC = fC[:len(dataDf)]

#         print(fC.shape)
#         print(len(dataDf))

#         dataDf['feedingClass'] = fC

#         dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)


#         dataFile = folder+trainsubjF+'testdata_labeled.csv'   
#         fC = edf['nonfeedingClass'].as_matrix().reshape(len(edf),1)
#         print(fC.shape)

#         dataDf = pd.read_csv(dataFile)
#         fC = np.hstack((fC,fC)).reshape(2*len(edf),1)

#         if len(dataDf) > len(edf)*2:
#             fC= np.vstack((fC,np.zeros(shape=(len(dataDf)-len(edf)*2,1))))
#             print(' len(dataDf) > len(edf) ')
#         elif  len(dataDf) < len(edf)*2:
#             fC = fC[:len(dataDf)]

#         print(fC.shape)
#         print(len(dataDf))

#         dataDf['nonfeedingClass'] = fC



#         dataDf.to_csv(folder+trainsubjF+'testdata_labeled.csv',index = None)



