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



save_flg = 1

subjs = ['Rawan','Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric']#

for active_participant_counter, subj in enumerate(subjs):
    if (not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)):

        subjfolder = subj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        datafile =  folder+subjfolder+"testdata.csv"
        segfolder = folder+subjfolder+"segmentation/"
        act_rootfolder = segfolder+'activity/'

        saveFileFolder = act_rootfolder + 'all_pred/'
        if not os.path.exists(saveFileFolder):
            os.makedirs(saveFileFolder)

        # read in the raw data file
        # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
        #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
        r_df = pd.read_csv(datafile)
        
        predActFilePath = act_rootfolder + 'pred_headtail_reduced_all.csv'
        if os.path.exists(predActFilePath):
            gt_headtail = pd.read_csv(predActFilePath, names = ['EnergyStart','EnergyEnd','EnergyDur','0'])
                
            gt_headtail['Start'] = gt_headtail['EnergyStart']*2
            gt_headtail['End'] = gt_headtail['EnergyEnd']*2 + 3

            for i in range(len(gt_headtail)):

                saveFileName = 'pred_gesture_' + str(i) + '.csv'
                saveFilePath = saveFileFolder + saveFileName

                dataStart = int(gt_headtail['Start'].iloc[i])
                dataEnd = int(gt_headtail['End'].iloc[i])

                r_df_gesture = r_df.iloc[dataStart:dataEnd]
                r_df_gesture.to_csv(saveFilePath)

