import os
import re
import csv
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn import preprocessing
from scipy import stats
from scipy import *
from scipy.stats import *
from scipy.signal import *
from sklearn.metrics import matthews_corrcoef
import numpy.polynomial.polynomial as poly
from stru_utils import *

    
# activities = ["answerPhone"]
# eating = ["chips", "chopsticks", "fork", "knifeFork", "pizza", "soupSpoon"]
# drinking = ["bottle", "cup", "drinkStraw"]
# smoking = ["smoke_im", "smoke_ti"]
# others = ["answerPhone", "brushTeeth", "cheek", "comb", "forehead",
#         "nose", "restNearMouth"]

subjs = ['Rawan','Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric']#

for active_participant_counter, subj in enumerate(subjs):
    if (not (active_participant_counter == 0)) and(not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)):
        print(active_participant_counter)

        allfeatDF = pd.DataFrame()

        subjfolder = subj + '(8Hz)/'
        folder = '../inlabStr/subject/'
        featFoler = folder+subjfolder+"feature/"
        datafile =  folder+subjfolder+"testdata.csv"
        segfolder = folder+subjfolder+"segmentation/"
        gtFolder = segfolder+'activity/all_pred/'

        for _, dirnames, filenames in os.walk(gtFolder):
            n_gest = len(filenames)

            if n_gest == 0:
                print("no prediction file for current subject")
                continue
            else:
                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    gtActFile = 'pred_gesture_' + str(i) + '.csv'
                    r_df = pd.read_csv(gtFolder+gtActFile)

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                    r_df = df_iter_flt(r_df)
                    r_df = add_pitch_roll(r_df)
                    # generate the features
                    feat = gen_feat(r_df)

                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    allfeatDF = pd.concat([allfeatDF,featDF])

    # if not os.path.exists(featFoler):
    #     os.makedirs(featFoler)

        outfile = featFoler + "pred_features_reduced.csv"
        allfeatDF.to_csv(outfile)

