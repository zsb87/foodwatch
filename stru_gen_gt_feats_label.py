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

subjs = ['trainJC']# ['Dzung', 'Cao', 'Shibo', 'Rawan', 'JC', 'Eric', 'Jiapeng']#, 'Matt'

for subj in subjs:

    allfeatDF = pd.DataFrame()

    activities = [
            'Spoon',            #1
            'HandBread',        #2
            'Cup',              #3
            'Chopstick',        #4
            'KnifeFork',        #5
            'Bottle',           #6
            'SaladFork',        #7
            'HandChips',        #8
            'Straw',            #9
            'SmokeMiddle',      #10
            'SmokeThumb',       #11
            'ChinRest',         #12
            'Phone',            #13
            'Mirror',           #14
            'Scratches',        #15
            'Nose',             #16
            'Teeth',            #17
    ]

    subjfolder = subj + '(8Hz)/'
    folder = '../inlabStr/subject/'
    featFoler = folder+subjfolder+"feature/"
    datafile =  folder+subjfolder+"testdata.csv"
    segfolder = folder+subjfolder+"segmentation/"


    # for act_ind in range(17):
    if 1:
        gtFolder = segfolder + 'gt_data/'

        for _, dirnames, filenames in os.walk(gtFolder):
            n_gest = len(filenames)

            if n_gest == 0:
                continue
            else:
                for i in range(n_gest):

                    # read in the raw data file
                    # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                    #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
                    gtActFile = 'gt_gesture_' + str(i) + '.csv'
                    gtActFilePath = gtFolder + gtActFile
                    r_df = pd.read_csv(gtActFilePath)

                    # pass raw data into filter
                    r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x', 'Linear_Accel_y', 'Linear_Accel_z']]
                    r_df = df_iter_flt(r_df)

                    print(r_df)

                    r_df = add_pitch_roll(r_df)

                    # generate the features
                    feat = gen_feat(r_df)

                    print(feat)


                    featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                    # featDF["activity"] = act_ind


                    allfeatDF = pd.concat([allfeatDF,featDF])

    if not os.path.exists(featFoler):
        os.makedirs(featFoler)

    outfile = featFoler + "gt_features.csv"
    allfeatDF.to_csv(outfile, index=None)

