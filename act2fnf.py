import os
import re
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import random

subjs = ['Rawan','Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric']#

if 1:
    for active_participant_counter, subj in enumerate(subjs):
        if (not (active_participant_counter == 3)) and (not (active_participant_counter == 4))  and (not (active_participant_counter == 6)):
            print(subj)
            
            subjfolder = subj + '(8Hz)/'
            folder = '../inlabStr/subject/'
            featFolder = folder+subjfolder+"feature/"
            datafile =  folder+subjfolder+"testdata.csv"
            segfolder = folder+subjfolder+"segmentation/"
            clsfolder = folder+subjfolder+"classification/"
            
            gtFeatPath = featFolder + "gt_features.csv"

            if not os.path.exists(featFolder+"all_features/gt/"):
                os.makedirs(featFolder+"all_features/gt/")

            outFeatFile = featFolder+"all_features/gt/gt_fnf_feats.csv"
            df = pd.read_csv(gtFeatPath)
            print(len(df))
            df = df.dropna() 
            print(len(df))
            
            # add one column
            equiv = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1,7:1, 8:1, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0}
            df["f-nf"] = df["activity"].map(equiv)
            df["subj"] = active_participant_counter
            
            df.to_csv(outFeatFile,index=None)
