import os
import numpy as np
import pandas as pd
import re
import sys
import csv

# concat
subj = 'Gleb'

walk_dir = '../'+subj+'/right/Gyroscope/'
data_file = '../'+subj+'/right/data.csv'

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(walk_dir) for f in filenames if os.path.splitext(f)[1] == '.csv']
df_list = [pd.read_csv(file) for file in result]
final_df = pd.concat(df_list)
final_df= final_df.sort_values('Time')

print(subj)
final_df.to_csv(data_file)


# # d sample
# print(len(final_df))
# l = range(0,len(final_df),4)
# final_df = final_df.iloc[l,:]

# dsamplefolder = "../"+subj+"(8Hz)/right/"
# if not os.path.exists(dsamplefolder):
#     os.makedirs(dsamplefolder)

# file = dsamplefolder+"data.csv"
# final_df.to_csv(file, index =None)