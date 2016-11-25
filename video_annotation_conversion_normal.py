subj = "Gleb"

# position = 'desk'
position = 'tripod'

subjfreq = subj
###########################
path = "../inlabUnstr/subject/"+subjfreq+"/annotation/"
chrono_annotations = 'annotations'+subj+'-'+position+'.csv' #orginal anotation file from Chronoviz
birth = '../'+position+' video birth.txt' #The video birth
output= 'annotations-edited-'+position+'.csv' # the output file name 
###########################


from datetime import datetime, timedelta 
import pandas as pd
import numpy as np
import time


chrono = pd.read_csv(path+ chrono_annotations, encoding = "ISO-8859-1")
for i in range(len(chrono)):
    try:
        chrono.StartTime.iloc[i] = pd.to_datetime(chrono.StartTime.iloc[i], format='%M:%S.%f')
        chrono.EndTime.iloc[i] = pd.to_datetime(chrono.EndTime.iloc[i], format='%M:%S.%f')

    except:
        chrono.StartTime.iloc[i] = pd.to_datetime(chrono.StartTime.iloc[i], format='%H:%M:%S.%f')
        chrono.EndTime.iloc[i] = pd.to_datetime(chrono.EndTime.iloc[i], format='%H:%M:%S.%f')
    
videoBirth = pd.read_csv(path + birth, header = None)
videoBirth = pd.to_datetime(videoBirth[0][0])
print(videoBirth)
chrono = chrono.sort_values(by='StartTime')

chrono = chrono.drop('Title', 1)
chrono = chrono.drop('Category2', 1)
chrono = chrono.drop('Category3', 1)

print(chrono[0:5])

for i in range(chrono.shape[0]): 
#     print(chrono.loc[i,'StartTime'].hour)
    try:
        chrono.StartTime[i] = videoBirth + timedelta(hours= chrono.loc[i,'StartTime'].hour ,minutes= chrono.loc[i,'StartTime'].minute ,seconds= chrono.loc[i,'StartTime'].second ,
                                                  microseconds= chrono.loc[i,'StartTime'].microsecond  )
        chrono.EndTime[i] = videoBirth + timedelta(hours= chrono.loc[i,'EndTime'].hour ,minutes= chrono.loc[i,'EndTime'].minute ,seconds= chrono.loc[i,'EndTime'].second ,
                                                  microseconds= chrono.loc[i,'EndTime'].microsecond  )
    except:
        chrono.StartTime[i] = videoBirth + timedelta(minutes= chrono.loc[i,'StartTime'].minute ,seconds= chrono.loc[i,'StartTime'].second ,
                                                      microseconds= chrono.loc[i,'StartTime'].microsecond  )
        chrono.EndTime[i] = videoBirth + timedelta(minutes= chrono.loc[i,'EndTime'].minute ,seconds= chrono.loc[i,'EndTime'].second ,
                                                  microseconds= chrono.loc[i,'EndTime'].microsecond)
chrono.to_csv(path + output, sep=',')
print(chrono)