#!/Users/tt/Desktop/kaggle_houseprice
# coding: utf-8

from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import sys
#from numpy import *
import numpy
import pandas as pd
from os import listdir


def drop_feature(data):
    df=data
    """to_drop =['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT',
           'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X',
           'Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
           'HSI_TP10', 'Battery', 'Elements']"""
    to_drop =['HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2',
           'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_ULF','HRV_VLF','Time','TimeStamp']
    # 丢弃特征 drop columns
    df.drop(to_drop, axis=1, inplace=True)
    #print("feature droped!")
    return df
def drop_feature2(data):
    df=data
    to_drop =['Date','StartTime','EndTime','Goal','world',
    'Unhappy-Happy','Calm-Excited','Controlled-Incontrol']
    df.drop(to_drop, axis=1, inplace=True)
    return df

def loadEmotion(dir_path):
    dirList = listdir(dir_path)
    print(dirList)
    data=[]
    label=[]
    my_matrix=[]
    for dir_name in dirList:
        if(dir_name==".DS_Store"):
            print("DS_Store!PASS")
            continue
        else:
             data_path=dir_path + dir_name
             print(data_path)
             sr_file_name = dir_name+"SR.csv" #cuidenwen-09-24SR.csv
             sr_df =  pd.read_csv('%s/%s' % (data_path, sr_file_name) )#SR.csv
             #sr_df = drop_feature2(sr_df)
             print(sr_df)
             #input()
             for i in range(6):
                data_output_path=data_path+"/output2"
                data_file_name="data"+str(i)+".csv"
                data_df =  pd.read_csv('%s/%s' % (data_output_path, data_file_name) )#data.csv
                data_df = drop_feature(data_df)
                #data_df = data_df[['HRV_pNN50','HRV_LFHF','Attention','Meditation','A_M']]
                #data_df = data_df[['HRV_pNN50','Attention','Meditation','A_M']]
                data_df = data_df[["HRV_pNN50","HRV_LFHF",
                "Accelerometer_X","Accelerometer_Y","Accelerometer_Z",
                "A_M"]]
                #print(data_df)
                data_=[]
                if(len(data_df)>=60):
                    for j in range(60):
                        data_.extend(data_df.iloc[j].values)
                else:
                    continue

                #sr_df
                #Happy Unhappy two class
                Happy_score = float(sr_df.iloc[i]["Relax"]*0.6+sr_df.iloc[i]["Excited"]*0.6 + sr_df.iloc[i]["Joy"]*1.2-sr_df.iloc[i]["Sadness"])/3
                UnHappy_score = float(sr_df.iloc[i]["Fear"] + sr_df.iloc[i]["Disgust"] + sr_df.iloc[i]["Sadness"]+sr_df.iloc[i]["Excited"]*0.4)/6
                Relax_score = sr_df.iloc[i]["Relax"]*1.0
                if(Happy_score >  UnHappy_score ):
                    label_=0
                elif(UnHappy_score >= Happy_score):
                    label_=1
                print(label_,"-->",Happy_score,UnHappy_score)
                data_.extend([label_])
                #print(data_)
                my_matrix.append(data_)
        
    import numpy as np
    from sklearn.model_selection import train_test_split
    my_matrix=np.array(my_matrix)



    #print(len(my_matrix))
    #print(my_matrix)
    X, y = my_matrix[:,:-1],my_matrix[:,-1]

 
    return X,y

#import our data
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)
print(y)

#------------------------------------
#my code at 2021.10.27
#------------------------------------
X, y = loadEmotion("/Users/liuyi/Desktop/Bio-ML/train/")
print(X)
print(X.shape)
#input()
print(y)


X=numpy.nan_to_num(X)


x_normed = X / X.max(axis=0)
X = x_normed
#------------------------------------




#split the data to  7:3
X_train,X_test,y_train,y_test = ts(X,y,test_size=0.3)

# select different type of kernel function and compare the score

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
print("The score of poly is : %f"%score_poly)

