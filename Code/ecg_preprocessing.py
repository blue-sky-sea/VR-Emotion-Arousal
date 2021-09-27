import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

def readcsv(filepath):
	df = pd.read_csv(filepath)
	return df

def read_concat_csv(filepath,n,file_format):
	#df = None
	frames = []
	for i in range(n):
		#print(i)
		df = readcsv(filepath+str(i)+file_format)
		frames.append(df)
	
	dataframe = pd.concat(frames)

	return dataframe


path_root ="./chenyongjie-09-27"
rawdata_path = "/rawdata/"
maturedata_path ="/output/"
ecgfile_name = "09-27df_ecg_polar"
srfile_name = "chenyongjie-09-27SR"
file_format=".csv"
date_stamp="20210927"
n=9

def drop_na(data):
	df=data
	df.dropna(axis=0, how='any', inplace=True)
	# 因为删除了几行数据,所以index的序列不再连续,需要重新reindex
	df.reset_index(drop=True, inplace=True)
	return df

def main():
	#read data

	filepath = path_root+rawdata_path+ecgfile_name
	ecg_rawdata = read_concat_csv(filepath,n,file_format)
	sr_data = readcsv(path_root+"/"+srfile_name+file_format)

	#print(ecg_rawdata)
	##------------------------------------
	##preprocessing
	##------------------------------------
	##drop feature
	#ecg_rawdata = drop_feature(ecg_rawdata)
	##drop na data
	ecg_rawdata = drop_na(ecg_rawdata)
	##------------------------------------
	

	for i in range(len(sr_data)):
		#print(sr_data.iloc[i])
		starttime = str(sr_data['StartTime'].iloc[i])
		endtime = str(sr_data['EndTime'].iloc[i])
		starttime_stamp = date_stamp + starttime.replace(':','').strip('PM').strip('AM')
		endtime_stamp = date_stamp + endtime.replace(':','').strip('PM').strip('AM')
		print(starttime_stamp ,endtime_stamp)

		ecg_rawdata['Time'] = pd.to_datetime(ecg_rawdata['Time'])
		data = ecg_rawdata[(ecg_rawdata['Time'] >=starttime_stamp) & (ecg_rawdata['Time'] <= endtime_stamp)]


		##------------------------------------
		##save data to csv
		##------------------------------------
		save_filePath = path_root+maturedata_path+"ecg"+str(i)+".csv"
		data.to_csv(save_filePath,index=0)
		print("saved to ",save_filePath)
		##------------------------------------
main()
