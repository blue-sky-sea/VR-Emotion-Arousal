import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


def readcsv(filepath):
	df = pd.read_csv(filepath)
	return df


path_root ="./cuidewen-09-24"
rawdata_path = "/rawdata/"
maturedata_path ="/maturedata/"
eegfile_name = "museMonitor_2021-09-24--14-45-33"+".csv"
srfile_name = "cuidewen-09-24SR"+".csv"
date_stamp="20210924"


def drop_feature(data):
	df=data
	to_drop =['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT',
	       'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X',
	       'Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
	       'HSI_TP10', 'Battery', 'Elements']
	# 丢弃特征 drop columns
	df.drop(to_drop, axis=1, inplace=True)
	print("feature droped!")
	return df

def drop_na(data):
	df=data
	df.dropna(axis=0, how='any', inplace=True)
	# 因为删除了几行数据,所以index的序列不再连续,需要重新reindex
	df.reset_index(drop=True, inplace=True)
	return df

def main():
	#read data
	eeg_rawdata = readcsv(path_root+rawdata_path+eegfile_name)
	sr_data = readcsv(path_root+"/"+srfile_name)


	##------------------------------------
	##preprocessing
	##------------------------------------
	##drop feature
	eeg_rawdata = drop_feature(eeg_rawdata)
	##drop na data
	eeg_rawdata = drop_na(eeg_rawdata)
	##------------------------------------
	

	print(type(eeg_rawdata['TimeStamp'][0]),eeg_rawdata['TimeStamp'][0])



	for i in range(len(sr_data)):
		#print(sr_data.iloc[i])
		starttime = str(sr_data['StartTime'].iloc[i])
		endtime = str(sr_data['EndTime'].iloc[i])
		starttime_stamp = date_stamp + starttime.replace(':','').strip('PM').strip('AM')
		endtime_stamp = date_stamp + endtime.replace(':','').strip('PM').strip('AM')
		print(starttime_stamp ,endtime_stamp)

		eeg_rawdata['TimeStamp'] = pd.to_datetime(eeg_rawdata['TimeStamp'])
		data = eeg_rawdata[(eeg_rawdata['TimeStamp'] >=starttime_stamp) & (eeg_rawdata['TimeStamp'] <= endtime_stamp)]


		##------------------------------------
		##save data to csv
		##------------------------------------
		save_filePath = path_root+maturedata_path+"eeg"+str(i)+".csv"
		data.to_csv(save_filePath,index=0)
		print("saved to ",save_filePath)
		##------------------------------------
main()




