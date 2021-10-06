# VR-Emotion-Arousal
基于VR的情感唤起

# How to update file bigger than 25mb

#cd to your workpath
$cd YOUR_DIR_PATH

#clone my respo to local
$git clone git@github.com:blue-sky-sea/VR-Emotion-Arousal.git

#add your new file which is bigger than 25mb
$git add /Users/liuyi/Desktop/VR-Emotion-Arousal/Data/09-24/cuidewen-09-24/rawdata/museMonitor_2021-09-24--0.csv.zip

#change your file
change your file in local respo

#commit with command
$git commit -m "test lfs"

#push your file from origin to main(the brach in my remote respo)
$git push -u origin main
