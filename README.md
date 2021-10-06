# VR-Emotion-Arousal
基于VR的情感唤起

# How to update file bigger than 25mb
#install lfs
$brew install git-lfs </br>
or </br>
install at https://git-lfs.github.com </br>

#cd to your workpath </br>
$cd YOUR_DIR_PATH

#clone my respo to local </br>
$git clone git@github.com:blue-sky-sea/VR-Emotion-Arousal.git

#add your new file which is bigger than 25mb </br>
$git add /Users/liuyi/Desktop/VR-Emotion-Arousal/Data/09-24/cuidewen-09-24/rawdata/museMonitor_2021-09-24--0.csv.zip

#change your file </br>
change your file in local respo

#commit with command </br>
$git commit -m "test lfs"

#push your file from origin to main(the brach in my remote respo) </br>
$git push -u origin main
