========================================================================

# VR-Emotion-Arousal
基于VR的情感唤起

========================================================================

![image](https://user-images.githubusercontent.com/26008298/132282618-0440b99c-af47-4e75-9c45-2253ba94f59d.png)

========================================================================

| author | mizukiyuta | <br />   
| department | Tokyo Metropolitan University System Design |  <br />

========================================================================
# data
ECG
EEG
self-report

# Experiment
One-person in VR relax，relax，sad，fear，disgust，joy

Two-person in VR relax，sad，Fear，Joy  meeting in VRworkroom


# Personal difference
By seeing the Distribution of data, We found some featrues may have personal differences
we found fat people have a less pnn20,pnn50(one of them have a nearly 0 pnn50 for all sections)
we found a collaborator have a much higher pnn20,pnn50 in relax and sad mode (about 8,normal people are about 1~3)  
and higher pnn50 in Fear,Disgust (about 15,normal people are about 8~10)


# algorithm
SVM KNN Naive Bayes  
CNN LSTM CNN-LSTM  

# How to update file bigger than 25mb
#install lfs
$brew install git-lfs </br>
or </br>
install at https://git-lfs.github.com </br>

#cd to your workpath </br>
$cd YOUR_DIR_PATH

#clone my respo to local </br>
$git clone git@github.com:blue-sky-sea/VR-Emotion-Arousal.git

#cd to your project</br>
$cd YOUR_PROJECT_PATH

#add your new file which is bigger than 25mb </br>
$git add /Users/liuyi/Desktop/VR-Emotion-Arousal/Data/09-24/cuidewen-09-24/rawdata/museMonitor_2021-09-24--0.csv.zip

#change your file </br>
change your file in local respo

#commit with command </br>
$git commit -m "test lfs"

#push your file from origin to main(the brach in my remote respo) </br>
$git push -u origin main
