wget -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/2
wget -O Gesture.zip https://figshare.com/ndownloader/articles/19930247/versions/1

unzip SleepEEG.zip -d datasets/SleepEEG/
unzip  Epilepsy.zip -d datasets/Epilepsy/
unzip  Gesture.zip -d datasets/Gesture/

rm {SleepEEG,Epilepsy,Gesture}.zip

cp -r datasets/* TS-TCC/data/