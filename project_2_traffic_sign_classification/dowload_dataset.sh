#!/bin/bash
mkdir -p data/train
mkdir -p data/test
echo 'Training dataset.'
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip -d data/train
echo 'Training part success.'
echo ''
echo 'Testing dataset.'
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_Images.zip -d data/test
echo 'Testing part success.'

