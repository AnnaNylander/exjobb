#!/bin/bash
####################################
#
# Record data in carla_settings
#
####################################
path_base=/media/annaochjacob/crucial/recorded_data/carla/
cd ~/Repos/carla/PythonClient/
source activate exenv

# Record sessions lasting one hour each
n_frames=36000
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n mango
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n banana
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n watermelon
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n kiwi
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n strawberry
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n carambola
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n litchi
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n rambutan
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n pitahaya
python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames $n_frames --save-path $path_base -n durian
