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
n_frames=18000 #30 min
settings=CarlaSettings.ini

python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n mango
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n banana
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n watermelon
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n kiwi
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n strawberry
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n carambola
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n litchi
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n rambutan
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pitahaya
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n durian
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n apple
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n orange
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pear
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n raspberry
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n papaya
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pineapple
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n peach
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n jackfruit
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n lemon
python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pomegranate
