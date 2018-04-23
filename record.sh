#!/bin/bash
####################################
#
# Record data in carla_settings
#
####################################
path_base=/media/annaochjacob/crucial/recorded_data/carla/test_and_val/
cd ~/Repos/carla/PythonClient/
source activate exenv

# Record sessions lasting one hour each
n_frames=9000 #30 min
settings=CarlaSettings.ini

# Test and validation set in town 2
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n panda
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n jaguar
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n fox
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n elephant
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n giraffe
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n frog
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n koala
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n lion
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n tiger
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n puma
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n rhino
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n moose
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n cat
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n dog
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n mouse
python3 client_collect_data.py -a -i -c $settings --frames $n_frames --save-path $path_base -n bird


#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n mango
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n banana
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n watermelon
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n kiwi
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n strawberry
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n carambola
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n litchi
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n rambutan
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pitahaya
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n durian
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n apple
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n orange
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pear
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n raspberry
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n papaya
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pineapple
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n peach
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n jackfruit
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n lemon
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n pomegranate

# These are recorded but not annotated:
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n apricot
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n mangostan
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n kirimoja
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n avocado
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n taggannona
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n cantaloupe
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n honeydew
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n charentais
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n hami
#python3 client_2.py -a -i -c $settings --frames $n_frames --save-path $path_base -n apollo
