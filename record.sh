#!/bin/bash
####################################
#
# Record data in carla_settings
#
####################################

cd ~/Repos/carla/PythonClient/
source activate exenv

python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 50 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n mango
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 50 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n banana
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 100 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n watermelon
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 100 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n kiwi
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n strawberry
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n carambola
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n litchi
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n rambutan
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n pitahaya
#python3 client_2.py -a -i -q Epic -c CarlaSettings.ini --frames 35000 --save-path /media/annaochjacob/crucial/recorded_data/carla/ -n durian
