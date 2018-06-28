#!/bin/bash

cd ~/Repos/exjobb/

EPOCHS=2
PLOT_FREQ=5000
PRINT_FREQ=10
BATCH_SIZE=12
DATASET=flexible/
N_DRIVE_FRAMES=100 # The number of frames that the car should drive in CARLA

# These values are to be set for every network
FOLDER_NAME=test_deluxe/cluster_1_100_cl_7_pc/
NETWORK_FILE=cluster_1
NETWORK_CLASS=ClusterNet
CLUSTER_PATH='/home/annaochjacob/Repos/exjobb/preprocessing/path_clustering/'

echo "*** Evaluating network on test set in order to generate predictions ***"
#python3 network/main.py \
python3 network/main_clusters.py \
                        -ncl 100 \
                        -npc 7 \
                        -cpath ${CLUSTER_PATH}100_clusters/ \
                        -e $EPOCHS \
                        -b $BATCH_SIZE \
                        -pl $PLOT_FREQ \
                        -p $PRINT_FREQ \
                        -a ${NETWORK_FILE}.${NETWORK_CLASS} \
                        -d $DATASET \
                        -s ${FOLDER_NAME} \
                        -ipf '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 20 30' \
                        -off '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30' \
                        --resume ${FOLDER_NAME}all_time_best.pt \
                        --evaluate


echo "*** Saving training and validation loss plot ***"
python3 plot_network_loss.py -m ${FOLDER_NAME}

echo "*** Saving visualizations of the generated predictions ***"
python3 save_visualization.py --prediction -m ${FOLDER_NAME} -r test/ -d ${DATASET}test/

#echo "*** Copying the network into carla repo so it can be used for driving ***"
#cp ./network/architectures/${NETWORK_FILE}.py ./carla/PythonClient/

#echo "*** Starting client which uses the specified network to drive ***"
#cd ~/Repos/exjobb/carla/PythonClient
#python3 client_test_drive.py -i -c CarlaSettings.ini -f $N_DRIVE_FRAMES \
#                              -s evaluations/ \
#                              -m /media/annaochjacob/crucial/models/${FOLDER_NAME}/checkpoint.pt \
#                              -n ${FOLDER_NAME}

#echo "*** Starting CARLA server (in new shell) to let the network drive in ***"
#cd carla/Unreal/CarlaUE4/carla_long_gear/LinuxNoEditor
#./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -windowed -ResX=400 -ResY=300 &
