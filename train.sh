#!/bin/bash
cd ~/Repos/exjobb/network/

EPOCHS=2
PLOT_FREQ=5000
PRINT_FREQ=10
BATCH_SIZE=16
FOLDER_NAME=curriculum_learning/
DATASET=eukaryote/

# Best network when trained with SGD and scheduler, but now trained with Adam to see the difference (using same lr)
#python3 main.py -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -a cnn_bias.CNNBiasAll -p $PRINT_FREQ -d $DATASET -s ${FOLDER_NAME}CNNBiasAll_2pf/ -pf 2 -fs 2 -bl -o "Adam( model.parameters(), lr=5*1e-6, weight_decay=1e-5, amsgrad=True)" #--resume ${FOLDER_NAME}CNNBiasAll_2pf/checkpoint.pt

# Do some simple curriculim learning, aka. remove all intentions first.
#python3 main.py --no-intention -e $EPOCHS -b $BATCH_SIZE -pl $PLOT_FREQ -a cnn_bias.CNNBiasAll -p $PRINT_FREQ -d $DATASET -s ${FOLDER_NAME}CNNBiasAll_Adam/ -bl -o "Adam( model.parameters(), lr=5*1e-6, weight_decay=1e-5, amsgrad=True)" #--resume ${FOLDER_NAME}CNNBiasAll_2pf/checkpoint.pt
python3 main.py -e 4 -b $BATCH_SIZE -pl $PLOT_FREQ -p $PRINT_FREQ -d $DATASET -s ${FOLDER_NAME}CNNBiasAll_Adam/ -bl --resume ${FOLDER_NAME}CNNBiasAll_Adam/checkpoint.pt
